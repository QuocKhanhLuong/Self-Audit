import argparse
import os
import json
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from utils import (
    set_logger, fix_seed_for_reproducability, worker_init_fn,
    get_faiss_module, get_init_centroids, module_update_centroids,
    feature_flatten, get_metric_as_conv, compute_negative_euclidean,
    initialize_classifier, freeze_all, collate_eval, collate_train,
    get_dataset, get_transform_params, AverageMeter,
)
from modules import fpn
from medical_metrics import compute_medical_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc', 'mnms'])
    parser.add_argument('--split_manifest', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', '..', 'splits',
                                             'acdc_patient_split_seed42.json'))
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--res', type=int, default=224)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--K_test', type=int, default=4)
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--num_classes', type=int, default=4)
    return parser.parse_args()


def load_checkpoint(args, model, logger):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k] = v
        else:
            new_state_dict['module.' + k] = v

    model.load_state_dict(new_state_dict)
    logger.info('Loaded checkpoint from [{}]'.format(args.checkpoint))

    classifier_sd = checkpoint.get('classifier1_state_dict', None)
    return model, classifier_sd


def predict_all(args, logger, dataloader, model, classifier):
    all_preds = []
    all_labels = []
    all_meta = []

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            has_meta = len(batch) == 4
            if has_meta:
                _, image, label, meta = batch
            else:
                _, image, label = batch
                meta = None

            image = image.to(args.device, non_blocking=True)
            feats = model(image)

            if args.metric_test == 'cosine':
                feats = F.normalize(feats, dim=1, p=2)

            probs = classifier(feats)
            probs = F.interpolate(probs, (args.res, args.res),
                                  mode='bilinear', align_corners=False)
            preds = probs.topk(1, dim=1)[1].squeeze(1).cpu().numpy()

            B = preds.shape[0]
            for b in range(B):
                all_preds.append(preds[b])
                if label is not None:
                    all_labels.append(label[b].numpy())
                if meta is not None:
                    all_meta.append(meta[b])

            if i % 20 == 0:
                logger.info('  Predicting: {}/{}'.format(i, len(dataloader)))

    return all_preds, all_labels, all_meta


def hungarian_remap(preds_flat, labels_flat, K):
    histogram = np.zeros((K, K), dtype=np.float64)
    for pred, gt in zip(preds_flat, labels_flat):
        mask = (gt >= 0) & (gt < K) & (pred >= 0) & (pred < K)
        histogram += np.bincount(
            K * gt[mask].ravel() + pred[mask].ravel(),
            minlength=K * K
        ).reshape(K, K)

    row_ind, col_ind = linear_sum_assignment(histogram.max() - histogram)
    mapping = np.zeros(K, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        mapping[c] = r
    return mapping


def evaluate_medical(args, logger):
    fix_seed_for_reproducability(args.seed)

    args.K_train = args.K_test
    args.pretrain = False
    args.optim_type = 'Adam'
    args.lr = 1e-4
    args.weight_decay = 0
    args.momentum = 0.9
    args.eval_only = True
    args.eval_path = args.checkpoint
    args.restart = False
    args.res1 = args.res
    args.res2 = args.res
    args.equiv = False
    args.augment = False
    args.thing = False
    args.stuff = False
    args.X = 80

    model = fpn.PanopticFPN(args)
    model = nn.DataParallel(model)
    model = model.to(args.device)

    model, classifier_sd = load_checkpoint(args, model, logger)

    classifier = initialize_classifier(args)
    if classifier_sd is not None:
        classifier.load_state_dict(classifier_sd)
        freeze_all(classifier)
    else:
        logger.info('No classifier in checkpoint, will use centroid-based.')

    evalset = get_dataset(args, mode='eval_test')
    evalloader = torch.utils.data.DataLoader(
        evalset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_eval, worker_init_fn=worker_init_fn(args.seed))

    logger.info('Predicting on {} ({} samples)...'.format(args.dataset, len(evalset)))
    all_preds, all_labels, all_meta = predict_all(
        args, logger, evalloader, model, classifier)

    logger.info('Running Hungarian matching...')
    K = args.K_test
    preds_flat = [p.ravel() for p in all_preds]
    labels_flat = [l.ravel() for l in all_labels]
    mapping = hungarian_remap(preds_flat, labels_flat, K)
    logger.info('Cluster-to-class mapping: {}'.format(mapping.tolist()))

    all_preds_remapped = [mapping[p] for p in all_preds]

    volumes = {}
    for i, meta in enumerate(all_meta):
        vn = meta['vol_name']
        if vn not in volumes:
            volumes[vn] = {
                'n_slices': meta['n_slices'],
                'spacing': meta['spacing'],
                'preds': {},
                'labels': {},
            }
        volumes[vn]['preds'][meta['slice_idx']] = all_preds_remapped[i]
        volumes[vn]['labels'][meta['slice_idx']] = all_labels[i]

    results = {}
    all_dice = []
    all_hd95 = []
    all_assd = []

    for vol_name in sorted(volumes.keys()):
        vol_data = volumes[vol_name]
        n_slices = vol_data['n_slices']
        spacing_zyx = vol_data['spacing']
        h, w = list(vol_data['preds'].values())[0].shape

        pred_3d = np.zeros((h, w, n_slices), dtype=np.int64)
        gt_3d = np.zeros((h, w, n_slices), dtype=np.int64)
        for z in range(n_slices):
            pred_3d[:, :, z] = vol_data['preds'][z]
            gt_3d[:, :, z] = vol_data['labels'][z]

        spacing = (spacing_zyx[1], spacing_zyx[2], spacing_zyx[0])

        metrics = compute_medical_metrics(pred_3d, gt_3d, spacing, args.num_classes)
        results[vol_name] = metrics

        fg_mask = ~np.isnan(metrics['dice'][1:])
        if fg_mask.any():
            all_dice.append(metrics['dice'][1:][fg_mask].mean())
        fg_hd = metrics['hd95'][1:]
        valid_hd = fg_hd[~np.isnan(fg_hd) & ~np.isinf(fg_hd)]
        if len(valid_hd) > 0:
            all_hd95.append(valid_hd.mean())
        fg_assd = metrics['assd'][1:]
        valid_assd = fg_assd[~np.isnan(fg_assd) & ~np.isinf(fg_assd)]
        if len(valid_assd) > 0:
            all_assd.append(valid_assd.mean())

        logger.info('  {} | Dice: {} | HD95: {} | ASSD: {}'.format(
            vol_name,
            np.array2string(metrics['dice'], precision=4),
            np.array2string(metrics['hd95'], precision=2),
            np.array2string(metrics['assd'], precision=2)))

    logger.info('\n========== SUMMARY ({}) =========='.format(args.dataset.upper()))
    logger.info('Mean Dice (FG): {:.4f} +/- {:.4f}'.format(
        np.mean(all_dice), np.std(all_dice)))
    logger.info('Mean HD95 (FG): {:.2f} +/- {:.2f}'.format(
        np.mean(all_hd95), np.std(all_hd95)))
    logger.info('Mean ASSD (FG): {:.2f} +/- {:.2f}'.format(
        np.mean(all_assd), np.std(all_assd)))

    class_names = ['Background', 'RV', 'MYO', 'LV']
    for c in range(args.num_classes):
        c_dice = [results[v]['dice'][c] for v in results if not np.isnan(results[v]['dice'][c])]
        c_hd = [results[v]['hd95'][c] for v in results
                if not np.isnan(results[v]['hd95'][c]) and not np.isinf(results[v]['hd95'][c])]
        c_assd = [results[v]['assd'][c] for v in results
                  if not np.isnan(results[v]['assd'][c]) and not np.isinf(results[v]['assd'][c])]
        logger.info('  {}: Dice={:.4f}, HD95={:.2f}, ASSD={:.2f}'.format(
            class_names[c],
            np.mean(c_dice) if c_dice else float('nan'),
            np.mean(c_hd) if c_hd else float('nan'),
            np.mean(c_assd) if c_assd else float('nan')))

    summary = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'mapping': mapping.tolist(),
        'mean_dice_fg': float(np.mean(all_dice)),
        'std_dice_fg': float(np.std(all_dice)),
        'mean_hd95_fg': float(np.mean(all_hd95)),
        'std_hd95_fg': float(np.std(all_hd95)),
        'mean_assd_fg': float(np.mean(all_assd)),
        'std_assd_fg': float(np.std(all_assd)),
        'per_volume': {v: {k: arr.tolist() for k, arr in m.items()}
                       for v, m in results.items()},
    }

    os.makedirs(args.save_root, exist_ok=True)
    out_path = os.path.join(args.save_root, 'eval_{}.json'.format(args.dataset))
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Results saved to {}'.format(out_path))


if __name__ == '__main__':
    args = parse_arguments()

    args.K_train = 4
    args.K_test = 4
    args.pretrain = False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_root, exist_ok=True)
    logger = set_logger(os.path.join(args.save_root, 'eval_{}.log'.format(args.dataset)))

    evaluate_medical(args, logger)
