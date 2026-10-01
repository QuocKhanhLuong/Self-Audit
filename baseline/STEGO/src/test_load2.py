from omegaconf import OmegaConf
import torch
from train_segmentation import LitUnsupervisedSegmenter

cfg = OmegaConf.load("configs/train_config.yml")
try:
    model = LitUnsupervisedSegmenter(cfg.dir_dataset_n_classes, cfg)
    ckpt = torch.load("../saved_models/cityscapes_vit_base_1.ckpt", map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", len(unexpected), "items")
    print("Example unexpected:", unexpected[:3] if unexpected else "[]")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
