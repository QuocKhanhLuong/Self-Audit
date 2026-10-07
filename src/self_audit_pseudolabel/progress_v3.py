"""Local live telemetry. Never reads references or changes the training recipe."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVENT_PREFIX = '[V3_METRIC] '
EVENT_SCHEMA = 'self_audit.progress.v1'
CLASS_NAMES = ('bg', 'rv', 'myo', 'lv')


def add_progress_arguments(parser):
    parser.add_argument('--log-every', type=int, default=50,
                        help='emit flushed live JSONL/metrics on first, last and every N batches; 0 disables')
    parser.add_argument('--no-progress', action='store_true', help='disable tqdm bars (metrics still log)')


def progress_bar(*, total, desc, enabled=True, unit='batch'):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        print('[PROGRESS] tqdm unavailable; live text metrics remain enabled', file=sys.stderr, flush=True)
        return None
    # Explicit False keeps progress visible when the orchestrator captures stderr.
    return tqdm(total=total, desc=desc, unit=unit, file=sys.stderr, disable=False,
                dynamic_ncols=True, mininterval=1.0, miniters=1)


def coverage_metrics(target, valid, *, prefix):
    """Count only accepted pseudo pixels; fractions divide by ALL batch pixels.

    UNKNOWN (255) is excluded by valid. These are support/coverage, not accuracy.
    Callers pass detached tensors already computed by the training/export path.
    """
    import torch
    with torch.no_grad():
        counts = torch.bincount(target.detach()[valid.detach()].long(), minlength=4).cpu().tolist()
    total = valid.numel()
    accepted = sum(counts)
    result = {f'{prefix}_total_pixels': total, f'{prefix}_valid_pixels': accepted,
              f'{prefix}_valid_fraction': accepted / total if total else 0.0,
              f'{prefix}_foreground_pixels': sum(counts[1:]),
              f'{prefix}_foreground_fraction': sum(counts[1:]) / total if total else 0.0}
    for name, count in zip(CLASS_NAMES, counts):
        result[f'{prefix}_{name}_pixels'] = count
        result[f'{prefix}_{name}_fraction'] = count / total if total else 0.0
    return result


class TrainingProgress:
    """Stream sampled batch metrics; ETA covers training only, never later stages."""
    def __init__(self, *, stage, path, epochs, batches_per_epoch, log_every=50, enabled=True):
        if log_every < 0:
            raise ValueError('log-every must be nonnegative')
        self.stage, self.path = stage, Path(path)
        self.epochs, self.batches_per_epoch = epochs, batches_per_epoch
        self.log_every, self.enabled = log_every, enabled
        self.bar = None
        self.file = None
        self.started = time.perf_counter()
        self.completed = 0
        self.samples = 0
        if log_every:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open('x', encoding='utf-8', buffering=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end_epoch()
        if self.file is not None:
            self.file.close()

    def start_epoch(self, epoch):
        self.end_epoch()
        self.epoch = epoch
        self.epoch_started = time.perf_counter()
        self.epoch_samples = 0
        self.bar = progress_bar(total=self.batches_per_epoch,
                                desc=f'{self.stage} {epoch + 1}/{self.epochs}', enabled=self.enabled)

    def should_log(self, step):
        return bool(self.log_every and
                    (step == 0 or (step + 1) % self.log_every == 0 or step + 1 == self.batches_per_epoch))

    def update(self, step, *, batch_samples, metrics):
        self.completed += 1
        self.samples += batch_samples
        self.epoch_samples += batch_samples
        elapsed = max(time.perf_counter() - self.epoch_started, 1e-9)
        stage_elapsed = max(time.perf_counter() - self.started, 1e-9)
        if self.bar is not None:
            self.bar.set_postfix({k: v for k, v in metrics.items()
                                  if k in ('loss', 'total', 'accepted_regions', 'skipped_empty')}, refresh=False)
            self.bar.update(1)
        if not self.should_log(step):
            return
        row = {**metrics, 'epoch': self.epoch, 'step': step, 'batch': step + 1,
               'batches_per_epoch': self.batches_per_epoch, 'global_batch': self.completed,
               'batch_samples': batch_samples, 'elapsed_epoch_seconds': elapsed,
               'elapsed_training_seconds': stage_elapsed,
               'batches_per_second': (step + 1) / elapsed,
               'samples_per_second': self.epoch_samples / elapsed,
               'epoch_eta_seconds': max(0, self.batches_per_epoch - step - 1) * elapsed / (step + 1),
               'training_eta_seconds': max(0, self.epochs * self.batches_per_epoch - self.completed)
                                       * stage_elapsed / self.completed}
        event = {'schema': EVENT_SCHEMA, 'stage': self.stage, 'metrics': row}
        line = json.dumps(event, sort_keys=True, allow_nan=False)
        if self.file is not None:
            self.file.write(line + '\n')
            self.file.flush()  # survives a later training failure; this is not a checkpoint
        if self.bar is not None:
            self.bar.write(EVENT_PREFIX + line, file=sys.stdout)
        else:
            print(EVENT_PREFIX + line, flush=True)
        sys.stdout.flush()

    def end_epoch(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
