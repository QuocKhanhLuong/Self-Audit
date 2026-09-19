from omegaconf import OmegaConf
import torch
from train_segmentation import LitUnsupervisedSegmenter

cfg = OmegaConf.load("configs/train_config.yml")
try:
    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.pretrained_weights)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
