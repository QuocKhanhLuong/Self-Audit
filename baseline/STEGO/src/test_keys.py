import torch
ckpt = torch.load("../saved_models/cityscapes_vit_base_1.ckpt", map_location='cpu', weights_only=False)
keys = list(ckpt['state_dict'].keys())
print("Total keys in state_dict:", len(keys))
print("First 5 keys:", keys[:5])
print("Are there 'net.patch_embed' keys? ", any('patch_embed' in k for k in keys))
