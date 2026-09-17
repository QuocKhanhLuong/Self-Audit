"""Only the Annotation loss accepts segmentation ground truth."""
import torch
import torch.nn.functional as F


def segmentation_loss(logits,target):
    if target.dtype!=torch.long or target.shape!=(len(logits),*logits.shape[-2:]):
        raise ValueError('int64 [B,H,W] segmentation target required')
    if target.min()<0 or target.max()>=logits.shape[1]: raise ValueError('target class outside range')
    ce=F.cross_entropy(logits,target,reduction='none').mean((1,2))
    p=logits.softmax(1)[:,1:]
    y=F.one_hot(target,logits.shape[1]).permute(0,3,1,2).to(p)[:,1:]
    dice=(2*(p*y).sum((2,3))+1e-5)/(p.sum((2,3))+y.sum((2,3))+1e-5)
    return (ce+1-dice.mean(1)).mean()
