"""A module that defines metrics for evaluating document image binarization (DIBCO)

The implemented metrics are the following:
- distance reciprocal distortion (DRD)
- F-measure
- pseudo F-measure
- peak signal-to-noise ratio (PSNR)

For more information on the DIBCO metrics, see the 2017 introductory paper,
https://ieeexplore.ieee.org/document/8270159
"""
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as nptyping

from common import Bitmap
from morphology import bwmorph_thin


def drd(references: Bitmap, preds: Bitmap) -> None:
    """The distance reciprocal distortion (DRD) metric"""
    pass
    

def fmeasure(references: Bitmap, preds: Bitmap, eps: float = 1e-6) -> nptyping.NDArray[float]:
    """The F-measure metric"""
    neg_references = 1 - references
    neg_preds = 1 - preds
    
    tpositives = neg_preds * neg_references
    fpositives = neg_preds * references
    fnegatives = preds * neg_references
    
    num_tpositives = np.sum(tpositives, axis=(1, 2))
    num_fpositives = np.sum(fpositives, axis=(1, 2))
    num_fnegatives = np.sum(fnegatives, axis=(1, 2))
    
    precision = num_tpositives / (num_fpositives + num_tpositives + eps)
    recall = num_tpositives / (num_fnegatives + num_tpositives + eps)
    
    nume = 2 * (precision * recall)
    deno = precision + recall + eps
    
    score = nume / deno
    return score


def psuedo_fmeasure(references: Bitmap, preds: Bitmap, **kwargs) -> None:
    """The pseudo F-measure metric"""
    neg_references = 1 - references
    neg_preds = 1 - preds
    
    skeletons = bwmorph_thin(neg_references, **kwargs)
    skeletons = skeletons.astype(np.uint8)
    
    neg_skeletons = 1 - skeletons
    
    tpositives = neg_preds * neg_references
    fpositives = neg_preds * references
    
    num_tpositives = np.sum(tpositives, axis=(1, 2))
    num_fpositives = np.sum(fpositives, axis=(1, 2))
    
    precision = num_tpositives / (num_fpositives + num_tpositives + eps)
    
    psuedo_tpositives = neg_preds * skeletons
    psuedo_fnegatives = preds * neg_skeletons
    
    num_pseudo_tpositives = np.sum(psuedo_tpositives, axis=(1, 2))
    num_pseudo_fnegatives = np.sum(psuedo_fnegatives, axis=(1, 2))
    
    pseudo_recall = num_pseudo_tpositives / (num_pseudo_fnegatives + num_pseudo_tpositives + eps)
    
    psuedo_nume = 2 * (precision * pseudo_recall)
    pseudo_deno = precision + pseudo_recall + eps
    
    pseudo_score = psuedo_nume / pseudo_deno
    return pseudo_score


def psnr(references: Bitmap, preds: Bitmap) -> nptyping.NDArray[float]:
    """The peak signal-to-noise ratio (PSNR) metric"""
    neg_references = 1 - references
    neg_preds = 1 - preds
    
    fpositives = neg_preds * references
    fnegatives = preds * neg_references
    
    ftotals = fpositives | fnegatives
    
    err = np.mean(ftotals, axis=(1, 2))
    score = 10 * np.log10(1 / err)
    
    return score


@dataclass
class DIBCO:
    num_thin_iters: int = -1
    eps: float = 1e-6
        
    def __call__(self, references: Bitmap, preds: Bitmap) -> typing.Dict[str, float]:
        if references.ndim not in [2, 3]: 
            raise ValueError("The references bitmap must be a 2D array or a batched 3D tensor")
            
        if preds.ndim not in [2, 3]: 
            raise ValueError("The preds bitmap must be a 2D array or a batched 3D tensor")
            
        if references.ndim != preds.ndim:
            raise ValueError("The references and preds bitmaps do not have the same dimensionality")

        if not np.all(np.in1d(references.flat, (0, 1))):
            raise ValueError("The references bitmap contains values other than 0 and 1")
            
        if not np.all(np.in1d(references.flat, (0, 1))):
            raise ValueError("The preds bitmap contains values other than 0 and 1")

        if not references.ndim == 3:
            references = np.expand_dims(references, 0)
            preds = np.expand_dims(preds, 0)
            
        if references.shape[0] != preds.shape[0]:
            raise ValueError("The references and preds bitmaps do not have the same number of examples")
        
        batch_drds = None
        batch_fmeasures = fmeasure(references, preds, self.eps)
        batch_pseudo_fmeasures = psuedo_fmeasure(references, preds, num_iters=self.num_thin_iters)
        batch_psnrs = psnr(references, preds)
        
        batch_metrics = {
            "drd": np.mean(batch_drds),
            "f-measure": np.mean(batch_fmeasures),
            "pseudo-f-measure": np.mean(batch_pseudo_fmeasures),
            "psnr": np.mean(batch_psnrs)
        }
        
        return batch_metrics
