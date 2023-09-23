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
from typing import Any

import numpy as np
import numpy.typing as np_typing

from common import Bitmap
from morphology import bwmorph_thin


try:
    import doxapy
    has_doxapy = True
except ImportError:
    has_doxapy = False


def drd(
    references: Bitmap, 
    preds: Bitmap, 
    eps: float = 1e-6, 
    block_size: int = 8, 
    block_mask_size: int = 5
) -> np_typing.NDArray[np.float_]:
    """The distance reciprocal distortion (DRD) metric"""
    batch_size, height, width = references.shape

    u_references = np.zeros((batch_size, height + 2, width + 2))
    u_references[:, 1:height + 1, 1:width + 1] = references

    II = np.cumsum(u_references, 1)  # integral images
    II = np.cumsum(II, 2)

    block_size_square = block_size ** 2
    nonuniform_pixels = np.zeros(batch_size)  # NUBN

    for idx in range(1, height - block_size + 1, block_size):
        for jdx in range(1, width - block_size + 1, block_size):
            block_sum = (
                  II[:, idx + block_size - 1, jdx + block_size - 1]
                - II[:, idx - 1, jdx + block_size - 1]
                - II[:, idx + block_size - 1, jdx - 1]
                + II[:, idx - 1, jdx - 1]
            )

            allwhite = block_sum == 0
            allblack = block_sum == block_size_square

            nonuniform = ~(allwhite | allblack)
            nonuniform_pixels += nonuniform

    weights = np.zeros((batch_size, block_mask_size, block_mask_size))

    half_mask_size = (block_mask_size + 1) / 2
    half_mask_size = int(half_mask_size)

    for idx in range(block_mask_size):
        for jdx in range(block_mask_size):
            nume = np.sqrt(
                  (idx + 1 - half_mask_size) ** 2 
                + (jdx + 1 - half_mask_size) ** 2
            )

            if nume == 0:
                continue

            weights[:, idx, jdx] = 1 / nume

    norm_weights = weights / np.sum(weights, axis=(1, 2), keepdims=True)

    u_references_resized = np.zeros((
        batch_size, 
        height + half_mask_size + 1, 
         width + half_mask_size + 1
    ))

    u_references_resized[
        :, 
        half_mask_size - 1:height + half_mask_size - 1, 
        half_mask_size - 1: width + half_mask_size - 1
    ] = references

    u_preds_resized = np.zeros((
        batch_size, 
        height + half_mask_size + 1, 
         width + half_mask_size + 1
    ))

    u_preds_resized[
        :,
        half_mask_size - 1:height + half_mask_size - 1, 
        half_mask_size - 1: width + half_mask_size - 1
    ] = preds

    neg_u_references_resized = 1 - u_references_resized
    neg_u_preds_resized = 1 - u_preds_resized

    fpositives_resized = neg_u_preds_resized * u_references_resized
    fpositives_resized = fpositives_resized.astype(bool)

    fnegatives_resized = u_preds_resized * neg_u_references_resized
    fnegatives_resized = fnegatives_resized.astype(bool)

    difference = fpositives_resized | fnegatives_resized

    _, height_resized, width_resized = difference.shape

    sum_score = np.zeros(batch_size)

    for idx in range(half_mask_size - 1, height_resized - half_mask_size + 1):
        for jdx in range(half_mask_size - 1, width_resized - half_mask_size + 1):
            equal = difference[:, idx, jdx] == 0

            local_u_references = u_references_resized[
                :,
                idx - half_mask_size + 1:idx + half_mask_size, 
                jdx - half_mask_size + 1:jdx + half_mask_size
            ]

            local_u_preds = u_preds_resized[:, idx, jdx]
            local_u_preds = np.expand_dims(local_u_preds, axis=(1, 2))

            local_fpositives = (1 - local_u_references) * local_u_preds
            local_fpositives = local_fpositives.astype(bool)

            local_fnegatives = (1 - local_u_preds) * local_u_references
            local_fnegatives = local_fnegatives.astype(bool)
            
            local_difference = local_fpositives | local_fnegatives
            local_difference = local_difference * norm_weights

            local_score = np.sum(local_difference, axis=(1, 2))
            local_score = local_score * ~equal

            sum_score += local_score

    nume = sum_score
    deno = nonuniform_pixels + eps

    score = nume / deno
    return score
    

def fmeasure(references: Bitmap, preds: Bitmap, eps: float = 1e-6) -> np_typing.NDArray[np.float_]:
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


def pseudo_fmeasure(references: Bitmap, preds: Bitmap, eps: float = 1e-6, **kwargs) -> np_typing.NDArray[np.float_]:
    """The pseudo F-measure metric"""
    neg_references = 1 - references
    neg_preds = 1 - preds
    
    skeletons = bwmorph_thin(neg_references, **kwargs).astype(np.uint8)
        
    tpositives = neg_preds * neg_references
    fpositives = neg_preds * references
    
    num_tpositives = np.sum(tpositives, axis=(1, 2))
    num_fpositives = np.sum(fpositives, axis=(1, 2))
    
    precision = num_tpositives / (num_fpositives + num_tpositives + eps)
    
    pseudo_tpositives = neg_preds * skeletons
    pseudo_fnegatives = preds * skeletons
    
    num_pseudo_tpositives = np.sum(pseudo_tpositives, axis=(1, 2))
    num_pseudo_fnegatives = np.sum(pseudo_fnegatives, axis=(1, 2))
    
    pseudo_recall = num_pseudo_tpositives / (num_pseudo_fnegatives + num_pseudo_tpositives + eps)
    
    pseudo_nume = 2 * (precision * pseudo_recall)
    pseudo_deno = precision + pseudo_recall + eps
    
    pseudo_score = pseudo_nume / pseudo_deno
    return pseudo_score


def psnr(references: Bitmap, preds: Bitmap, eps: float = 1e-6) -> np_typing.NDArray[np.float_]:
    """The peak signal-to-noise ratio (PSNR) metric"""
    neg_references = 1 - references
    neg_preds = 1 - preds
    
    fpositives = neg_preds * references
    fnegatives = preds * neg_references
    
    ftotals = fpositives | fnegatives
    
    err = np.mean(ftotals, axis=(1, 2)) + eps
    score = 10 * np.log10(1 / err)
    
    return score


@dataclass
class slow_DIBCO:
    num_thin_iters: int = -1
    eps: float = 1e-6
    block_size: int = 8
    block_mask_size: int = 5

    def __call__(self, references: Bitmap, preds: Bitmap) -> typing.Dict[str, float]:
        batch_drds = drd(references, preds, self.eps, self.block_size, self.block_mask_size)
        batch_fmeasures = fmeasure(references, preds, self.eps)
        batch_pseudo_fmeasures = pseudo_fmeasure(references, preds, self.eps, num_iters=self.num_thin_iters)
        batch_psnrs = psnr(references, preds, self.eps)

        batch_metrics = {
            "DRD": np.mean(batch_drds),
            "F-measure": np.mean(batch_fmeasures),
            "pseudo-F-measure": np.mean(batch_pseudo_fmeasures),
            "PSNR": np.mean(batch_psnrs)
        }

        return batch_metrics


class fast_DIBCO:
    def __init__(self) -> None:
        if not has_doxapy:
            raise ValueError(
                "The fast DIBCO suite requires Doxapy. "
                "Try in your venv 'python -m pip install doxapy'"
            )

    def __call__(self, references: Bitmap, preds: Bitmap) -> typing.Dict[str, float]:
        batch_size = references.shape[0]

        batch_metrics = {
            "DRD": [],
            "F-measure": [],
            "PSNR": []
        }

        for idx in range(batch_size):
            metrics = doxapy.calculate_performance_ex(
                references[idx], 
                preds[idx],
                fm=True,
                psnr=True,
                drdm=True
            )

            batch_metrics["DRD"].append(metrics["drdm"])
            batch_metrics["F-measure"].append(metrics["fm"])
            batch_metrics["PSNR"].append(metrics["psnr"])
            
        batch_metrics = {key: np.mean(val) for key, val in batch_metrics.items()}

        return batch_metrics


class DIBCO:
    """An evaluation suite of document image binarization (DIBCO) metrics"""
    def __init__(self, fast: bool = False, **kwargs: typing.Dict[str, Any] = None) -> None:    
        __DIBCO = fast_DIBCO if fast else slow_DIBCO
        self.metric = __DIBCO(**(kwargs or {}))
        
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

        batch_metrics = self.metric(references, preds)

        return batch_metrics
