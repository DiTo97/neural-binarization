"""module that defines metrics for evaluating document image binarization (DIBCO)

The implemented metrics are the following:
- distance reciprocal distortion (DRD)
- F-measure
- pseudo F-measure
- peak signal-to-noise ratio (PSNR)

The pseudo F-measure metric is available only in slow mode.

For more information on DIBCO, see the original paper,
https://ieeexplore.ieee.org/document/8270159
"""

from dataclasses import dataclass
from typing import Any

import numpy
from numpy.typing import NDArray

from .typing import Bitmap


try:
    import doxapy

    fastmode = True
except ImportError:
    fastmode = False
    from .morphology import bwmorph_thinning


def drd(
    references: Bitmap,
    preds: Bitmap,
    eps: float = 1e-6,
    block_size: int = 8,
    block_mask_size: int = 5,
) -> NDArray[numpy.float_]:
    """distance reciprocal distortion (DRD)"""
    batch_size, *shape = references.shape

    u_references = numpy.zeros((batch_size, shape[0] + 2, shape[1] + 2))
    u_references[:, 1 : shape[0] + 1, 1 : shape[1] + 1] = references

    block_size_square = block_size**2

    II = numpy.cumsum(u_references, 1)  # integral images
    II = numpy.cumsum(II, 2)

    nonuniform = numpy.zeros(batch_size)  # NUBN

    for idx in range(1, shape[0] - block_size + 1, block_size):
        for jdx in range(1, shape[1] - block_size + 1, block_size):
            block_sum = (
                II[:, idx + block_size - 1, jdx + block_size - 1]
                - II[:, idx - 1, jdx + block_size - 1]
                - II[:, idx + block_size - 1, jdx - 1]
                + II[:, idx - 1, jdx - 1]
            )

            allwhite = block_sum == 0
            allblack = block_sum == block_size_square

            nonuniform += ~(allwhite | allblack)

    attention = numpy.zeros((batch_size, block_mask_size, block_mask_size))

    half_mask_size = (block_mask_size + 1) / 2
    half_mask_size = int(half_mask_size)

    for idx in range(block_mask_size):
        for jdx in range(block_mask_size):
            nume = numpy.sqrt(
                (idx + 1 - half_mask_size) ** 2 + (jdx + 1 - half_mask_size) ** 2
            )

            if nume == 0:
                continue

            attention[:, idx, jdx] = 1 / nume

    attention = attention / numpy.sum(attention, axis=(1, 2), keepdims=True)

    u_references_resized = numpy.zeros(
        (batch_size, shape[0] + half_mask_size + 1, shape[1] + half_mask_size + 1)
    )

    u_references_resized[
        :,
        half_mask_size - 1 : shape[0] + half_mask_size - 1,
        half_mask_size - 1 : shape[1] + half_mask_size - 1,
    ] = references

    u_preds_resized = numpy.zeros(
        (batch_size, shape[0] + half_mask_size + 1, shape[1] + half_mask_size + 1)
    )

    u_preds_resized[
        :,
        half_mask_size - 1 : shape[0] + half_mask_size - 1,
        half_mask_size - 1 : shape[1] + half_mask_size - 1,
    ] = preds

    neg_u_references_resized = 1 - u_references_resized
    neg_u_preds_resized = 1 - u_preds_resized

    FP_resized = neg_u_preds_resized * u_references_resized
    FP_resized = FP_resized.astype(bool)

    FN_resized = u_preds_resized * neg_u_references_resized
    FN_resized = FN_resized.astype(bool)

    difference = FP_resized | FN_resized

    _, *shape_resized = difference.shape

    sum_score = numpy.zeros(batch_size)

    for idx in range(half_mask_size - 1, shape_resized[0] - half_mask_size + 1):
        for jdx in range(half_mask_size - 1, shape_resized[1] - half_mask_size + 1):
            equal = difference[:, idx, jdx] == 0

            local_u_references = u_references_resized[
                :,
                idx - half_mask_size + 1 : idx + half_mask_size,
                jdx - half_mask_size + 1 : jdx + half_mask_size,
            ]

            local_u_preds = u_preds_resized[:, idx, jdx]
            local_u_preds = numpy.expand_dims(local_u_preds, axis=(1, 2))

            local_FP = (1 - local_u_references) * local_u_preds
            local_FP = local_FP.astype(bool)

            local_FN = (1 - local_u_preds) * local_u_references
            local_FN = local_FN.astype(bool)

            local_difference = local_FP | local_FN
            local_difference = local_difference * attention

            local_score = numpy.sum(local_difference, axis=(1, 2))
            local_score = local_score * ~equal

            sum_score += local_score

    nume = sum_score
    deno = nonuniform + eps

    score = nume / deno
    return score


def fmeasure(
    references: Bitmap, preds: Bitmap, eps: float = 1e-6
) -> NDArray[numpy.float_]:
    """F-measure"""
    neg_references = 1 - references
    neg_preds = 1 - preds

    TP = neg_preds * neg_references
    FP = neg_preds * references
    FN = preds * neg_references

    num_TP = numpy.sum(TP, axis=(1, 2))
    num_FP = numpy.sum(FP, axis=(1, 2))
    num_FN = numpy.sum(FN, axis=(1, 2))

    precision = num_TP / (num_FP + num_TP + eps)
    recall = num_TP / (num_FN + num_TP + eps)

    nume = 2 * (precision * recall)
    deno = precision + recall + eps

    score = nume / deno
    return score


def pseudo_fmeasure(
    references: Bitmap, preds: Bitmap, eps: float = 1e-6, **kwargs
) -> NDArray[numpy.float_]:
    """pseudo F-measure"""
    neg_references = 1 - references
    neg_preds = 1 - preds

    skeletons = bwmorph_thinning(neg_references, **kwargs).astype(numpy.uint8)

    TP = neg_preds * neg_references
    FP = neg_preds * references

    num_TP = numpy.sum(TP, axis=(1, 2))
    num_FP = numpy.sum(FP, axis=(1, 2))

    precision = num_TP / (num_FP + num_TP + eps)

    pseudo_TP = neg_preds * skeletons
    pseudo_FN = preds * skeletons

    num_pseudo_TP = numpy.sum(pseudo_TP, axis=(1, 2))
    num_pseudo_FN = numpy.sum(pseudo_FN, axis=(1, 2))

    pseudo_recall = num_pseudo_TP / (num_pseudo_FN + num_pseudo_TP + eps)

    pseudo_nume = 2 * (precision * pseudo_recall)
    pseudo_deno = precision + pseudo_recall + eps

    pseudo_score = pseudo_nume / pseudo_deno
    return pseudo_score


def psnr(references: Bitmap, preds: Bitmap, eps: float = 1e-6) -> NDArray[numpy.float_]:
    """peak signal-to-noise ratio (PSNR)"""
    neg_references = 1 - references
    neg_preds = 1 - preds

    FP = neg_preds * references
    FN = preds * neg_references

    mistakes = FP | FN

    error = numpy.mean(mistakes, axis=(1, 2)) + eps
    score = 10 * numpy.log10(1 / error)

    return score


@dataclass
class _slow_DIBCO:
    num_thin_iters: int = -1
    eps: float = 1e-6
    block_size: int = 8
    block_mask_size: int = 5

    def __call__(self, references: Bitmap, preds: Bitmap) -> dict[str, float]:
        batch_metrics = {
            "DRD": numpy.mean(
                drd(references, preds, self.eps, self.block_size, self.block_mask_size)
            ),
            "F-measure": numpy.mean(fmeasure(references, preds, self.eps)),
            "pseudo-F-measure": numpy.mean(
                pseudo_fmeasure(
                    references, preds, self.eps, num_iters=self.num_thin_iters
                )
            ),
            "PSNR": numpy.mean(psnr(references, preds, self.eps)),
        }

        return batch_metrics


class _fast_DIBCO:
    def __call__(self, references: Bitmap, preds: Bitmap) -> dict[str, float]:
        batch_size = references.shape[0]

        batch_metrics = {"DRD": [], "F-measure": [], "PSNR": []}

        for idx in range(batch_size):
            metrics = doxapy.calculate_performance_ex(
                references[idx], preds[idx], fm=True, psnr=True, drdm=True
            )

            batch_metrics["DRD"].append(metrics["drdm"])
            batch_metrics["F-measure"].append(metrics["fm"])
            batch_metrics["PSNR"].append(metrics["psnr"])

        batch_metrics = {key: numpy.mean(val) for key, val in batch_metrics.items()}

        return batch_metrics


class DIBCO:
    """evaluation suite of document image binarization (DIBCO) metrics"""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        _DIBCO = _fast_DIBCO if fastmode else _slow_DIBCO
        self.metric = _DIBCO(**(kwargs or {}))

    def __call__(self, references: Bitmap, preds: Bitmap) -> dict[str, float]:
        if references.ndim not in [2, 3]:
            raise ValueError(
                "The references bitmap must be a 2D array or a batched 3D tensor"
            )

        if preds.ndim not in [2, 3]:
            raise ValueError(
                "The preds bitmap must be a 2D array or a batched 3D tensor"
            )

        if references.ndim != preds.ndim:
            raise ValueError(
                "The references and preds bitmaps do not have the same dimensionality"
            )

        if not numpy.all(numpy.in1d(references.flat, (0, 1))):
            raise ValueError("The references bitmap contains values other than 0 and 1")

        if not numpy.all(numpy.in1d(references.flat, (0, 1))):
            raise ValueError("The preds bitmap contains values other than 0 and 1")

        if not references.ndim == 3:
            references = numpy.expand_dims(references, 0)
            preds = numpy.expand_dims(preds, 0)

        if references.shape[0] != preds.shape[0]:
            raise ValueError(
                "The references and preds bitmaps do not have the same number of examples"
            )

        batch_metrics = self.metric(references, preds)

        return batch_metrics
