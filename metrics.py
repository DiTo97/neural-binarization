"""A module that defines metrics for evaluating document image binarization (DIBCO)

The implemented metrics are the following:
- F-measure
- pseudo F-measure
- peak signal-to-noise ratio (PSNR)
- distance reciprocal distortion (DRD)

For more information on the DIBCO metrics, see the 2017 introductory paper,
https://ieeexplore.ieee.org/document/8270159
"""
import typing

from common import Bitmap
from morphology import bwmorph_thin


def fmeasure() -> None:
    """The F-measure metric"""
    pass


def psnr() -> None:
    """The peak signal-to-noise ratio (PSNR) metric"""
    pass


def drd() -> None:
    """The distance reciprocal distortion (DRD) metric"""
    pass


class DIBCO:
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
            
        batch_size = references.shape[0]
