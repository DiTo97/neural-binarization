import numpy
import scipy

from .typing import Bitmap


G123_LUT = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
], dtype=bool)


G123P_LUT = numpy.array([
    0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=bool)


def bwmorph_thinning(bitmap: Bitmap, num_iters: int = -1) -> Bitmap:
    """The BWmorph thinning algorithm

    For more information on the algorithm, see
    https://it.mathworks.com/help/images/ref/bwmorph.html#bui7chk-1
    """
    if bitmap.ndim not in [2, 3]:
        raise ValueError("The bitmap must be a 2D array or a batched 3D tensor")

    if not numpy.all(numpy.in1d(bitmap.flat, (0, 1))):
        raise ValueError("The bitmap contains values other than 0 and 1")

    if num_iters <= 0 and num_iters != -1:
        raise ValueError("num_iters must be > 0 or equal to -1")

    bitmap = numpy.array(bitmap).astype(numpy.uint8)

    batched3d = bitmap.ndim == 3

    if not batched3d:
        bitmap = numpy.expand_dims(bitmap, 0)

    # neighborhood
    kernel = numpy.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=numpy.uint8)

    finished = numpy.zeros(bitmap.shape[0], dtype=bool)

    batch_size = bitmap.shape[0]
    num_pixels_before = numpy.sum(bitmap, axis=(1, 2))

    while num_iters != 0:
        # subiterations
        for LUT in [G123_LUT, G123P_LUT]:
            for idx in range(batch_size):  # batched seems slower
                if finished[idx]:
                    continue

                N = scipy.ndimage.correlate(bitmap[idx], kernel, mode="constant")
                D = numpy.take(LUT, N)

                bitmap[idx][D] = 0

        num_pixels = numpy.sum(bitmap, axis=(1, 2))

        finished = num_pixels == num_pixels_before

        if numpy.all(finished):
            break

        num_pixels_before = num_pixels

        num_iters -= 1

    if not batched3d:
        bitmap = numpy.squeeze(bitmap, axis=0)

    return bitmap.astype(bool)
