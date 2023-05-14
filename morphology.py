import numpy as np
import numpy.typing as nptyping
import scipy


Bitmap = nptyping.NDArray[bool]


G123_LUT = np.array([
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
    1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0
], dtype=bool)

G123P_LUT = np.array([
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
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=bool)


def bwmorph_thin(bitmap: Bitmap, num_iters: int = -1) -> Bitmap:
    """The bwmorph thinning algorithm 
    
    For more information on the algorithm, see 
    https://it.mathworks.com/help/images/ref/bwmorph.html#bui7chk-1
    """
    if bitmap.ndim not in [2, 3]: 
        raise ValueError("The bitmap must be a 2D array or a batched 3D tensor")
        
    if not np.all(np.in1d(bitmap.flat, (0, 1))):
        raise ValueError("The bitmap contains values other than 0 and 1")

    if num_iters <= 0 and num_iters != -1:
        raise ValueError("num_iters must be > 0 or equal to -1")

    bitmap = np.array(bitmap).astype(np.uint8)
    
    batched3d = bitmap.ndim == 3
    
    if not batched3d:
        bitmap = np.expand_dims(bitmap, 0)
 
    # The neighborhood kernel
    kernel = np.array([
        [ 8,  4,   2],
        [16,  0,   1],
        [32, 64, 128]
    ], dtype=np.uint8)
    
    finished = np.zeros(bitmap.shape[0], dtype=bool)

    batch_size = bitmap.shape[0]
    num_pixels_before = np.sum(bitmap, axis=(1, 2))

    while num_iters != 0:
        # The two subiterations
        for lut in [G123_LUT, G123P_LUT]:
            for idx in range(batch_size):  # It is faster than the batched operation
                if finished[idx]:
                    continue

                N = scipy.ndimage.correlate(bitmap[idx], kernel, mode="constant")
                D = np.take(lut, N)
                
                bitmap[idx][D] = 0

        num_pixels = np.sum(bitmap, axis=(1, 2))
        
        finished = num_pixels == num_pixels_before

        if np.all(finished):
            break

        num_pixels_before = num_pixels
        
        num_iters -= 1
    
    if not batched3d:
        bitmap = np.squeeze(bitmap, axis=0)

    return bitmap.astype(bool)
