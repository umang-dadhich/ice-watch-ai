from __future__ import annotations
import numpy as np
from skimage.morphology import remove_small_objects, closing, square

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except Exception:
    dcrf = None


def morphology_refine(prob: np.ndarray, min_size: int = 256) -> np.ndarray:
    mask = prob > 0.5
    mask = remove_small_objects(mask, min_size=min_size)
    mask = closing(mask, square(3))
    return mask.astype(np.uint8)


def crf_refine(image: np.ndarray, prob: np.ndarray, n_iter: int = 5) -> np.ndarray:
    if dcrf is None:
        return (prob > 0.5).astype(np.uint8)
    h, w = prob.shape
    d = dcrf.DenseCRF2D(w, h, 2)
    U = unary_from_softmax(np.vstack([1 - prob, prob]))
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)
    Q = d.inference(n_iter)
    refined = np.array(Q)[1].reshape(h, w)
    return (refined > 0.5).astype(np.uint8)
