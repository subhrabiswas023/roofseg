import numpy as np

type ImageArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type MaskArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]