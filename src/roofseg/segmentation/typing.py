import numpy as np
import torch
from jaxtyping import Float, Int

ImageArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
MaskArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]

PairedArray = tuple[ImageArray, MaskArray]

C = "channel"
H = "height"
W = "width"
B = "batch"

ImageTensor = Float[torch.Tensor, f"{C} {H} {W}"]
MaskTensor = Int[torch.Tensor, f"{H} {W}"]

BatchedImageTensor = Float[torch.Tensor, f"{B} {C} {H} {W}"]
BatchedMaskTensor = Int[torch.Tensor, f"{B} {H} {W}"]

PairedTensor = tuple[ImageTensor, MaskTensor]
PairedBatchedTensor = tuple[BatchedImageTensor, BatchedMaskTensor]
