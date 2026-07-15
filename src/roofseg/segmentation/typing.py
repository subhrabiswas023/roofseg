import numpy as np
import torch
from jaxtyping import Float, Int

ImageArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
MaskArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]

ImageTensor = Float[torch.Tensor, "channel height width"]
MaskTensor = Int[torch.Tensor, "height width"]

BatchedImageTensor = Float[torch.Tensor, "batch channel height width"]
BatchedMaskTensor = Int[torch.Tensor, "batch height width"]

PairedTensor = tuple[ImageTensor, MaskTensor]
PairedBatchedTensor = tuple[BatchedImageTensor, BatchedMaskTensor]
