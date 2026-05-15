from dataclasses import asdict, dataclass

import numpy as np
import torch
from jaxtyping import Float, Int


@dataclass(frozen=True)
class Dataclass:
    def asdict(self) -> dict[str, object]:
        return asdict(self)


RGBArray = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
GrayscaleArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]

RGBTensor = Float[torch.Tensor, "channel height width"]
GrayScaleTensor = Int[torch.Tensor, "height width"]

BatchedRGBTensor = Float[torch.Tensor, "batch channel height width"]
BatchedGrayscaleTensor = Int[torch.Tensor, "batch height width"]