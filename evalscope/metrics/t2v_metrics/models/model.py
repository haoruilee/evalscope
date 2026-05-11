import contextlib
import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Optional

from ..constants import CACHE_DIR


def image_loader(image_path):
    if image_path.split('.')[-1] == 'npy':
        return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
    else:
        return Image.open(image_path).convert('RGB')


def _is_npu_available() -> bool:
    """True iff a usable Ascend NPU is reachable via torch_npu."""
    try:
        from transformers.utils import is_torch_npu_available
        return bool(is_torch_npu_available())
    except Exception:
        pass
    # Fall back to torch_npu's own probe when transformers isn't around.
    try:
        return hasattr(torch, 'npu') and torch.npu.is_available()  # type: ignore[attr-defined]
    except Exception:
        return False


def _is_mps_available() -> bool:
    """True iff Apple MPS is usable on this host."""
    try:
        from transformers.utils import is_torch_mps_available
        return bool(is_torch_mps_available())
    except Exception:
        pass
    try:
        return bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _autodetect_device() -> str:
    """Return the best locally-available device, mirroring evalscope's helper."""
    try:
        from evalscope.utils.model_utils import get_device
        return get_device()
    except Exception:
        if _is_npu_available():
            return 'npu'
        if _is_mps_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve the runtime device for score models.

    Falls back to an actually-available accelerator (NPU/MPS/CUDA) or CPU
    when the requested device is ``None`` or unavailable. This makes the
    t2v metric models (e.g. FGA-BLIP2) work on non-CUDA hardware such as
    Ascend NPUs without manual code changes — see issue #1331.
    """
    auto_device = _autodetect_device()

    if device is None:
        return auto_device

    device_str = str(device)
    device_type = device_str.split(':', 1)[0]

    if device_type == 'cuda' and not torch.cuda.is_available():
        return auto_device
    if device_type == 'npu' and not _is_npu_available():
        return auto_device
    if device_type == 'mps' and not _is_mps_available():
        return auto_device

    return device_str


class ScoreModel(ABC):

    def __init__(self, model_name='clip-flant5-xxl', device: Optional[str] = 'cuda', cache_dir=CACHE_DIR):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.image_loader = image_loader
        self.load_model()

    def maybe_autocast(self, dtype: torch.dtype = torch.float16):
        """Return an autocast context appropriate for ``self.device``.

        Mirrors the helper on ``lavis.Blip2Base.maybe_autocast`` so model
        wrappers can opt into mixed precision without hard-coding CUDA.
        Autocast is disabled on CPU.
        """
        try:
            device_type = torch.device(self.device).type
        except Exception:
            device_type = 'cpu'

        if device_type == 'cpu':
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=device_type, dtype=dtype)

    @abstractmethod
    def load_model(self):
        """Load the model, tokenizer, and etc.
        """
        pass

    @abstractmethod
    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        pass

    @abstractmethod
    def forward(self, images: List[str], texts: List[str], **kwargs) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        pass
