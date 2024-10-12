import os
import random
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np

from .import_util import (
    is_polars_available,
    is_tf_available,
    is_torch_available,
)


def is_temporal(val):
    return isinstance(val, (datetime, date, time, timedelta))


def is_decimal(val):
    return isinstance(val, Decimal)


def as_py(val):
    # return as int64
    if isinstance(val, datetime):
        return val.timestamp()
    elif isinstance(val, date):
        return val.toordinal()
    elif isinstance(val, time):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    elif isinstance(val, timedelta):
        return val.total_seconds()
    elif isinstance(val, Decimal):
        return float(val)
    elif isinstance(val, np.float16):
        return float(val)
    elif isinstance(val, bytes):
        return f"base64:{val.hex()}"
    elif isinstance(val, dict):
        return {k: as_py(v) for k, v in val.items()}
    return val


def enable_full_determinism(seed: int, warn_only: bool = False):
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """
    # set seed first
    set_seed(seed)

    if is_torch_available():
        import torch

        # Enable PyTorch deterministic mode. This potentially requires either the environment
        # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
        # depending on the CUDA version, so we set them both here
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_tf_available():
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    if is_polars_available():
        from polars import set_random_seed

        set_random_seed(seed)

    def is_torch_npu_available():
        try:
            import torch

            return torch.npu.is_available()
        except ImportError:
            return False

    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)
