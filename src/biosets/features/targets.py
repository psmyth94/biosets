from dataclasses import InitVar, dataclass, field
from typing import Any, ClassVar, List, Optional, Union

import pyarrow as pa
from datasets import ClassLabel
from datasets.features.features import string_to_arrow


@dataclass
class RegressionTarget:
    dtype: str = "float"
    id: Optional[str] = None
    pa_type: ClassVar[Any] = None
    _type: str = field(default="RegressionTarget", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "float":
            self.pa_type = pa.float32()
        elif self.dtype == "double":
            self.pa_type = pa.float64()
        elif self.dtype in ["float16", "float32", "float64"]:
            self.pa_type = string_to_arrow(self.dtype)
        else:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def __call__(self):
        return self.pa_type


@dataclass
class BinClassLabel(ClassLabel):
    positive_labels: Optional[Union[str, List[str]]] = None
    negative_labels: Optional[Union[str, List[str]]] = None
    num_classes: InitVar[Optional[int]] = 2
    _type: str = field(default="BinClassLabel", init=False, repr=False)
