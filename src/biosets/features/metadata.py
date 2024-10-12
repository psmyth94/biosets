from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import pyarrow as pa
from datasets.features.features import Value, string_to_arrow


@dataclass
class ValueWithMetadata(Value):
    metadata: dict = field(default_factory=dict)
    _type: str = field(default="ValueWithMetadata", init=False, repr=False)


@dataclass
class Metadata:
    """Metadata features that can be used to describe the dataset."""

    feature: Any = None
    id: Optional[str] = None
    dtype: str = "dict"
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Metadata", init=False, repr=False)

    def __post_init__(self):
        if self.feature:
            if isinstance(self.feature, Metadata):
                self.pa_type = self.feature.pa_type
                self.feature = self.feature.feature
        elif self.dtype == "dict":
            self.pa_type = pa.struct(
                {
                    k: string_to_arrow(v["dtype"])
                    if isinstance(v, dict)
                    else string_to_arrow(v.dtype)
                    for k, v in self.feature.items()
                }
            )
        else:
            if self.dtype == "double":  # fix inferred type
                self.dtype = "float64"
            if self.dtype == "float":  # fix inferred type
                self.dtype = "float32"
            self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type


@dataclass
class Sample:
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Sample", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        else:
            return value


@dataclass
class Batch:
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Batch", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        else:
            return value
