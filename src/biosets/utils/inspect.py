import inspect
import sys

from biosets.utils import logging

logger = logging.get_logger(__name__)


class InvalidColumnSelectionError(Exception):
    pass


def get_kwargs(kwargs, method):
    method_args = [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p != p.VAR_KEYWORD
    ]
    return {k: kwargs[k] for k in method_args if k in kwargs}


def get_required_args(method):
    """retrieves arguments with no default values"""
    return [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.default == p.empty and p != p.VAR_KEYWORD
    ]


def pa_array_from_pandas_kwargs(kwargs: dict):
    return {
        "mask": kwargs.get("mask", None),
        "type": kwargs.get("type", None),
        "safe": kwargs.get("safe", True),
        "memory_pool": kwargs.get("memory_pool", None),
    }


def pa_table_from_pandas_kwargs(kwargs: dict):
    schema = kwargs.get("schema", None)
    preserve_index = kwargs.get("preserve_index", False)
    nthreads = kwargs.get("nthreads", None)
    columns = kwargs.get("columns", None)
    safe = kwargs.get("safe", True)
    return {
        "schema": schema,
        "preserve_index": preserve_index,
        "nthreads": nthreads,
        "columns": columns,
        "safe": safe,
    }


def pa_table_to_pandas_kwargs(kwargs: dict):
    # memory_pool=None, categories=None, bool strings_to_categorical=False, bool zero_copy_only=False, bool integer_object_nulls=False, bool date_as_object=True, bool timestamp_as_object=False, bool use_threads=True, bool deduplicate_objects=True, bool ignore_metadata=False, bool safe=True, bool split_blocks=False, bool self_destruct=False, unicode maps_as_pydicts=None, types_mapper=None, bool coerce_temporal_nanoseconds=False
    memory_pool = kwargs.get("memory_pool", None)
    categories = kwargs.get("categories", None)
    strings_to_categorical = kwargs.get("strings_to_categorical", False)
    zero_copy_only = kwargs.get("zero_copy_only", False)
    integer_object_nulls = kwargs.get("integer_object_nulls", False)
    date_as_object = kwargs.get("date_as_object", True)
    timestamp_as_object = kwargs.get("timestamp_as_object", False)
    use_threads = kwargs.get("use_threads", True)
    deduplicate_objects = kwargs.get("deduplicate_objects", True)
    ignore_metadata = kwargs.get("ignore_metadata", False)
    safe = kwargs.get("safe", True)
    split_blocks = kwargs.get("split_blocks", False)
    self_destruct = kwargs.get("self_destruct", False)
    maps_as_pydicts = kwargs.get("maps_as_pydicts", None)
    types_mapper = kwargs.get("types_mapper", None)
    coerce_temporal_nanoseconds = kwargs.get("coerce_temporal_nanoseconds", False)

    return {
        "memory_pool": memory_pool,
        "categories": categories,
        "strings_to_categorical": strings_to_categorical,
        "zero_copy_only": zero_copy_only,
        "integer_object_nulls": integer_object_nulls,
        "date_as_object": date_as_object,
        "timestamp_as_object": timestamp_as_object,
        "use_threads": use_threads,
        "deduplicate_objects": deduplicate_objects,
        "ignore_metadata": ignore_metadata,
        "safe": safe,
        "split_blocks": split_blocks,
        "self_destruct": self_destruct,
        "maps_as_pydicts": maps_as_pydicts,
        "types_mapper": types_mapper,
        "coerce_temporal_nanoseconds": coerce_temporal_nanoseconds,
    }


def pa_array_to_pandas_kwargs(kwargs: dict):
    return {
        "memory_pool": kwargs.get("memory_pool", None),
        "categories": kwargs.get("categories", None),
        "strings_to_categorical": kwargs.get("strings_to_categorical", False),
        "zero_copy_only": kwargs.get("zero_copy_only", False),
        "integer_object_nulls": kwargs.get("integer_object_nulls", False),
        "date_as_object": kwargs.get("date_as_object", True),
        "timestamp_as_object": kwargs.get("timestamp_as_object", False),
        "use_threads": kwargs.get("use_threads", True),
        "deduplicate_objects": kwargs.get("deduplicate_objects", True),
        "ignore_metadata": kwargs.get("ignore_metadata", False),
        "safe": kwargs.get("safe", True),
        "split_blocks": kwargs.get("split_blocks", False),
        "self_destruct": kwargs.get("self_destruct", False),
        "maps_as_pydicts": kwargs.get("maps_as_pydicts", None),
        "types_mapper": kwargs.get("types_mapper", None),
        "coerce_temporal_nanoseconds": kwargs.get("coerce_temporal_nanoseconds", False),
    }


def pa_table_from_arrays_kwargs(kwargs: dict):
    schema = kwargs.get("schema", None)
    metadata = kwargs.get("metadata", None)
    return {"schema": schema, "metadata": metadata}


def ds_init_kwargs(kwargs: dict):
    return get_kwargs(kwargs, sys.modules["biosets"].Dataset.__init__)


def np_array_kwargs(kwargs: dict):
    dtype = kwargs.get("dtype", None)
    copy = kwargs.get("copy", False)
    order = kwargs.get("order", "K")
    subok = kwargs.get("subok", True)
    ndmin = kwargs.get("ndmin", 0)
    like = kwargs.get("like", None)
    return {
        "dtype": dtype,
        "copy": copy,
        "order": order,
        "subok": subok,
        "ndmin": ndmin,
        "like": like,
    }
