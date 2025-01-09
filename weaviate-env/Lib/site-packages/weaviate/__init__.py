from importlib.metadata import version, PackageNotFoundError

from . import classes

try:
    __version__ = version("weaviate")
except PackageNotFoundError:
    __version__ = "unknown version"


def connect_to_custom(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def connect_to_embedded(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def connect_to_local(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def connect_to_wcs(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def connect_to_weaviate_cloud(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def Client(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def WeaviateClient(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def WeaviateAsyncClient(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def use_async_with_custom(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def use_async_with_embedded(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def use_async_with_local(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


def use_async_with_weaviate_cloud(*args, **kwargs):
    raise NotImplementedError(
        "This is a placeholder package. To use the weaviate python client please do `pip install weaviate-client` and uninstall this package with `pip remove weaviate`"
    )


__all__ = ["classes",]
