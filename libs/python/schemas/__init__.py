"""Schema models generated from contracts."""

from importlib import metadata

from .base import ContractModel, to_dict


__all__ = ["__version__", "ContractModel", "to_dict"]


def __version__() -> str:
    try:
        return metadata.version("carbon254-libs-python")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "0.0.0-dev"

