"""Base utilities for schema validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ContractModel(BaseModel):
    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        frozen = True


def to_dict(model: ContractModel) -> dict[str, Any]:
    return model.dict(by_alias=True)


__all__ = ["ContractModel", "to_dict"]

