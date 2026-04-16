from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ApiError:
    code: str
    message: str
    detail: Optional[str] = None

