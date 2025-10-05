"""
Local fallback for DataQualityFramework used when shared package is unavailable.
This lightweight implementation supports only the methods required by ml-service.
"""

from typing import Dict, Any


class DataQualityFramework:
    def __init__(self) -> None:
        self._rules: Dict[str, Dict[str, Dict[str, float]]] = {}

    def register_metric_rules(self, domain: str, rules: Dict[str, Dict[str, float]], overwrite: bool = False) -> None:
        key = domain.lower()
        if key in self._rules and not overwrite:
            raise ValueError(f"Metric rules for {domain!r} already exist; pass overwrite=True to replace.")
        self._rules[key] = rules

    def get_metric_rules(self, domain: str) -> Dict[str, Dict[str, float]]:
        return self._rules.get(domain.lower(), {})
