"""Versioned seven-class label rules; these are not model configurations."""
from dataclasses import asdict, dataclass
import hashlib
import json

from .errors import AdapterError


@dataclass(frozen=True)
class V1LabelRules:
    name: str = 'v1_original_7class'
    version: str = 'original-v2'
    guardrail: str = 'none'
    actor_projection: str = 'planar'
    actor_floor: str = 'none'
    road_support_distance_m: float = 2.
    road_support_z_spread_m: float = .75
    actor_deck_clearance_m: float = 1.
    guardrail_deck_clearance_m: float = .5

    @property
    def enriched(self):
        return self.name == 'v1_semantic_enriched'

    def identity(self):
        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()


def label_rules(value=None):
    if value is None or value == 'v1_original_7class':
        return V1LabelRules()
    if value == 'v1_semantic_enriched':
        return V1LabelRules(name=value, version='m4-v1', guardrail='flat-pivot-vertical-surface-v1',
                            actor_projection='tilted-world-xy-hull-v1', actor_floor='diagnose')
    if isinstance(value, dict):
        try:
            base = label_rules(value.get('name'))
            result = V1LabelRules(**dict(base.identity(), **value))
        except (TypeError, ValueError) as exc:
            raise AdapterError('invalid_label_rules', str(exc)) from exc
        if not result.enriched and result != V1LabelRules():
            raise AdapterError('invalid_label_rules', 'original rules cannot be overridden')
        if (result.version != base.version or result.guardrail != base.guardrail
                or result.actor_projection != base.actor_projection
                or result.actor_floor not in ('none', 'diagnose', 'filter-supported-road')
                or any(type(v) not in (int, float) or not 0 < v < float('inf') for v in
                       (result.road_support_distance_m, result.road_support_z_spread_m,
                        result.actor_deck_clearance_m, result.guardrail_deck_clearance_m))):
            raise AdapterError('invalid_label_rules', value)
        return result
    raise AdapterError('invalid_label_rules', value)
