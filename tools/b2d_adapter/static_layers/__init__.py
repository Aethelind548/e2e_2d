"""Independent static-map labels; never imported by the original V1 builders."""
from .opendrive import MapStaticLayers, StaticLayerRules, ROADMARK_TYPES, load_static_query

__all__ = ['MapStaticLayers', 'StaticLayerRules', 'ROADMARK_TYPES', 'load_static_query']
