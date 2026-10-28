"""CARLA Mercator inverse from local Zoo planner, with explicit map reference."""

import math
from numbers import Real
from typing import Tuple

from agent.common.errors import ConfigurationError, SensorError

EARTH_RADIUS_M = 6378137.0


class CarlaGeoReference:
    def __init__(self, lat_ref_deg: float, lon_ref_deg: float):
        if (any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                for v in (lat_ref_deg, lon_ref_deg))
                or not -90 < lat_ref_deg < 90 or not -180 <= lon_ref_deg <= 180):
            raise ConfigurationError("Explicit finite CARLA lat_ref/lon_ref in valid geographic bounds required")
        self.lat_ref_deg, self.lon_ref_deg = lat_ref_deg, lon_ref_deg
        self.scale = math.cos(math.radians(lat_ref_deg))
        self.origin_x = math.radians(lon_ref_deg) * EARTH_RADIUS_M * self.scale
        self.origin_y = self._mercator_y(lat_ref_deg)

    def _mercator_y(self, latitude: float) -> float:
        return math.log(math.tan(math.radians(latitude + 90.) / 2.)) * EARTH_RADIUS_M * self.scale

    def to_world_xy(self, latitude: float, longitude: float) -> Tuple[float, float]:
        if (not all(math.isfinite(v) for v in (latitude, longitude))
                or not -90 < latitude < 90 or not -180 <= longitude <= 180):
            raise SensorError("GNSS latitude/longitude outside finite CARLA Mercator domain")
        return (math.radians(longitude) * EARTH_RADIUS_M * self.scale - self.origin_x,
                self.origin_y - self._mercator_y(latitude))
