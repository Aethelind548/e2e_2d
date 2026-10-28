"""No synthetic features; offline shared algorithms are linked explicitly."""

from typing import Any, Mapping
import math

from agent.common.errors import FeatureError, PendingImplementationError
from agent.common.types import CoordinateFrame, EgoState, ModelFeatures, NavigationState, SensorObservation
from agent.config.settings import FeatureConfig


class OnlineFeatureAdapter:
    # Confirmed unbatched shapes in v1_builders/native_builders, not guesses.
    V1_SHAPES = {"camera_feature": (3, 256, 1024),
                 "lidar_feature": (1, 256, 256), "status_feature": (8,)}

    def __init__(self, config: FeatureConfig):
        self.config = config

    def build_features(self, observation: SensorObservation, ego: EgoState,
                       navigation: NavigationState) -> ModelFeatures:
        if ego.timestamp != observation.timestamp or navigation.timestamp != observation.timestamp:
            raise FeatureError("Observation, ego state and navigation must share the current simulation timestamp")
        # TODO: verify BGRA->RGB decoding and camera ids/calibration, online
        # LiDAR sensor->Actor reference including accumulation/height, causal
        # motion and navigation policy. Reuse offline helpers below afterwards.
        raise PendingImplementationError("Online sensor-to-V1 feature alignment is not implemented")

    def camera_from_verified_rgb(self, cameras: Mapping[str, Any]):
        """Reuse offline crop only for already verified RGB 900x1600 arrays."""
        if self.config.camera_policy != "b2d-proportional-crop-v1":
            raise FeatureError("Explicit b2d-proportional-crop-v1 camera policy required")
        from types import SimpleNamespace
        from tools.b2d_adapter.camera_adapter import proportional_camera_feature
        return proportional_camera_feature(SimpleNamespace(cameras=cameras))

    def build_lidar(self, observation: SensorObservation, ego: EgoState):
        # TODO: feed verified Actor-frame XYZ and explicit geometry policy to
        # tools.b2d_adapter.lidar_adapter.candidate_lidar_feature; live sensor
        # XYZ cannot be assumed identical to annotated LAZ Actor-frame XYZ.
        raise PendingImplementationError("Online LiDAR frame/height/accumulation contract is pending")

    def build_status(self, ego: EgoState, navigation: NavigationState):
        """Return the model Status tensor from the single online builder.

        The builder already computed Navigation4 from the dense plan and Motion4
        from the A0 rule.  This adapter only enforces shape/dtype/one-hot and the
        policy identity; it never re-derives navigation or falls back.
        """
        import torch
        from agent.navigation.status_builder import STATUS_POLICY, validate_command4
        if navigation.policy != STATUS_POLICY:
            raise FeatureError(f"Status policy must be {STATUS_POLICY}")
        if navigation.timestamp != ego.timestamp:
            raise FeatureError("Status and ego state must share the current timestamp")
        if navigation.status_validity != 'VALID' or navigation.status8 is None:
            raise FeatureError(f"Status8 is not valid: {navigation.status_reason}")
        if len(navigation.status8) != 8:
            raise FeatureError("Status8 must contain 8 values")
        if not all(math.isfinite(float(v)) for v in navigation.status8):
            raise FeatureError("Status8 must be finite")
        validate_command4(tuple(navigation.status8[:4]))
        tensor = torch.tensor(navigation.status8, dtype=torch.float32)
        if tuple(tensor.shape) != (8,):
            raise FeatureError("status_feature must be float32 (8,)")
        return tensor

    @classmethod
    def validate_features(cls, features: ModelFeatures) -> None:
        """Validate supplied real tensors without constructing model inputs."""
        import torch
        if not math.isfinite(features.timestamp.seconds):
            raise FeatureError("Model feature timestamp must be finite")
        if features.frame != CoordinateFrame.MODEL_REAR:
            raise FeatureError("Model features must declare the verified rear/left coordinate frame")
        tensors = features.as_dict()
        for name, shape in cls.V1_SHAPES.items():
            tensor = tensors[name]
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.float32):
                raise FeatureError(f"{name} must be finite float32 {shape}")
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1 or any(device.type == "meta" for device in devices):
            raise FeatureError("Model features must share one materialized device")
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all().item():
                raise FeatureError(f"{name} must be finite float32 {cls.V1_SHAPES[name]}")
            if name != "status_feature" and not ((tensor >= 0).all() & (tensor <= 1).all()).item():
                raise FeatureError(f"{name} values must be in [0,1]")
        from agent.navigation.status_builder import validate_command4
        validate_command4(tuple(features.status_feature[:4].detach().cpu().tolist()))

    def reset(self) -> None:
        """No preprocessing state allocated in V0.1."""
