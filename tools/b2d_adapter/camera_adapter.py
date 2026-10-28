"""Independent B2D Camera pixel policies; official NAVSIM code stays unchanged."""

from .errors import AdapterError
from .route_index import CAMERAS


def proportional_camera_feature(frame):
    """Version b2d-proportional-crop-v1: one crop/stitch/linear resize/ToTensor.

    This policy is defined for RGB uint8 1600x900 sources. Other resolutions are
    rejected rather than implicitly scaled or assigned the same cache identity.
    """
    import cv2
    import numpy as np
    from torchvision.transforms import ToTensor

    names = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
    for name in names:
        image = frame.cameras.get(name)
        if image is None or image.shape != (900, 1600, 3) or image.dtype != np.uint8:
            raise AdapterError("invalid_camera_shape", {"camera": name,
                               "required": "RGB uint8 900x1600x3 for b2d-proportional-crop-v1"})
    left, front, right = (frame.cameras[name] for name in names)
    stitched = np.concatenate((left[23:-23, 346:-346], front[23:-23, :],
                               right[23:-23, 346:-346]), axis=1)
    return ToTensor()(cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_LINEAR))


def camera_metadata(frame):
    return {name: {"shape": list(image.shape), "dtype": str(image.dtype),
                   "color": "RGB", "calibration": frame.sensors.get(name),
                   "source": frame.provenance["source_files"].get(name)}
            for name, image in frame.cameras.items()}


def candidate_camera_feature(frame):
    import numpy as np
    from PIL import Image

    if set(frame.cameras) != set(CAMERAS):
        raise AdapterError("missing_modality", "three decoded front cameras required")
    # Deliberately no copying of NAVSIM's source-pixel crop (different resolution/FOV).
    images = [np.asarray(Image.fromarray(frame.cameras[name]).resize((width, 256), Image.BILINEAR))
              for name, width in zip(CAMERAS, (256, 512, 256))]
    value = np.concatenate(images, axis=1).transpose(2, 0, 1).astype(np.float32) / 255.0
    return {"candidate_camera_feature": value, "candidate_only": True,
            "policy": "full B2D RGB views resized independently to 256/512/256x256; NOT equivalent geometry"}
