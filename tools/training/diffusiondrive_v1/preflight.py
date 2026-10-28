"""Bounded real-data readiness check. Exit 2 means full V1 training is blocked.

Does not instantiate the network, download weights, fabricate BEV labels, or
train a reduced model. No historical BNDP caches/weights are read as V1 proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "repositories/DiffusionDrive"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_info(tensor):
    import torch
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(tensor).all()),
            "min": float(tensor.min()), "max": float(tensor.max())}


def check(config):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tools.b2d_adapter import AdapterError, FrameReader, RouteIndex
    from tools.b2d_adapter.v1_builders import (DiffusionDriveV1Dataset, V1_ROOT,
                                               official_v1_module, v1_native_config, validate_v1_config)
    from tools.b2d_adapter.v1_bev import B2DOpenDriveBEVBuilder
    if type(config["sample_count"]) is not int or not 1 <= config["sample_count"] <= 8:
        raise ValueError("preflight permits only 1..8 explicit consecutive samples")
    if type(config["start_frame"]) is not int or config["start_frame"] < 2:
        raise ValueError("start_frame must be an integer >=2 for causal state")
    source = {}
    for name in ("transfuser_config", "transfuser_features", "transfuser_model_v2",
                 "transfuser_loss", "transfuser_agent", "transfuser_backbone"):
        module = official_v1_module(name)
        source[name] = {"path": module.__file__, "sha256": digest(module.__file__)}
    model_config = official_v1_module("transfuser_config").TransfuserConfig()
    from tools.training.diffusiondrive_v1.runtime import native_configuration
    native_config = native_configuration(config)
    anchor_path = ROOT / config["plan_anchor_path"]
    model_config.plan_anchor_path = str(anchor_path)
    if config.get("bkb_path"):
        model_config.bkb_path = str(ROOT / config["bkb_path"])
    report = {
        "scope": "official V1 imports + bounded real B2D adapter preflight; no network training",
        "python": sys.executable, "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "config": config,
        "repository": str(V1_ROOT), "sources": source,
        "git_branch": subprocess.check_output(["git", "-C", str(V1_ROOT), "branch", "--show-current"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "-C", str(V1_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "contract": validate_v1_config(native_config, model_config), "blockers": [],
        "stages": {name: "not_run" for name in ("real_data_batch", "full_model_forward", "official_full_loss",
                    "backward", "optimizer_update", "checkpoint_roundtrip", "tiny_overfit")},
    }
    anchor = np.load(anchor_path, allow_pickle=False)
    anchor_sha = digest(anchor_path)
    anchor_ok = anchor.shape == (20,8,2) and np.isfinite(anchor).all() and anchor_sha == config["expected_anchor_sha256"]
    report["anchor"] = {"path": str(anchor_path), "sha256": anchor_sha,
                        "shape": list(anchor.shape), "dtype": str(anchor.dtype), "valid": bool(anchor_ok)}
    if not anchor_ok:
        report["blockers"].append({"code": "invalid_anchor", "detail": report["anchor"]})
    report["image_pretrained"] = {"configured_path": model_config.bkb_path,
                                  "local_file_exists": Path(model_config.bkb_path).is_file(),
                                  "timm_cache_or_online_download_tested": False}
    if not report["image_pretrained"]["local_file_exists"]:
        report["blockers"].append({"code": "image_pretrained_unresolved",
                                   "detail": "configure an existing resnet34 image checkpoint before offline network initialization"})
    if not report["cuda_available"]:
        report["blockers"].append({"code": "cuda_unavailable", "detail": "GPU smoke cannot run in this session; CPU adapter checks remain valid"})
    bev_provider = B2DOpenDriveBEVBuilder(config["map_paths"]) if config.get("map_paths") else None
    if bev_provider is None:
        report["blockers"].append({"code": "missing_bev_semantic_map", "detail": "configure a real Route OpenDRIVE provider"})
    report["bev_map_candidates"] = []
    for path in (
        "/home/aethelind/carla/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/Town12/OpenDrive/Town12.xodr",
        "/home/aethelind/carla/AdditionalMaps_0.9.15/CarlaUE4/Content/Carla/Maps/Town12/OpenDrive/Town12.xodr",
    ):
        report["bev_map_candidates"].append({"path": path, "exists": Path(path).is_file(),
            "collection_version_and_label_generation_verified": False})
    try:
        reader = FrameReader(RouteIndex(config["route"]))
        refs = [reader.index.ref(config["start_frame"]+i) for i in range(config["sample_count"])]
        dataset = DiffusionDriveV1Dataset(reader, refs, native_config, model_config, config["lidar_decoder_python"], bev_provider)
        samples = [(dataset.build_sample(i) if bev_provider is not None else dataset.diagnose(i)) for i in range(len(dataset))]
        features, targets = next(iter(DataLoader([(s.features,s.targets) for s in samples],
                                                 batch_size=len(samples),num_workers=0)))
        report["batch"] = {"features": {k: tensor_info(v) for k,v in features.items()},
                           "targets" if bev_provider is not None else "partial_targets": {k: tensor_info(v) for k,v in targets.items()},
                           "valid_vehicle_counts": targets["agent_labels"].sum(1).tolist()}
        report["samples"] = [s.metadata for s in samples]
        report["stages"]["real_data_batch"] = "complete_targets_pass" if bev_provider is not None else "partial_targets_only_pass"
        try:
            if bev_provider is not None:
                report["full_dataset_gate"] = {"status": "complete_real_targets"}
            else:
                dataset[0]
        except AdapterError as exc:
            if exc.code != "missing_bev_semantic_map":
                raise
            report["full_dataset_gate"] = {"status": "pass", "code": exc.code, "detail": exc.detail}
        else:
            if bev_provider is None:
                raise AssertionError("full Dataset accepted missing BEV supervision")
    except (OSError, AdapterError) as exc:
        report["stages"]["real_data_batch"] = "blocked"
        report["blockers"].append({"code": getattr(exc,"code","data_access_error"),
                                   "type": type(exc).__name__, "errno": getattr(exc,"errno",None), "detail": str(exc)})
    report["ready_for_full_v1_training"] = not report["blockers"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("baseline.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (ROOT / config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/e2e_v1_mpl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    report = check(config)
    (output / "preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"ready_for_full_v1_training": report["ready_for_full_v1_training"],
                      "blockers": report["blockers"], "report": str(output / "preflight.json")},ensure_ascii=False,indent=2))
    return 0 if report["ready_for_full_v1_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
