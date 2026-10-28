"""Bounded recovery of tilted vehicles proven irrelevant to both V1 target heads.

Never complete missing future motion. Baseline indexes/builders stay unchanged;
the stricter geometric evidence and isolated producer permit explicit review.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')

from tools.b2d_adapter import FrameReader
from tools.b2d_adapter.errors import AdapterError
from tools.b2d_adapter.v1_builders import DiffusionDriveV1TargetBuilder
from tools.training.diffusiondrive_v1 import target_cache as core, target_jobs as jobs
from tools.training.diffusiondrive_v1.features_cache import (
    StoragePause, digest, durable_json, file_digest, pin)

RULE = 'outside-detection-and-zero-planar-and-3d-bev-v1'


def irrelevant_tilt(box, raw, geometry, official):
    """Conservative: retain any projected/official pixel or selected detection.

    No height cutoff or yaw flattening is introduced. Require both the existing
    official planar box mask and the actual projected eight-corner 3D mask empty.
    """
    import numpy as np
    from types import SimpleNamespace
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from tools.b2d_adapter.v1_bev import bev_annotations
    from tools.b2d_adapter.v1_geometry import actor_local_hull, polygon_mask
    isolated = copy.copy(raw)
    isolated.annotation = dict(raw.annotation, bounding_boxes=[box])
    annotations, _ = bev_annotations(isolated, geometry)
    if annotations.names != ['vehicle']:
        raise AdapterError('invalid_vehicle_box', 'Recovery only handles real vehicle boxes')
    _, labels = official._compute_agent_targets(annotations)
    official_pixels = int(official._compute_box_mask(annotations, [TrackedObjectType.VEHICLE]).sum())
    hull, _ = actor_local_hull(box, raw, geometry)
    projected_pixels = int(polygon_mask(official, [hull]).sum())
    selected = bool(labels.any())
    # Also require no continuous 3D polygon overlap: quantization alone is not
    # proof that an object just below one pixel is spatially irrelevant.
    from shapely.geometry import box as rectangle
    polygon_intersects = hull.intersects(rectangle(0, -32, 32, 32))
    return {'actor_id': str(box['id']), 'pitch_roll_yaw_deg': list(box['rotation']),
            'detection_selected': selected, 'official_planar_bev_pixels': official_pixels,
            'projected_3d_bev_pixels': projected_pixels, 'projected_3d_intersects_bev': polygon_intersects,
            'local_center_xy_m': annotations.boxes[0, :2].tolist(),
            'safely_irrelevant': not selected and official_pixels == 0 and projected_pixels == 0 and not polygon_intersects}


class IrrelevantActorRecoveryBuilder(DiffusionDriveV1TargetBuilder):
    def build_agent_targets(self, raw):
        import numpy as np
        from tools.b2d_adapter.navsim_bridge import is_b2d_vehicle
        vehicles = [b for b in raw.annotation['bounding_boxes'] if isinstance(b, dict) and b.get('class') == 'vehicle']
        ids = [str(b.get('id')) for b in vehicles]
        if len(ids) != len(set(ids)) or any(not is_b2d_vehicle(b) or 'id' not in b for b in vehicles):
            raise AdapterError('invalid_vehicle_box', 'Recovery retains all actor identity validation')
        omit = set(); evidence = []
        for box in vehicles:
            rotation = np.asarray(box.get('rotation'), dtype=float)
            if rotation.shape != (3,) or not np.isfinite(rotation).all():
                raise AdapterError('invalid_vehicle_box', 'finite rotation3 required')
            if np.max(np.abs(rotation[:2])) > self.config.geometry.max_planar_tilt_deg:
                proof = irrelevant_tilt(box, raw, self.config.geometry, self.official_builder)
                evidence.append(proof)
                if proof['safely_irrelevant']: omit.add(id(box))
        filtered = copy.copy(raw)
        filtered.annotation = dict(raw.annotation, bounding_boxes=[b for b in raw.annotation['bounding_boxes'] if id(b) not in omit])
        # Original checks still reject every tilted actor that could affect either
        # head. Full BEV retains the unmodified raw annotation; proven masks are 0.
        result = super().build_agent_targets(filtered)
        result.provenance['recovery'] = {'rule': RULE, 'excluded_actor_ids': [p['actor_id'] for p in evidence if p['safely_irrelevant']], 'evidence': evidence}
        return result


def run(config, apply=False):
    import fcntl
    import torch
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration, bev_provider
    state = ROOT / config['target_state_dir']; state.mkdir(parents=True, exist_ok=True)
    source = ROOT / config['source_target_state_dir']
    with (state / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        model, native = model_configuration(config), native_configuration(config)
        base_producer = core.producer_identity(config, model, native)
        producer = dict(base_producer, recovery={'rule': RULE, 'implementation_sha256': file_digest(Path(__file__)),
                                               'missing_future_recovery': False})
        sha = digest(producer)
        source_db = sqlite3.connect('file:' + str((source / 'index.sqlite').resolve()) + '?mode=ro', uri=True); source_db.row_factory = sqlite3.Row
        if digest(json.loads(source_db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])) != digest(base_producer):
            raise StoragePause('Source baseline target producer changed')
        rows = [dict(r) for r in source_db.execute("SELECT * FROM samples WHERE state='ineligible' AND error_code='invalid_vehicle_box' ORDER BY route,frame")]
        missing_future = source_db.execute("SELECT COUNT(*) FROM samples WHERE state='ineligible' AND error_code='insufficient_future'").fetchone()[0]
        baseline_identity = digest(rows)
        source_db.close()
        provider = bev_provider(dict(config, bev_cache_dir=None))
        db = None; used = 0; spec = config['target_root']; root = Path(spec['path'])
        report = {'rule': RULE, 'applied': apply, 'source_geometry_rejects': len(rows), 'recovered': [], 'still_rejected': [],
                  'producer_sha256': sha, 'baseline_producer_sha256': digest(base_producer), 'future_boundary_recoveries': 0}
        if apply:
            core.storage_check(spec, True); root.mkdir(parents=True, exist_ok=True)
            marker = root / 'target_root.json'; identity = {'schema': core.SCHEMA, 'state_id': str(state.resolve()), 'root_id': spec['root_id']}
            if marker.exists():
                if json.loads(marker.read_text()) != identity: raise StoragePause('Recovery root owned by another state')
            else:
                if any(root.iterdir()): raise StoragePause('Unmarked nonempty recovery root')
                durable_json(marker, identity)
            db = jobs.open_index(state / 'index.sqlite'); pin(db, 'producer', producer); pin(db, 'dataset_id', config['dataset_id'])
            pin(db, 'target_root', spec); pin(db, 'source_geometry_reject_identity', baseline_identity)
            used = sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in root.rglob('*') if p.is_file())
        current_route = None
        try:
            for row in rows:
                if current_route != row['route']:
                    reader = FrameReader(jobs.AnnotationIndex.from_dataset_root(config['dataset_root'], row['route']))
                    builder = IrrelevantActorRecoveryBuilder(reader, native, model, provider); current_route = row['route']
                ref = reader.index.ref(row['frame']); raw = reader.read(ref)
                try:
                    agents = builder.build_agent_targets(raw)
                    proof = agents.provenance['recovery']
                    if apply:
                        prior = db.execute('SELECT * FROM samples WHERE sample_id=?', (row['sample_id'],)).fetchone()
                        if prior and prior['state'] == 'complete':
                            core.load_target_row(dict(prior), spec, sha)
                        else:
                            candidate = dict(row, state='pending', error_code=None, error=None)
                            cached, pixels, allocated = jobs.write_target(config, candidate, producer, reader, builder, native, provider, used)
                            cols = list(cached)
                            db.execute('INSERT OR REPLACE INTO samples(' + ','.join(cols) + ') VALUES(' + ','.join('?' for _ in cols) + ')', list(cached.values()))
                            db.execute('INSERT OR REPLACE INTO bev_counts VALUES(?,?)', (row['sample_id'], json.dumps(pixels)))
                            db.commit(); used += allocated
                    report['recovered'].append({'sample_id': row['sample_id'], 'route': row['route'], 'frame': row['frame'], 'proof': proof})
                except AdapterError as exc:
                    classification = jobs.failure_state(exc)  # Environment faults safely pause.
                    reason = jobs.event(row, classification, sha, exc.code, str(exc))
                    report['still_rejected'].append(reason)
                    if db is not None:
                        db.execute('INSERT OR IGNORE INTO samples(sample_id,dataset_id,route,frame,state,error_code,error,updated) VALUES(?,?,?,?,?,?,?,?)',
                                   (row['sample_id'], config['dataset_id'], row['route'], row['frame'], classification, exc.code, str(exc), time.time())); db.commit()
            if db is not None:
                accepted = [dict(r) for r in db.execute("SELECT * FROM samples WHERE state='complete' ORDER BY route,frame")]
                durable_json(state / 'target_manifest.json', {'schema': core.SCHEMA, 'producer': producer, 'dataset_id': config['dataset_id'], 'root_id': spec['root_id'], 'accepted': accepted})
            report['counts'] = {'recovered': len(report['recovered']), 'still_rejected': len(report['still_rejected']), 'missing_future_not_recoverable': missing_future}
            durable_json(state / ('recovery_result.json' if apply else 'recovery_dry_run.json'), report)
            print(json.dumps(report['counts']))
            return report
        finally:
            if db is not None: db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'tools/training/diffusiondrive_v1/targets_recovery_irrelevant_v1.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try: run(json.loads(args.config.read_text()), args.apply)
    except (StoragePause, OSError) as exc:
        print('Recovery safely paused: ' + str(exc), file=sys.stderr); raise SystemExit(2)
