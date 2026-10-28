"""Route-level full generation for Route Preview V1.

The canonical /data source may be unavailable; callers may point at a readable
alternate mount.  This script refuses to start unless the inventory scan
succeeds.  It is deliberately separate from the training Feature Producer and
writes only to an explicit derived-data root.

Properties:
  * one Route per atomic shard directory;
  * a DONE marker containing the source signature for idempotent resume;
  * optional process-level Route parallelism (``--workers``);
  * explicit per-frame failures, never silent frame drops;
  * per-route geometry/work limits and a worker deadline;
  * pathological Routes are quarantined under ``quarantine/`` and skipped on
    later ``--resume`` unless ``--retry-quarantined`` is supplied;
  * explicit provenance and navigation source;
  * no raw camera/LiDAR or HD-map copies.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.b2d_adapter.experimental_causal_state import A0, causal_state  # noqa: E402
from tools.b2d_adapter.experimental_route_navigation import (  # noqa: E402
    CompactMap, build_segments, reconstruct,
)
from tools.b2d_adapter.route_preview_contract import preview_command4, status8  # noqa: E402
from tools.validation.b2d.route_navigation_preview import (  # noqa: E402
    read_route, route_id_of, sha256, town_of,
)

DEFAULT_SOURCE = Path('/data/datasets/Bench2Drive-V0.0.4/v1')
DEFAULT_OUTPUT = Path('/data/datasets/derived/Bench2Drive-V0.0.4/route_preview_v1')
DEFAULT_COMPACT = ROOT / 'outputs/diffusiondrive_v1/route_preview_data_audit/20261022_readonly'
DEFAULT_MAX_SEGMENT_POINTS = 200_000
DEFAULT_MAX_SEGMENT_LENGTH_M = 100_000.0
DEFAULT_MAX_ROUTE_POINTS = 400_000
DEFAULT_MAX_PREVIEW_WORK = 100_000_000
DEFAULT_ROUTE_TIMEOUT_S = 600.0
QUARANTINE_VERSION = 'route-preview-v1.1-resource-guard'


class RouteResourceExceeded(RuntimeError):
    """A route reconstruction is too large to safely preview."""

    def __init__(self, details):
        self.details = details
        reasons = ', '.join(details.get('violations', [])) or 'resource_limit_exceeded'
        super().__init__(reasons)


class RouteTimeout(RuntimeError):
    """A route exceeded its bounded worker deadline."""

    def __init__(self, seconds):
        self.seconds = float(seconds)
        super().__init__(f'route_timeout_{self.seconds:g}s')


def safe_name(name):
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def route_resource_profile(segments):
    """Return cheap size/work measurements without deriving a new path."""
    point_counts = []
    lengths = []
    work = 0
    for segment in segments or []:
        points = segment.get('points_xy') or []
        point_count = len(points)
        frame_count = max(0, int(segment.get('end_frame', -1)) -
                          int(segment.get('start_frame', 0)) + 1)
        point_counts.append(point_count)
        lengths.append(float(segment.get('length_m') or 0.0))
        work += point_count * frame_count
    return {
        'segment_count': len(point_counts),
        'max_segment_points': max(point_counts, default=0),
        'total_segment_points': sum(point_counts),
        'max_segment_length_m': max(lengths, default=0.0),
        'preview_work_units': int(work),
    }


def enforce_route_resources(segments, limits):
    """Reject pathological reconstruction geometry before previewing frames."""
    profile = route_resource_profile(segments)
    violations = []
    if profile['max_segment_points'] > int(limits['max_segment_points']):
        violations.append('max_segment_points')
    if profile['total_segment_points'] > int(limits['max_route_points']):
        violations.append('max_route_points')
    if profile['max_segment_length_m'] > float(limits['max_segment_length_m']):
        violations.append('max_segment_length_m')
    if profile['preview_work_units'] > int(limits['max_preview_work']):
        violations.append('max_preview_work')
    if violations:
        raise RouteResourceExceeded({
            'resource_profile': profile,
            'limits': dict(limits),
            'violations': violations,
        })
    return profile


def quarantine_path(output_root, route_name):
    return Path(output_root) / 'quarantine' / f'{safe_name(route_name)}.json'


def read_quarantine(output_root, route_name, source_signature_value):
    path = quarantine_path(output_root, route_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if payload.get('source_signature') != source_signature_value:
        return None
    return payload


def write_quarantine(output_root, route_name, source_signature_value, reason, details):
    path = quarantine_path(output_root, route_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'route-preview-quarantine-v1',
        'version': QUARANTINE_VERSION,
        'route': route_name,
        'source_signature': source_signature_value,
        'reason': reason,
        'details': details,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)
    return path


def remove_quarantine(output_root, route_name):
    path = quarantine_path(output_root, route_name)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def route_deadline(seconds):
    """Bound Python-level route work; the caller still writes no shard."""
    if not seconds or float(seconds) <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):
        raise RouteTimeout(seconds)

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def source_signature(route_dir):
    result = route_dir / 'result.json'
    try:
        return sha256(result) if result.is_file() else None
    except OSError:
        return None


def frame_row(route, town, frame, preview, source, trust='HIGH', status8_value=None,
              motion=None, segment_id=None):
    command = preview.get('command4')
    return {
        'route': route, 'town': town, 'frame': int(frame),
        'preview_status': preview.get('status'),
        'preview_valid': bool(preview.get('preview_valid')),
        'preview_reason': preview.get('reason'),
        'preview_category': preview.get('category'),
        'command4_json': json.dumps(command) if command is not None else None,
        'status8_json': json.dumps(status8_value) if status8_value is not None else None,
        'motion4_json': json.dumps(motion) if motion is not None else None,
        'navigation_source': source,
        'provenance': 'RECONSTRUCTED_ROUTE' if preview.get('preview_valid') else 'NOT_VERIFIED',
        'trust': trust,
        'route_progress_m': preview.get('route_progress_m'),
        'progress_fraction': preview.get('progress_fraction'),
        'preview_point_xy_json': json.dumps(preview.get('preview_point_xy')),
        'local_xy_m_json': json.dumps(preview.get('local_xy_m')),
        'lateral_m': preview.get('lateral_m'),
        'heading_delta_deg': preview.get('heading_delta_deg'),
        'projection_distance_m': preview.get('projection_distance_m'),
        'segment_id': segment_id,
        'uturn_status': preview.get('uturn_status'),
        'v1_compatible': preview.get('v1_compatible'),
        'source_pose': 'read_route_rear_axle',
        'source_pose_certified': False,
    }


def route_inventory(source_root):
    try:
        entries = list(os.scandir(source_root))
    except OSError as exc:
        return [], f'{source_root}: {exc}'
    names = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                names.append(entry.name)
        except OSError as exc:
            return names, f'{entry.path}: {exc}'
    return sorted(names), None


def write_route_shard(route_name, source_root, output_root, compact_dir,
                      limits=None, route_timeout_s=DEFAULT_ROUTE_TIMEOUT_S):
    """Worker entry: generate one Route and atomically publish its shard."""
    routes_root = output_root / 'routes'
    route_out = routes_root / safe_name(route_name)
    signature = source_signature(source_root / route_name)
    temp = None
    old = None
    try:
        with route_deadline(route_timeout_s):
            rows, segments, route_failures, manifest = generate_route(
                route_name, source_root, output_root, compact_dir, limits=limits)
        routes_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=routes_root))
        pd.DataFrame(rows).to_parquet(temp / 'frame_route_preview.parquet', index=False)
        with (temp / 'route_segment_manifest.jsonl').open('w') as stream:
            for row in segments:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (temp / 'failures.jsonl').open('w') as stream:
            for row in route_failures:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(temp / 'manifest.json', manifest)
        write_json(temp / 'DONE.json', {'route': route_name,
                                        'source_signature': manifest.get('source_signature'),
                                        'version': 'route-preview-v1.1'})
        if route_out.exists():
            old = route_out.with_name(route_out.name + '.old-' + uuid.uuid4().hex)
            os.replace(route_out, old)
        os.replace(temp, route_out)
        temp = None
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
            old = None
        remove_quarantine(output_root, route_name)
        return {'status': 'ok', 'route': route_name, 'manifest': manifest}
    except RouteResourceExceeded as exc:
        marker = write_quarantine(output_root, route_name, signature,
                                  'resource_limit_exceeded', exc.details)
        return {'status': 'quarantined', 'route': route_name,
                'reason': 'resource_limit_exceeded', 'details': exc.details,
                'quarantine': str(marker)}
    except RouteTimeout as exc:
        marker = write_quarantine(output_root, route_name, signature,
                                  'route_timeout', {'timeout_s': exc.seconds})
        return {'status': 'quarantined', 'route': route_name,
                'reason': 'route_timeout', 'details': {'timeout_s': exc.seconds},
                'quarantine': str(marker)}
    except Exception as exc:
        return {'status': 'failed', 'route': route_name,
                'reason': type(exc).__name__, 'message': str(exc)}
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        if old is not None:
            # Keep the old complete shard if publication failed; it is still
            # better than deleting the only copy.
            pass


def generate_route(route_name, source_root, output_root, compact_dir, replace=False,
                   limits=None):
    route_dir = source_root / route_name
    town = town_of(route_name)
    compact = compact_dir / f'{town}_centers.npz'
    metadata = compact_dir / f'{town}_center_metadata.json'
    if not compact.is_file() or not metadata.is_file():
        raise RuntimeError('compact_map_unavailable')
    town_map = CompactMap(compact, metadata, source=f'{town}_HD_map.npz -> compact')
    try:
        anno_paths = sorted((route_dir / 'anno').glob('*.json.gz'))
    except OSError as exc:
        raise RuntimeError(f'annotation_directory_unreadable: {exc}') from exc
    frames, poses, z_values, raw_current = read_route(route_dir)
    if not poses:
        raise RuntimeError('no_readable_annotation_frames')
    frame_ids = {int(path.stem.split('.')[0]) for path in anno_paths}
    readable_ids = {int(frame) for frame in frames}
    missing_frame_ids = sorted(frame_ids - readable_ids)
    result = reconstruct(town_map, poses, z_values, route_name=route_name)
    segments = result.get('segments') or []
    # Stop before previewing a pathological path; the shard is never created.
    resource_profile = enforce_route_resources(segments, limits or {
        'max_segment_points': DEFAULT_MAX_SEGMENT_POINTS,
        'max_route_points': DEFAULT_MAX_ROUTE_POINTS,
        'max_segment_length_m': DEFAULT_MAX_SEGMENT_LENGTH_M,
        'max_preview_work': DEFAULT_MAX_PREVIEW_WORK,
    })
    # Segment paths are the only source for Level 2; no full-route fallback.
    preview_rows = []
    failure_rows = []
    for frame in missing_frame_ids:
        row = frame_row(route_name, town, frame, {
            'status': 'FAIL', 'preview_valid': False,
            'reason': 'annotation_unreadable_or_invalid', 'category': 'Unknown',
            'command4': [0, 0, 0, 1], 'v1_compatible': False,
        }, 'NOT_VERIFIED', trust='NOT_VERIFIED')
        preview_rows.append(row)
        failure_rows.append({'route': route_name, 'frame': frame,
                             'reason': 'annotation_unreadable_or_invalid'})
    for segment in segments:
        for index in range(int(segment['start_frame']), int(segment['end_frame']) + 1):
            if index >= len(poses):
                continue
            frame = frames[index]
            preview = preview_command4(segment['points_xy'], poses[index][:2], poses[index][2],
                                       max_projection_m=5.0)
            motion = None
            vector = None
            if index >= 2:
                state = causal_state(poses[index - 2:index + 1], method=A0, dt=.1)
                motion = np.concatenate((state.velocity, state.acceleration)).astype(float).tolist()
                if preview.get('preview_valid'):
                    vector = status8(preview['command4'], motion)
            preview_rows.append(frame_row(route_name, town, frame, preview,
                                          'RECONSTRUCTED_ROUTE_SEGMENT',
                                          status8_value=vector, motion=motion,
                                          segment_id=segment['segment_id']))
    in_segment = {int(row['frame']) for row in preview_rows}
    for frame in frames:
        if int(frame) not in in_segment:
            preview_rows.append(frame_row(route_name, town, frame, {
                'status': 'FAIL', 'preview_valid': False,
                'reason': 'not_in_verified_segment', 'category': 'Unknown',
                'command4': [0, 0, 0, 1], 'v1_compatible': False,
            }, 'NOT_VERIFIED', trust='NOT_VERIFIED', segment_id=None))
    # Deterministic order.
    preview_rows.sort(key=lambda row: row['frame'])
    segment_rows = []
    for segment in segments:
        segment_rows.append({
            'route': route_name, 'town': town, 'segment_id': segment['segment_id'],
            'start_frame': int(segment['start_frame']), 'end_frame': int(segment['end_frame']),
            'points_xy': segment['points_xy'], 'length_m': segment['length_m'],
            'join_gaps_m': segment['join_gaps_m'], 'break_before': segment['break_before'],
            'break_after': segment['break_after'], 'navigation_source': segment['navigation_source'],
            'provenance': segment['provenance'], 'trust': segment['trust'],
        })
    failed = sum(not row['preview_valid'] for row in preview_rows)
    manifest = {
        'route': route_name, 'town': town, 'route_id': route_id_of(route_name),
        'source_signature': source_signature(route_dir),
        'frames': len(frames), 'annotation_files': len(anno_paths),
        'missing_annotation_frames': len(missing_frame_ids),
        'preview_valid': len(preview_rows) - failed,
        'preview_failed': failed, 'segment_count': len(segments),
        'decision_count': len(result.get('events') or []),
        'resource_profile': resource_profile,
        'source': 'RECONSTRUCTED_ROUTE', 'provenance': 'RECONSTRUCTED_ROUTE',
        'reference_path_online_available': False,
    }
    return preview_rows, segment_rows, failure_rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--compact-dir', type=Path, default=DEFAULT_COMPACT)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int,
                        default=max(1, min(8, os.cpu_count() or 1)),
                        help='parallel Route workers; default 8 on this host')
    parser.add_argument('--max-segment-points', type=int,
                        default=DEFAULT_MAX_SEGMENT_POINTS)
    parser.add_argument('--max-route-points', type=int,
                        default=DEFAULT_MAX_ROUTE_POINTS)
    parser.add_argument('--max-segment-length-m', type=float,
                        default=DEFAULT_MAX_SEGMENT_LENGTH_M)
    parser.add_argument('--max-preview-work', type=int,
                        default=DEFAULT_MAX_PREVIEW_WORK)
    parser.add_argument('--route-timeout-s', type=float,
                        default=DEFAULT_ROUTE_TIMEOUT_S,
                        help='per-route worker deadline; 0 disables the deadline')
    parser.add_argument('--retry-quarantined', action='store_true',
                        help='ignore matching quarantine markers and retry those routes')
    parser.add_argument('--only-quarantined', action='store_true',
                        help='retry only routes with existing quarantine markers')
    parser.add_argument('--routes-file', type=Path, default=None,
                        help='newline-delimited Route names for an explicit versioned repair subset')
    args = parser.parse_args()
    if args.only_quarantined and not args.resume:
        parser.error('--only-quarantined requires --resume')
    if args.only_quarantined:
        args.retry_quarantined = True
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    for name in ('max_segment_points', 'max_route_points',
                 'max_segment_length_m', 'max_preview_work'):
        if getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    if args.route_timeout_s < 0:
        parser.error('--route-timeout-s must be nonnegative')
    limits = {
        'max_segment_points': args.max_segment_points,
        'max_route_points': args.max_route_points,
        'max_segment_length_m': args.max_segment_length_m,
        'max_preview_work': args.max_preview_work,
    }
    names, error = route_inventory(args.source_root)
    try:
        output_exists = args.output_root.exists()
    except OSError as exc:
        output_exists = None
        error = error or f'{args.output_root}: {exc}'
    preflight = {
        'schema': 'route-preview-generation-preflight-v1',
        'source_root': str(args.source_root), 'output_root': str(args.output_root),
        'output_exists': output_exists, 'routes_observed': len(names),
        'source_scan_error': error,
        'resource_limits': limits,
        'route_timeout_s': args.route_timeout_s,
        'only_quarantined': args.only_quarantined,
    }
    if error:
        print(json.dumps({**preflight, 'status': 'BLOCKED_SOURCE_IO'}, ensure_ascii=False))
        return 2
    if args.inventory_only:
        print(json.dumps({**preflight, 'status': 'INVENTORY_PASS'}, ensure_ascii=False))
        return 0
    args.output_root.mkdir(parents=True, exist_ok=False if not args.resume else True)
    routes_root = args.output_root / 'routes'
    routes_root.mkdir(exist_ok=True)
    if args.only_quarantined:
        quarantine_names = set()
        for marker in (args.output_root / 'quarantine').glob('*.json'):
            try:
                payload = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            route = payload.get('route')
            if route in names:
                quarantine_names.add(route)
        names = [route for route in names if route in quarantine_names]
        if not names:
            print(json.dumps({**preflight, 'status': 'NO_QUARANTINED_ROUTES'},
                             ensure_ascii=False))
            return 0
    if args.routes_file is not None:
        requested = [line.strip() for line in args.routes_file.read_text().splitlines() if line.strip()]
        known = set(names)
        unknown = sorted(set(requested) - known)
        if unknown:
            parser.error('--routes-file contains routes outside the source inventory: ' + ','.join(unknown[:5]))
        keep = set(requested)
        names = [route for route in names if route in keep]
        preflight['routes_file'] = str(args.routes_file)
        preflight['routes_filtered_from'] = len(known)
        preflight['routes_selected'] = len(names)
    if args.limit:
        names = names[:args.limit]
    global_manifest = []
    failures = []
    quarantined = []
    pending = []
    for route in names:
        route_out = routes_root / safe_name(route)
        done_path = route_out / 'DONE.json'
        signature = source_signature(args.source_root / route)
        if args.resume and done_path.is_file():
            try:
                if json.loads(done_path.read_text()).get('source_signature') == signature:
                    global_manifest.append(json.loads((route_out / 'manifest.json').read_text()))
                    continue
            except (OSError, ValueError):
                pass
        if args.resume and not args.retry_quarantined:
            marker = read_quarantine(args.output_root, route, signature)
            if marker is not None:
                quarantined.append({
                    'route': route, 'reason': marker.get('reason'),
                    'quarantine': str(quarantine_path(args.output_root, route)),
                    'source_signature': signature, 'resumed_skip': True,
                    'details': marker.get('details'),
                })
                continue
        pending.append(route)

    def handle_result(result, number):
        if result.get('status') == 'ok':
            global_manifest.append(result['manifest'])
        elif result.get('status') == 'quarantined':
            quarantined.append(result)
        else:
            failures.append({'route': result['route'], 'reason': result.get('reason'),
                             'message': result.get('message')})
        if number % 25 == 0 or number == len(pending):
            print(json.dumps({'event': 'route_progress', 'routes': number,
                              'of': len(pending), 'workers': args.workers,
                              'written': len(global_manifest),
                              'quarantined': len(quarantined),
                              'failed': len(failures)}), flush=True)

    if args.workers > 1 and len(pending) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(write_route_shard, route, args.source_root,
                                        args.output_root, args.compact_dir,
                                        limits=limits,
                                        route_timeout_s=args.route_timeout_s): route
                       for route in pending}
            for number, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                except Exception as exc:
                    route = futures[future]
                    result = {'status': 'failed', 'route': route,
                              'reason': type(exc).__name__, 'message': str(exc)}
                handle_result(result, number)
    else:
        for number, route in enumerate(pending, 1):
            handle_result(write_route_shard(route, args.source_root, args.output_root,
                                            args.compact_dir, limits=limits,
                                            route_timeout_s=args.route_timeout_s), number)

    global_manifest.sort(key=lambda row: row['route'])
    failures.sort(key=lambda row: row['route'])
    quarantined.sort(key=lambda row: row['route'])
    (args.output_root / 'route_navigation_manifest.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in global_manifest) + '\n')
    (args.output_root / 'failures.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in failures) + '\n')
    (args.output_root / 'quarantine_manifest.jsonl').write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in quarantined) + '\n')
    write_json(args.output_root / 'generation_manifest.json', {
        **preflight,
        'status': 'DONE' if not failures and not quarantined else 'PARTIAL',
        'routes_written': len(global_manifest), 'routes_failed': len(failures),
        'routes_quarantined': len(quarantined), 'failures': len(failures),
        'workers': args.workers, 'retry_quarantined': args.retry_quarantined,
        'version': 'route-preview-v1.1',
    })
    print(json.dumps({'event': 'generation_done', 'routes': len(global_manifest),
                      'failed': len(failures), 'quarantined': len(quarantined),
                      'output': str(args.output_root)}, ensure_ascii=False))
    return 0 if not failures and not quarantined else 1


if __name__ == '__main__':
    raise SystemExit(main())
