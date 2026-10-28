"""Versioned, CPU-only Route navigation inventory and graded certification.

Rebuilds local-XML proxy plans with CARLA 0.9.15 Map/OpenDRIVE and its actual
GlobalRoutePlanner; no Client, server, simulation, model or training is started.
Route-level proxy acceptance never certifies a frame. Strict frame acceptance
also requires the frozen eight-check session evidence previously reviewed by
certified_overlay_v4 and fresh per-frame agreement. No Status overlay is written.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.b2d_adapter.experimental_route_preview import RoutePolyline
from tools.b2d_adapter.route_preview_contract import preview_command4
from tools.validation.b2d.online_global_plan import record_sha as planner_record_sha
from tools.validation.b2d.session_plan_agent import record_sha as session_record_sha
from tools.validation.b2d.route_source_certification import accept_match, dedupe_points
from tools.validation.b2d.collection_source_recovery import sha
from tools.training.diffusiondrive_v1.certified_overlay_v4 import CHECKS

VERSION = 'route-validation-inventory-v1'
BASE = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261023_status_integration'
CERT = BASE / 'online_certification_20261023'
FULL = Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/route_preview_v1')
MANIFEST = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/baseline_training_preflight/baseline_pssd_manifest.json'
B2D = Path('/home/aethelind/work/environment-build_test/Bench2Drive')
CARLA = Path('/home/aethelind/carla/CARLA_0.9.15')
RULE = dict(preview_m=20.0, lateral_threshold_m=2.0, max_projection_m=5.0)


class CachedRoute(RoutePolyline):
    """The shared geometry with cached lengths and exact vectorized projection."""
    def __post_init__(self):
        super().__post_init__()
        vectors = np.diff(self.points_xy, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        object.__setattr__(self, '_vectors', vectors)
        object.__setattr__(self, '_lengths', lengths)
        object.__setattr__(self, '_cumulative', np.r_[0., np.cumsum(lengths)])

    @property
    def segment_lengths(self):
        return self._lengths

    @property
    def cumulative_s(self):
        return self._cumulative

    def project(self, point_xy):
        p = np.asarray(point_xy, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError('finite 2D rear pose required')
        factors = np.clip(np.einsum('ij,ij->i', p-self.points_xy[:-1], self._vectors)/self._lengths**2,0,1)
        projected = self.points_xy[:-1]+factors[:,None]*self._vectors
        distances = np.linalg.norm(projected-p,axis=1)
        index = int(np.argmin(distances))
        return (float(self._cumulative[index]+factors[index]*self._lengths[index]),
                float(distances[index]),projected[index],index)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def stats(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return dict(n=len(a), p50=float(np.median(a)) if len(a) else None,
                p90=float(np.percentile(a, 90)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def load_xml_candidates(roots):
    candidates, failures = {}, []
    for folder in roots:
        for path in sorted(Path(folder).rglob('*.xml')):
            try:
                tree = ET.parse(path).getroot()
                for route in tree.findall('route'):
                    points = route.findall('waypoints/position') or route.findall('waypoint')
                    xyz = [[float(p.get('x')), float(p.get('y')), float(p.get('z', 0.0))] for p in points]
                    town = route.get('town')
                    if not town or len(xyz) < 2:
                        continue
                    key = digest(dict(town=town, xyz=xyz))
                    evidence = dict(path=str(path.resolve()), sha256=sha(path), xml_id=route.get('id'))
                    if key in candidates:
                        candidates[key]['xml_evidence'].append(evidence)
                    else:
                        candidates[key] = dict(key=key, town=town, xyz=xyz, xml_evidence=[evidence])
            except (ET.ParseError, OSError, ValueError) as exc:
                failures.append(dict(path=str(path), reason=str(exc)))
    return candidates, failures


def projection_metrics(points, poses):
    """Exact point-to-polyline distances and actual-body heading compatibility."""
    pts = np.asarray(dedupe_points([p[:2] for p in points]), dtype=float)
    if len(pts) < 2 or not len(poses):
        return dict(coverage=0.0, median_m=None, heading_compatible_rate=0.0)
    a, v = pts[:-1], np.diff(pts, axis=0)
    lengths2 = np.einsum('ij,ij->i', v, v)
    distances, headings = [], []
    for offset in range(0, len(poses), 128):
        block = np.asarray(poses[offset:offset+128], dtype=float)
        delta = block[:, None, :2]-a
        t = np.clip(np.einsum('nsi,si->ns', delta, v)/lengths2, 0, 1)
        d2 = np.sum((delta-t[:, :, None]*v)**2, axis=2)
        index = np.argmin(d2, axis=1)
        distances.extend(np.sqrt(d2[np.arange(len(block)), index]).tolist())
        tangent = np.arctan2(v[index, 1], v[index, 0])
        headings.extend(np.abs((block[:, 2]-tangent+np.pi)%(2*np.pi)-np.pi).tolist())
    return dict(coverage=float(np.mean(np.asarray(distances) <= 6.0)),
                median_m=float(np.median(distances)),
                heading_compatible_rate=float(np.mean(np.asarray(headings) <= np.pi/3)))


def candidate_accepted(metrics):
    return (metrics.get('median_m') is not None and accept_match(metrics['coverage'], metrics['median_m'])
            and metrics['heading_compatible_rate'] >= .8)


def intervals(rows, predicate, kind):
    result, run = [], None
    for row in sorted(rows, key=lambda r: r['frame']):
        if not predicate(row):
            run = None
            continue
        if run is None or row['frame'] != run['end_frame']+1:
            run = dict(route=row['route'], kind=kind, start_frame=row['frame'],
                       end_frame=row['frame'], frames=0, sample_ids=[])
            result.append(run)
        run['end_frame'] = row['frame']
        run['frames'] += 1
        run['sample_ids'].append(row['sample_id'])
    return result


def route_pass(rows, source_verified):
    """Diagnostic Route candidate; never a frame certification override."""
    comparable = [r for r in rows if r.get('comparable')]
    runs = intervals(rows, lambda r: r.get('command_mismatch', False), 'COMMAND_MISMATCH')
    coverage = len(comparable)/len(rows) if rows else 0.0
    agreement = sum(r.get('command_match', False) for r in comparable)/len(comparable) if comparable else 0.0
    geometry_rate = sum(r.get('geometry_matched',False) for r in comparable)/len(comparable) if comparable else 0.
    passed = bool(source_verified and coverage >= .8 and agreement >= .95 and geometry_rate >= .8
                  and not any(r.get('left_right_reversal') or r.get('geometry_jump') for r in rows)
                  and not any(r['frames'] >= 10 for r in runs))
    return passed, coverage, agreement


def route_states(row):
    """Independent aggregate diagnostics; these never promote a frame."""
    states = ['ROUTE_SOURCE_VERIFIED' if row['source_status'] == 'ROUTE_SOURCE_VERIFIED'
              and row['plan_status'] == 'PASS' else 'SOURCE_NOT_VERIFIED']
    comparable = row['comparable_frames']
    if (row['effective_coverage'] >= .8 and comparable
            and row['geometry_matched_frames']/comparable >= .8):
        states.append('ROUTE_GEOMETRY_MATCHED')
    if row['effective_coverage'] >= .8 and row['command_agreement_rate'] >= .95:
        states.append('ROUTE_COMMAND_MATCHED')
    if row['command_mismatch_frames']:
        states.append('COMMAND_MISMATCH')
    return states


def navigation_target(frozen_status, preview_command):
    """A missing Status is distinct from the legitimate Unknown one-hot."""
    if frozen_status is None:
        return preview_command, None, 'PREVIEW_ONLY_STATUS_MISSING'
    a = np.asarray(frozen_status, dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError('invalid_frozen_status8')
    command = frozen_status[:4]
    if command not in [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]:
        raise ValueError('invalid_frozen_navigation4')
    return command, command == preview_command, 'FROZEN_STATUS_NAVIGATION4'


def finalize_inventory(output, raw_root):
    """Add provenance/recovery inventory to our own completed CPU audit only."""
    routes = list(jsonl(output/'route_validation_index.jsonl'))
    candidates = json.loads((output/'xml_candidates.json').read_text())
    summary = json.loads((output/'route_validation_summary.json').read_text())
    recovery = []
    for row in routes:
        folder = raw_root/row['route']
        files = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        result_path = folder/'result.json'
        result, result_error = {}, None
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError) as exc:
                result_error = str(exc)
        records = result.get('_checkpoint', {}).get('records', [])
        record_ids = [x.get('route_id') for x in records if x.get('route_id')]
        ids = [m.group(1) for rid in record_ids
               if (m := re.fullmatch(r'RouteScenario_(\d+)_rep\d+', rid))]
        # An equal numeric id is a recovery lead, never evidence of identical
        # route geometry, source XML SHA or the original collection session.
        exact_id_leads = [dict(key=k, xml=e) for k, c in candidates.items()
                          if c['town'] == row['town'] for e in c['xml_evidence']
                          if e['xml_id'] in ids]
        recovery.append(dict(route=row['route'], source_status=row['source_status'],
            association_basis=row['association_basis'], root_files=files,
            result_path=str(result_path), result_sha256=sha(result_path) if result_path.is_file() else None,
            result_error=result_error, result_route_ids=record_ids,
            exact_id_leads=exact_id_leads, automatic_promotion=False))
    with (output/'route_certification_manifest.jsonl').open('x') as stream:
        for row in routes:
            row = dict(row, route_states=route_states(row),
                       frame_promotion_by_route=False, certification_review_pending=True)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    source_paths = [B2D/'docs/v004_update.md', B2D/'README.md', B2D/'tools/data_collect.py']
    dump(output/'source_recovery_inventory.json', dict(version=VERSION,
        source_scope='Local checkout and raw route root metadata; no remote download',
        supporting_sources=[dict(path=str(p),sha256=sha(p)) for p in source_paths if p.is_file()],
        raw_root=str(raw_root), records=recovery,
        no_original_session_replay_proven=True,
        topology_does_not_prove_route_intent=True))
    states = Counter(s for r in routes for s in route_states(r))
    dump(output/'route_certification_rollup.json', dict(version=VERSION,
        route_states=dict(states), association_basis=dict(Counter(r['association_basis'] for r in routes)),
        towns=dict(Counter(r['town'] for r in routes if r['plan_status']=='PASS')),
        routes_without_any_local_town_xml=sum(r['candidate_xml_count']==0 for r in routes),
        raw_result_present=sum(bool(r['result_sha256']) for r in recovery),
        routes_with_numeric_xml_id_lead=sum(bool(r['exact_id_leads']) for r in recovery),
        command_mismatch_frames=sum(r['command_mismatch_frames'] for r in routes),
        left_right_reversal_frames=sum(r['left_right_reversals'] for r in routes),
        status_formable=summary['status_formable'], frame_certified=summary['strict_frame_certified'],
        training_admitted=0, certification_review_pending=True))


def strict_frame_checks(row, frozen, pair, previous, session_ok, current_overlay_sha, index_sha):
    """Intersection with all existing eight checks; no Route-wide promotion."""
    reasons = []
    if not session_ok:
        reasons.append('SESSION_SOURCE_NOT_VERIFIED')
    if not previous or not all(previous.get('contract_checks', {}).get(k) is True for k in CHECKS):
        reasons.append('EXISTING_EIGHT_CHECK_CONTRACT_NOT_SATISFIED')
    if not row.get('session_command_match'):
        reasons.append('SESSION_COMMAND_MISMATCH_OR_INVALID')
    if not row.get('command_match') or not row.get('geometry_matched') or row.get('geometry_jump'):
        reasons.append('CURRENT_FRAME_COMMAND_OR_GEOMETRY_ANOMALY')
    source = frozen.get('source_feature_identity', {})
    if (source.get('pair_index_sha256') != index_sha or source.get('feature_sha256') != pair['feature_sha']
            or source.get('target_sha256') != pair['target_sha']):
        reasons.append('FROZEN_PAIR_IDENTITY_MISMATCH')
    if previous and previous.get('frozen_overlay_sha256') != current_overlay_sha:
        reasons.append('FROZEN_OVERLAY_SHA_MISMATCH')
    if previous and (previous.get('evidence',{}).get('session_capture_sha256') != row.get('session_evidence_sha256')
                     or previous.get('evidence',{}).get('planner_capture_sha256') != row.get('session_planner_evidence_sha256')):
        reasons.append('SESSION_EVIDENCE_SHA_MISMATCH')
    status, a0 = frozen.get('status8'), frozen.get('a0_motion4')
    if (not isinstance(status, list) or len(status) != 8 or frozen.get('motion_source') != 'A0'
            or not isinstance(a0, list) or len(a0) != 4 or status[4:] != a0
            or not np.isfinite(np.asarray(status, dtype=float)).all()):
        reasons.append('FROZEN_A0_STATUS_UNAVAILABLE')
    return not reasons, reasons


def map_file(carla_root, town):
    matches = [p for p in (carla_root/'CarlaUE4/Content/Carla/Maps').rglob(town+'.xodr')
               if p.stem == town]
    if len(matches) != 1:
        raise ValueError('map_not_unique_or_missing:'+town)
    return matches[0]


def plan_town(args):
    """Build actual planner graphs offline. CARLA.Client is deliberately unused."""
    import carla
    sys.path.insert(0, str(args.carla_root/'PythonAPI/carla'))
    import agents.navigation.global_route_planner as grp_module
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    if importlib.metadata.version('carla') != '0.9.15':
        raise ValueError('CARLA_0.9.15_required')
    candidates = json.loads((args.output/'xml_candidates.json').read_text())
    wanted = json.loads((args.output/'plan_requests.json').read_text())
    odr = map_file(args.carla_root, args.plan_town)
    carla_map = carla.Map(args.plan_town, odr.read_text())
    planner = GlobalRoutePlanner(carla_map, 1.0)
    evidence = dict(carla_version=importlib.metadata.version('carla'), carla_library=carla.__file__,
                    carla_library_sha256=sha(Path(carla.__file__).parent/'libcarla.cpython-39-x86_64-linux-gnu.so')
                    if (Path(carla.__file__).parent/'libcarla.cpython-39-x86_64-linux-gnu.so').exists() else None,
                    planner_path=grp_module.__file__, planner_sha256=sha(grp_module.__file__),
                    map_path=str(odr), map_sha256=sha(odr), map_name=carla_map.name,
                    hop_resolution_m=1.0, mode='OFFLINE_CARLA_MAP_FROM_OPENDRIVE_NO_SERVER')
    for key in wanted:
        item = candidates[key]
        if item['town'] != args.plan_town:
            continue
        try:
            nodes, options = [], []
            for start, end in zip(item['xyz'], item['xyz'][1:]):
                trace = planner.trace_route(carla.Location(*start), carla.Location(*end))
                if not trace:
                    raise ValueError('empty_planner_trace')
                for waypoint, option in trace:
                    point = waypoint.transform.location
                    nodes.append([point.x, point.y, point.z])
                    options.append(option.name)
            if len(dedupe_points([p[:2] for p in nodes])) < 2:
                raise ValueError('planner_plan_too_short')
            record = dict(schema='route-xml-dense-plan-v1', version=VERSION, key=key,
                          status='PASS', points_xyz=nodes, road_options=options,
                          xml_evidence=item['xml_evidence'], **evidence)
        except Exception as exc:
            record = dict(schema='route-xml-dense-plan-v1', version=VERSION, key=key,
                          status='FAIL', reason=type(exc).__name__+':'+str(exc), **evidence)
        dump(args.output/'plans'/f'{key}.json', record)
    print(json.dumps(dict(town=args.plan_town, map_sha256=evidence['map_sha256'])), flush=True)


def read_frozen(manifest, overlay_path, manifest_sha256):
    pairs, poses, invalid = {}, {}, []
    pair_path = Path(manifest['pair_index'])
    if sha(pair_path) != manifest['index_sha256']:
        raise ValueError('pair_index_sha_changed')
    with sqlite3.connect(pair_path.as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        pairs = {r['sample_id']: dict(r) for r in db.execute('SELECT * FROM pairs')}
    for source_name, spec in manifest['target_sources'].items():
        path = Path(spec['state'])/'index.sqlite'
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA query_only=ON')
            meta = dict(db.execute('SELECT key,value FROM meta'))
            if digest(json.loads(meta['producer'])) != spec['producer_sha256']:
                raise ValueError('target_producer_identity_changed')
            for r in db.execute("SELECT sample_id,route,frame,sha256,current_anno_sha256,reference_json FROM samples WHERE state='complete'"):
                sid = r['sample_id']
                pair = pairs.get(sid)
                if not pair or pair['target_source'] != source_name:
                    continue
                try:
                    if any(r[k] != pair[k] for k in ['route','frame']) or r['sha256'] != pair['target_sha'] or r['current_anno_sha256'] != pair['current_anno_sha']:
                        raise ValueError('frozen_target_pose_identity_mismatch')
                    pose = json.loads(r['reference_json'])['rear_world_carla_xy_yaw_rad']
                    if len(pose) != 3 or not np.isfinite(pose).all():
                        raise ValueError('invalid_frozen_rear_pose')
                    poses[sid] = pose
                except (ValueError, KeyError, TypeError) as exc:
                    invalid.append(dict(sample_id=sid, reason=str(exc)))
    frozen = {}
    for r in jsonl(overlay_path):
        sid = r['sample_id']
        if sid in frozen or sid not in pairs or any(r['sample_identity'].get(k) != pairs[sid][k] for k in ['route','frame','split']):
            raise ValueError('frozen_overlay_identity_mismatch:'+sid)
        if r.get('source_feature_identity',{}).get('manifest_sha256') != manifest_sha256:
            raise ValueError('frozen_overlay_manifest_identity_mismatch:'+sid)
        frozen[sid] = r
    if set(frozen) != set(pairs):
        raise ValueError('frozen_overlay_incomplete')
    return pairs, poses, frozen, invalid


def run(args):
    import pyarrow.parquet as pq
    if (args.output/'route_validation_index.jsonl').exists():
        raise FileExistsError('Use a fresh explicit versioned directory')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    pairs, poses, frozen, invalid_poses = read_frozen(manifest, args.overlay, sha(args.manifest))
    by_route = defaultdict(list)
    for sid, pair in pairs.items():
        by_route[pair['route']].append(sid)
    global_rows = list(jsonl(args.preview_root/'route_navigation_manifest.jsonl'))
    routes = [r['route'] for r in global_rows]
    if len(routes) != 1100 or len(set(routes)) != len(routes):
        raise ValueError('1100_unique_routes_required')
    candidates, xml_errors = load_xml_candidates(args.xml_roots)
    towns = defaultdict(list)
    for key, item in candidates.items():
        towns[item['town']].append(key)
    old_association = {r['route']: r for r in json.loads((CERT/'xml_preview_comparison/original_route_source_association.json').read_text())['records']}
    sessions = {}
    for path in (CERT/'session_capture_20261023/session_captures').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('schema') != 'leaderboard-session-plan-v1':
            continue
        meta = r['metadata']
        sessions[meta['dataset_route']] = dict(record=r, path=str(path), file_sha256=sha(path),
            content_sha_valid=session_record_sha(r)==r.get('sha256'))
    existing_strict = {r['sample_id']: r for r in jsonl(CERT/'certified_overlay_v4/certified_status_overlay_v4.jsonl')}
    old_planners = {}
    for path in (CERT/'carla_plan_capture').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('schema') == 'online-global-plan-v1':
            old_planners[r['metadata']['route']] = dict(record=r, file_sha256=sha(path),
                content_sha_valid=planner_record_sha(r)==r.get('sha256'))
    requests, associations = set(), []
    for route in routes:
        town = 'Town'+route.split('_Town')[1].split('_')[0]
        sid_list = sorted(by_route.get(route, []), key=lambda s: pairs[s]['frame'])
        route_poses = [poses[s] for s in sid_list if s in poses]
        screened = []
        # Every local candidate is screened; shortlisted candidates are then
        # checked against ALL frozen poses, not only the legacy five frames.
        sample = route_poses[::max(1, len(route_poses)//25)]
        for key in towns.get(town, []):
            m = projection_metrics(candidates[key]['xyz'], sample)
            if candidate_accepted(m):
                full_metrics = projection_metrics(candidates[key]['xyz'], route_poses)
                if candidate_accepted(full_metrics):
                    screened.append(dict(key=key, **full_metrics))
        session = sessions.get(route)
        primary = [c for c in screened if any(Path(e['path']).name in
            ('bench2drive220.xml','bench2drive_0.0.4_val.xml') for e in candidates[c['key']]['xml_evidence'])]
        eligible = primary or screened
        chosen, basis = None, 'NO_RELIABLE_XML'
        if session and old_association.get(route, {}).get('accepted'):
            meta = session['record']['metadata']
            matching = [key for key in towns.get(town, []) if any(
                x['sha256']==meta.get('xml_sha256') and x['xml_id']==meta.get('xml_id')
                for x in candidates[key]['xml_evidence'])]
            if len(matching)==1:
                chosen, basis = matching[0], 'EXISTING_SESSION_XML_BINDING'
        if chosen is None and len(eligible)==1:
            chosen, basis = eligible[0]['key'], 'UNIQUE_LOCAL_XML_GEOMETRY_PROXY' if primary else 'UNIQUE_LEGACY_XML_PROXY'
        elif chosen is None and len(eligible)>1:
            basis = 'AMBIGUOUS_LOCAL_XML_CANDIDATES'
        if chosen:
            requests.add(chosen)
        legacy = old_association.get(route, {})
        associations.append(dict(route=route, town=town, frozen_status_rows=len(sid_list),
            frozen_pose_rows=len(route_poses), candidate_xml_count=len(towns.get(town, [])),
            accepted_candidates=screened, selected_key=chosen, association_basis=basis,
            source_status='ROUTE_SOURCE_VERIFIED' if chosen and basis!='UNIQUE_LEGACY_XML_PROXY' else 'SOURCE_NOT_VERIFIED',
            source_scope='LOCAL_XML_PROXY' if chosen else 'UNVERIFIED',
            legacy_five_frame_accepted=bool(legacy.get('accepted')),
            legacy_candidate=legacy.get('best'), session_evidence_available=bool(session)))
    dump(args.output/'xml_candidates.json', candidates)
    dump(args.output/'plan_requests.json', sorted(requests))
    (args.output/'plans').mkdir(exist_ok=True)
    for town in sorted({candidates[k]['town'] for k in requests}):
        with (args.output/f'planner_{town}.log').open('x') as log:
            result = subprocess.run([str(args.carla_python), str(Path(__file__).resolve()), '--plan-town', town,
                '--output', str(args.output), '--carla-root', str(args.carla_root)], stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(phase='planner_town',town=town,returncode=result.returncode)),flush=True)
    association_by_route = {r['route']:r for r in associations}
    overlay_sha, index_sha = sha(args.overlay), sha(manifest['pair_index'])
    output_rows, route_rows, anomalies, usable = [], [], [], []
    for number, route in enumerate(routes, 1):
        association = association_by_route[route]
        key = association['selected_key']
        plan_path = args.output/'plans'/f'{key}.json' if key else None
        plan_record = json.loads(plan_path.read_text()) if plan_path and plan_path.exists() else None
        plan = CachedRoute(dedupe_points([p[:2] for p in plan_record['points_xyz']]), 'CARLA_XML_PROXY', True) if plan_record and plan_record['status']=='PASS' else None
        plan_file_sha = sha(plan_path) if plan_path and plan_path.exists() else None
        session = sessions.get(route)
        session_record = session['record'] if session else None
        session_plan = CachedRoute(dedupe_points([p[:2] for p in session_record['raw']['world_points']]), 'REAL_SESSION_DENSE',True) if session_record else None
        meta = session_record['metadata'] if session_record else {}
        old_planner = old_planners.get(route)
        old_meta = old_planner['record']['metadata'] if old_planner else {}
        evidence_matches = bool(key and any(x['sha256']==meta.get('xml_sha256') and x['xml_id']==meta.get('xml_id') for x in candidates[key]['xml_evidence']))
        local_sources = {'leaderboard_evaluator_sha256': B2D/'leaderboard/leaderboard/leaderboard_evaluator.py',
            'route_scenario_sha256':B2D/'leaderboard/leaderboard/scenarios/route_scenario.py',
            'route_manipulation_sha256':B2D/'leaderboard/leaderboard/utils/route_manipulation.py'}
        session_ok = bool(session and session['content_sha_valid'] and old_planner
            and old_planner['content_sha_valid'] and evidence_matches
            and meta.get('dataset_route')==route and meta.get('town')==association['town']
            and meta.get('xml_sha256')==old_meta.get('xml_sha256')
            and meta.get('carla_server_version')=='0.9.15'
            and str(meta.get('map_name','')).split('/')[-1]==association['town']
            and str(old_meta.get('map_name','')).split('/')[-1]==association['town']
            and all(path.is_file() and meta.get(field)==sha(path) for field,path in local_sources.items()))
        columns=['frame','preview_valid','command4_json','preview_point_xy_json']
        stored = {int(r['frame']):r for r in pq.read_table(args.preview_root/'routes'/route/'frame_route_preview.parquet',columns=columns).to_pylist()}
        frame_rows=[]
        previous=None
        for sid in sorted(by_route.get(route,[]),key=lambda s:pairs[s]['frame']):
            pair, frozen_row = pairs[sid], frozen[sid]
            p = poses.get(sid)
            raw = stored.get(pair['frame'],{})
            command=json.loads(raw['command4_json']) if raw.get('preview_valid') and raw.get('command4_json') else None
            status = frozen_row.get('status8')
            status_present = status is not None
            # Status is the authoritative comparison target where it exists;
            # missing Status permits only a separately labelled geometry audit.
            command, frozen_preview_agreement, comparison_target = navigation_target(status, command)
            target=json.loads(raw['preview_point_xy_json']) if raw.get('preview_point_xy_json') else None
            preview=preview_command4(plan,p[:2],p[2],**RULE) if plan is not None and p is not None else None
            valid=bool(preview and preview.get('preview_valid'))
            comparable=bool(valid and command is not None)
            match=bool(comparable and preview['command4']==command)
            a0=frozen_row.get('a0_motion4')
            motion_ok=bool(isinstance(a0,list) and len(a0)==4 and np.isfinite(a0).all() and frozen_row.get('motion_source')=='A0')
            point_distance=float(math.dist(preview['preview_point_xy'],target)) if valid and target is not None else None
            row=dict(sample_id=sid,route=route,frame=pair['frame'],split=pair['split'],
                frozen_status_available=status_present,
                frozen_status_preview_command_match=frozen_preview_agreement,
                comparison_target=comparison_target,
                source_scope='LEADERBOARD_SESSION_LOCAL_XML_PROXY' if session_ok else 'LOCAL_XML_PROXY' if plan else 'UNVERIFIED',
                source_status=association['source_status'] if plan else 'SOURCE_NOT_VERIFIED',
                preview_valid=valid,comparable=comparable,command4=command,
                reference_command4=preview.get('command4') if valid else None,
                command_match=match,command_mismatch=bool(comparable and not match),
                left_right_reversal=bool(comparable and (command,preview['command4']) in
                    [([1.,0.,0.,0.],[0.,0.,1.,0.]),([0.,0.,1.,0.],[1.,0.,0.,0.])]),
                preview_point_distance_m=point_distance,
                geometry_matched=bool(comparable and point_distance is not None and point_distance <= 1.),
                reference_projection_m=preview.get('projection_distance_m') if preview else None,
                reference_progress_m=preview.get('route_progress_m') if preview else None,
                reference_preview_xy=preview.get('preview_point_xy') if preview else None,
                reference_failure_reason=preview.get('reason') if not valid and preview else None,
                geometry_jump=False,frozen_pose_available=p is not None,
                status_formable=bool(valid and motion_ok),training_admitted=False,
                map_sha256=plan_record.get('map_sha256') if plan_record else None,
                planner_sha256=plan_record.get('planner_sha256') if plan_record else None,
                plan_file_sha256=plan_file_sha,
                session_evidence_sha256=session_record.get('sha256') if session_record else None)
            row['session_planner_evidence_sha256']=old_planner['record'].get('sha256') if old_planner else None
            if valid and previous and previous['preview_valid'] and pair['frame']==previous['frame']+1:
                old_pose=poses[previous['sample_id']]
                displacement=math.dist(p[:2],old_pose[:2])
                jump=math.dist(row['reference_preview_xy'],previous['reference_preview_xy'])
                if jump>max(5.,2*displacement+2.):
                    row['geometry_jump']=True;previous['geometry_jump']=True
            if p is not None and session_plan is not None:
                session_preview=preview_command4(session_plan,p[:2],p[2],**RULE)
                row['session_command_match']=bool(session_preview.get('preview_valid') and command is not None and session_preview['command4']==command)
            else:
                row['session_command_match']=False
            frame_rows.append(row);previous=row
        for row in frame_rows:
            sid=row['sample_id']
            certified,reasons=strict_frame_checks(row,frozen[sid],pairs[sid],existing_strict.get(sid),session_ok,overlay_sha,index_sha)
            row['frame_certified']=certified;row['strict_rejection_reasons']=reasons
            row['states']=[row['source_status']]
            if row['geometry_matched']:row['states'].append('ROUTE_GEOMETRY_MATCHED')
            if row['command_match']:row['states'].append('ROUTE_COMMAND_MATCHED')
            if row['command_mismatch']:row['states'].append('COMMAND_MISMATCH')
            if row['geometry_jump']:row['states'].append('GEOMETRY_DISCONTINUITY')
            if certified:row['states'].append('FRAME_CERTIFIED')
        mismatch_runs=intervals(frame_rows,lambda r:r['command_mismatch'],'COMMAND_MISMATCH')
        jump_runs=intervals(frame_rows,lambda r:r['geometry_jump'],'GEOMETRY_DISCONTINUITY')
        invalid_runs=intervals(frame_rows,lambda r:bool(plan) and not r['preview_valid'],'REFERENCE_PREVIEW_INVALID')
        for run in mismatch_runs:
            run['isolated']=run['frames']==1;run['long_run']=run['frames']>=10
            run['physical_duration_s']=None
        anomalies.extend(mismatch_runs+jump_runs+invalid_runs)
        usable.extend(intervals(frame_rows,lambda r:r['command_match'] and r['geometry_matched'] and r['status_formable'] and not r['geometry_jump'],'USABLE_PROXY_MATCHED_FRAGMENT'))
        passed,coverage,agreement=route_pass(frame_rows,bool(plan) and association['source_status']=='ROUTE_SOURCE_VERIFIED')
        route_row=dict(**association,route_pass_candidate=passed,validation_frames=len(frame_rows),
            plan_status='PASS' if plan else 'MISSING_OR_FAILED',
            evidence_grade='REAL_LEADERBOARD_SESSION_LOCAL_XML_PROXY' if session_ok else 'CARLA_0.9.15_LOCAL_XML_PROXY' if plan else 'SOURCE_NOT_VERIFIED',
            preview_valid_frames=sum(r['preview_valid'] for r in frame_rows),
            comparable_frames=sum(r['comparable'] for r in frame_rows),effective_coverage=coverage,
            geometry_matched_frames=sum(r['geometry_matched'] for r in frame_rows),
            command_agreement_rate=agreement,left_right_reversals=sum(r['left_right_reversal'] for r in frame_rows),
            command_mismatch_frames=sum(r['command_mismatch'] for r in frame_rows),
            isolated_anomaly_frames=sum(r['frames']==1 for r in mismatch_runs),
            continuous_mismatch_intervals=sum(r['frames']>=2 for r in mismatch_runs),
            long_mismatch_intervals=sum(r['frames']>=10 for r in mismatch_runs),
            geometry_jump_frames=sum(r['geometry_jump'] for r in frame_rows),
            preview_point_distance_m=stats([r['preview_point_distance_m'] for r in frame_rows]),
            status_formable=sum(r['status_formable'] for r in frame_rows),
            frame_certified=sum(r['frame_certified'] for r in frame_rows),training_admitted=0,
            map_sha256=plan_record.get('map_sha256') if plan_record else None,
            planner_sha256=plan_record.get('planner_sha256') if plan_record else None,
            plan_file_sha256=plan_file_sha)
        route_rows.append(route_row);output_rows.extend(frame_rows)
        if number%100==0:print(json.dumps(dict(phase='compare',routes=number,frames=len(output_rows))),flush=True)
    for name, values in [('route_validation_index.jsonl',route_rows),('frame_validation_index.jsonl',output_rows),
                          ('anomaly_interval_index.jsonl',anomalies),('usable_fragment_index.jsonl',usable)]:
        with (args.output/name).open('x') as stream:
            for value in values:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(version=VERSION,routes=len(routes),frozen_pair_rows=len(pairs),frozen_pose_rows=len(poses),
        compared_frames=sum(r['comparable'] for r in output_rows),
        frozen_status_present_frames=sum(r['frozen_status_available'] for r in output_rows),
        frozen_status_comparable_frames=sum(r['comparable'] and r['frozen_status_available'] for r in output_rows),
        preview_only_comparable_frames=sum(r['comparable'] and not r['frozen_status_available'] for r in output_rows),
        proxy_plan_routes=sum(r['plan_status']=='PASS' for r in route_rows),
        session_verified_routes=sum(r['evidence_grade']=='REAL_LEADERBOARD_SESSION_LOCAL_XML_PROXY' for r in route_rows),
        source_verified_proxy_routes=sum(r['source_status']=='ROUTE_SOURCE_VERIFIED' and r['plan_status']=='PASS' for r in route_rows),
        route_pass_candidates=sum(r['route_pass_candidate'] for r in route_rows),
        route_pass_rate_all=sum(r['route_pass_candidate'] for r in route_rows)/len(routes),
        route_pass_rate_validated=sum(r['route_pass_candidate'] for r in route_rows)/max(1,sum(r['plan_status']=='PASS' for r in route_rows)),
        isolated_anomaly_frames=sum(r['isolated_anomaly_frames'] for r in route_rows),
        continuous_mismatch_intervals=sum(r['continuous_mismatch_intervals'] for r in route_rows),
        long_mismatch_intervals=sum(r['long_mismatch_intervals'] for r in route_rows),
        geometry_jump_frames=sum(r['geometry_jump'] for r in output_rows),
        status_formable=sum(r['status_formable'] for r in output_rows),
        proxy_command_matched_status=sum(r['command_match'] and r['geometry_matched'] and r['status_formable'] and not r['geometry_jump'] for r in output_rows),
        strict_frame_certified=sum(r['frame_certified'] for r in output_rows),
        previous_eight_check_frames=len(existing_strict),training_admitted=0,
        remaining_source_not_verified=sum(r['plan_status']!='PASS' or r['source_status']!='ROUTE_SOURCE_VERIFIED' for r in route_rows),
        ambiguous_xml_routes=sum(r['association_basis']=='AMBIGUOUS_LOCAL_XML_CANDIDATES' for r in route_rows),
        no_reliable_xml_routes=sum(r['association_basis']=='NO_RELIABLE_XML' for r in route_rows),
        xml_files=len({x['path'] for c in candidates.values() for x in c['xml_evidence']}),
        xml_unique_geometries=len(candidates),xml_parse_errors=xml_errors,invalid_frozen_poses=invalid_poses,
        rule=RULE,association_rule='coverage>=0.8 within6m; median<=8m; heading<=60deg rate>=0.8; unique geometry or existing session XML binding',
        route_candidate_rule='comparable coverage>=0.8; agreement>=0.95; geometry point delta<=1m rate>=0.8; no L/R reversal; no >=10-frame mismatch; no geometry jump',
        interval_rule='numeric consecutive frame ids; >=10 frames is diagnostic, not a physical time standard',
        certification_rule='fresh agreement AND frozen existing eight-check eligible row AND verified session identity; Route pass never promotes frame',
        overlay_sha256=overlay_sha,pair_index_sha256=index_sha,
        training_started=False,status_comparison_experiment=False,carla_server_started=False,
        source_scope='Local XML proxy; real leaderboard captures are later local-XML sessions, not proof of original data-collection replay')
    dump(args.output/'route_validation_summary.json',summary)
    finalize_inventory(args.output, args.raw_root)
    print(json.dumps({k:summary[k] for k in ['routes','proxy_plan_routes','compared_frames','route_pass_candidates','strict_frame_certified','remaining_source_not_verified']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-root',type=Path,default=FULL)
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    p.add_argument('--overlay',type=Path,default=BASE/'overlay/full_status_overlay_manifest.jsonl')
    p.add_argument('--xml-roots',type=Path,nargs='+',default=[B2D/'leaderboard/data',B2D/'scenario_runner/srunner/data'])
    p.add_argument('--carla-root',type=Path,default=CARLA)
    p.add_argument('--carla-python',type=Path,default=Path('/home/aethelind/carla/client_env/bin/python'))
    p.add_argument('--plan-town')
    p.add_argument('--raw-root',type=Path,default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1'))
    p.add_argument('--finalize-existing',action='store_true',help='Inventory raw root metadata and summarize an existing audit; no planner rerun')
    args=p.parse_args()
    if args.plan_town:plan_town(args)
    elif args.finalize_existing:finalize_inventory(args.output,args.raw_root)
    else:run(args)


if __name__=='__main__':main()
