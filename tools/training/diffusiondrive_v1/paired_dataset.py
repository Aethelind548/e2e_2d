"""Strict supervision intersection and cache-only Dataset, separate from Features identity."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1.features_cache import digest, durable_json, load_feature_row, sample_id
from tools.training.diffusiondrive_v1.target_cache import (
    ROOT, CONTRACT_PATH, current_annotation_source, feature_records,
    producer_identity, reference_pose)
from tools.training.diffusiondrive_v1.target_storage import load_target_row

SCHEMA = 'b2d-v1-paired-cache-v1'
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))


def splits(config):
    from tools.training.diffusiondrive_v1.runtime import route_group
    mapping = {}; groups = {}; towns = {}
    for split, entries in config['data'].items():
        if split not in ('train', 'validation', 'test'): raise AdapterError('pair_split_mismatch', split)
        groups[split], towns[split] = set(), set()
        for e in entries:
            route = str((ROOT / e['route']).resolve().relative_to(Path(config['dataset_root']).resolve()))
            group = route_group(route); groups[split].add(group); towns[split].add(group.split(':')[0])
            for frame in e['frames']:
                sid = sample_id(config['dataset_id'], route, frame)
                if sid in mapping: raise AdapterError('pair_split_leakage', sid)
                mapping[sid] = {'split': split, 'route': route, 'frame': frame}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] & groups[b] or (config.get('paired_split_policy', 'town') == 'town' and towns[a] & towns[b]):
                raise AdapterError('pair_split_leakage', a + '/' + b)
    return mapping


def pair_checks(feature, target, expected, native, target_producer):
    f = feature['row']; t = target
    if (f['sample_id'] != t['sample_id'] or f['sample_id'] != sample_id(feature['dataset_id'], expected['route'], expected['frame'])
        or feature['dataset_id'] != t['dataset_id'] or f['route'] != t['route'] or f['frame'] != t['frame']
        or f['route'] != expected['route'] or f['frame'] != expected['frame']):
        raise AdapterError('pair_sample_id_mismatch', f['sample_id'])
    fsource = current_annotation_source(json.loads(f['source_json']), f['route'], f['frame'])
    tsource = current_annotation_source(json.loads(t['sources_json']), t['route'], t['frame'])
    if fsource['sha256'] != t['current_anno_sha256'] or tsource['sha256'] != t['current_anno_sha256']:
        raise AdapterError('pair_source_mismatch', f['sample_id'])
    from tools.b2d_adapter.v1_label_rules import label_rules
    if target_producer['rules'] != label_rules().identity() or target_producer['geometry'] != native.geometry.provenance():
        raise AdapterError('pair_coordinate_mismatch', 'Geometry/label rules differ')


def build_manifest(config, output):
    from tools.b2d_adapter import FrameReader, RouteIndex
    from tools.training.diffusiondrive_v1.runtime import model_configuration, native_configuration
    selection = splits(config); records = feature_records(config)
    native = native_configuration(config); producer = producer_identity(config, model_configuration(config), native)
    state = ROOT / config['target_state_dir']
    db = sqlite3.connect('file:' + str((state / 'index.sqlite').resolve()) + '?mode=ro', uri=True); db.row_factory = sqlite3.Row
    if digest(json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])) != digest(producer):
        raise AdapterError('target_contract_mismatch', 'Current target producer differs from index')
    rows = []; unavailable = []
    for sid, expected in selection.items():
        feature = records.get(sid); target = db.execute('SELECT * FROM samples WHERE sample_id=?', (sid,)).fetchone()
        if feature is None or target is None or target['state'] != 'complete':
            unavailable.append({'sample_id': sid, **expected, 'reason': 'feature_unavailable' if feature is None else target['error_code'] if target else 'target_not_generated'})
            continue
        target = dict(target); pair_checks(feature, target, expected, native, producer)
        ix = RouteIndex.from_dataset_root(config['dataset_root'], expected['route']); ref = ix.ref(expected['frame'])
        from tools.training.diffusiondrive_v1.features_cache import file_digest
        if file_digest(ix.path(ref, 'anno')) != target['current_anno_sha256']:
            raise AdapterError('pair_source_mismatch', sid)
        if digest(reference_pose(FrameReader(ix).read(ref), native)) != digest(json.loads(target['reference_json'])):
            raise AdapterError('pair_coordinate_mismatch', sid)
        rows.append({'sample_id': sid, 'dataset_id': config['dataset_id'], **expected,
                     'feature_root_id': feature['job_id'] + ':' + feature['row']['root_id'],
                     'feature_relative_path': feature['row']['relative_path'],
                     'target_root_id': config['target_root']['root_id'], 'target_relative_path': target['relative_path'],
                     'feature_producer_sha': feature['producer_sha256'], 'target_producer_sha': digest(producer),
                     'feature_config': feature['config'], 'feature_row': feature['row'], 'target_row': target})
    db.close()
    result = {'schema': SCHEMA, 'dataset_id': config['dataset_id'], 'selection_sha256': digest(config['data']),
              'split_policy': config.get('paired_split_policy', 'town'), 'target_producer': producer,
              'rows': rows, 'unavailable': unavailable}
    durable_json(output, result); return result


class PairedDataset:
    def __init__(self, rows, config, target_producer):
        self.rows, self.config = rows, config
        self.target_sha = digest(target_producer)
        self.tokens = [str(Path(config['dataset_root']) / r['route']) + '::' + str(r['frame']) for r in rows]
        self.feature_roots = {}
        from tools.training.diffusiondrive_v1.features_cache import feature_producer
        from tools.training.diffusiondrive_v1.feature_jobs import config_and_state
        self.feature_sha = digest(feature_producer(config))
        collection = json.loads((ROOT / config['features_collection']).read_text())
        owners = {entry['config']: entry['job_id'] for entry in collection['jobs']}
        target_db = sqlite3.connect('file:' + str((ROOT / config['target_state_dir'] / 'index.sqlite').resolve()) + '?mode=ro', uri=True)
        target_db.row_factory = sqlite3.Row
        feature_dbs = {}
        for row in rows:
            if row['feature_config'] not in owners:
                raise AdapterError('pair_location_mismatch', 'Feature config outside expected collection')
            cfg, _ = config_and_state(ROOT / row['feature_config'])
            if cfg['dataset_id'] != config['dataset_id'] or row['feature_root_id'] != owners[row['feature_config']] + ':' + row['feature_row']['root_id']:
                raise AdapterError('pair_location_mismatch', row['sample_id'])
            self.feature_roots[row['feature_config']] = cfg['cache_roots']
            if row['feature_producer_sha'] != self.feature_sha or row['target_producer_sha'] != self.target_sha:
                raise AdapterError('pair_producer_mismatch', row['sample_id'])
            if row['target_root_id'] != config['target_root']['root_id']:
                raise AdapterError('pair_location_mismatch', row['sample_id'])
            if row['feature_config'] not in feature_dbs:
                db = sqlite3.connect('file:' + str((ROOT / cfg['state_dir'] / 'index.sqlite').resolve()) + '?mode=ro', uri=True)
                db.row_factory = sqlite3.Row
                if digest(json.loads(db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()[0])) != self.feature_sha:
                    raise AdapterError('pair_producer_mismatch', row['feature_config'])
                feature_dbs[row['feature_config']] = db
            f_actual = feature_dbs[row['feature_config']].execute('SELECT * FROM samples WHERE sample_id=?', (row['sample_id'],)).fetchone()
            t_actual = target_db.execute('SELECT * FROM samples WHERE sample_id=?', (row['sample_id'],)).fetchone()
            for actual, cached, keys in [(f_actual, row['feature_row'], ['sample_id','route','frame','root_id','relative_path','sha256','bytes','source_json']),
                                         (t_actual, row['target_row'], ['sample_id','dataset_id','route','frame','current_anno_sha256','reference_json','sources_json','map_json','relative_path','sha256','bytes'])]:
                if actual is None or actual['state'] != 'complete' or any(actual[k] != cached[k] for k in keys):
                    raise AdapterError('pair_index_mismatch', row['sample_id'])
        target_db.close()
        for db in feature_dbs.values(): db.close()

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        from tools.training.diffusiondrive_v1.target_relocation import TargetLocationResolver
        if not hasattr(self, '_target_locations'):
            self._target_locations = TargetLocationResolver({}, self.config)
        target_spec = self._target_locations.resolve(
            self.config['target_root'], self.target_sha, ROOT / self.config['target_state_dir'])
        features = load_feature_row(row['feature_row'], self.feature_roots[row['feature_config']], self.feature_sha)
        payload = load_target_row(row['target_row'], target_spec, self.target_sha)
        return features, payload['targets']


def paired_datasets(path, config, model_config):
    from tools.training.diffusiondrive_v1.runtime import native_configuration
    from tools.training.diffusiondrive_v1.features_cache import file_digest
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') == 'b2d-v1-explicit-recovery-paired-v1':
        from tools.training.diffusiondrive_v1.recovery_paired import datasets
        return datasets(path, config, model_config)
    expected = splits(config); native = native_configuration(config)
    producer = producer_identity(config, model_config, native)
    if (manifest['schema'] != SCHEMA or manifest['dataset_id'] != config['dataset_id']
        or manifest['selection_sha256'] != digest(config['data']) or digest(manifest['target_producer']) != digest(producer)
        or manifest['split_policy'] != config.get('paired_split_policy', 'town')):
        raise AdapterError('paired_contract_mismatch', 'Manifest/version/data split differs')
    groups = {split: [] for split in config['data']}; seen = set()
    from tools.training.diffusiondrive_v1.features_cache import feature_producer
    fp = feature_producer(config)
    if digest(fp['native_policy']['parameters']['geometry']) != digest(producer['geometry']['parameters']):
        raise AdapterError('pair_coordinate_mismatch', 'Feature vs Target rear/clock contract differs')
    for row in manifest['rows']:
        sid = row['sample_id']
        if sid not in expected or sid in seen or row['split'] != expected[sid]['split'] or row['dataset_id'] != config['dataset_id']:
            raise AdapterError('pair_split_mismatch', sid)
        seen.add(sid)
        f = {'row': row['feature_row'], 'dataset_id': config['dataset_id']}
        pair_checks(f, row['target_row'], expected[sid], native, producer)
        if (row['feature_relative_path'] != row['feature_row']['relative_path']
            or row['target_relative_path'] != row['target_row']['relative_path']
            or row['route'] != expected[sid]['route'] or row['frame'] != expected[sid]['frame']
            or row['feature_root_id'].split(':', 1)[-1] != row['feature_row']['root_id']):
            raise AdapterError('pair_location_mismatch', sid)
        groups[row['split']].append(row)
    result = {split: PairedDataset(rows, config, producer) for split, rows in groups.items()}
    if any(not len(result[s]) for s in ('train', 'validation')): raise AdapterError('empty_paired_split', str(path))
    identity = {'path': str(Path(path).resolve()), 'sha256': file_digest(path), 'counts': {s: len(d) for s, d in result.items()},
                'feature_producer_sha256': digest(fp), 'target_producer_sha256': digest(producer), 'unavailable': manifest['unavailable']}
    return result, identity


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); result = build_manifest(cfg, args.output)
    print(json.dumps({'paired': len(result['rows']), 'unavailable': result['unavailable']}))


if __name__ == '__main__': main()
