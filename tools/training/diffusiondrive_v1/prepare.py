"""Pre-generate complete V1 tensors for an explicit route/frame list, with failures.

Existing content-addressed artifacts resume safely. All rejects are written to a
JSONL journal; only the separate accepted manifest can be consumed by Trainer.
No recursive dataset scan, fake targets or training-time failure skipping.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'repositories/DiffusionDrive'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2e_v1_mpl')


def main():
    import torch
    from tools.b2d_adapter import FrameReader, RouteIndex, AdapterError
    from tools.b2d_adapter.v1_builders import DiffusionDriveV1Dataset, v1_native_config
    from tools.training.diffusiondrive_v1.runtime import fit_configuration, model_configuration, bev_provider, route_group, native_configuration
    from tools.training.diffusiondrive_v1.prepared import (
        SCHEMA, digest, file_digest, producer_identity, sample_sources, validate_tensors, load_artifact, atomic_json)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config, _ = fit_configuration(args.config)
    torch.set_num_threads(config.get('threads', 4))
    native = native_configuration(config)
    model = model_configuration(config)
    producer = producer_identity(model, config.get("label_rules"), native.camera_policy)
    provider = bev_provider(config)
    output = args.output_dir.resolve()
    if native.camera_policy == 'b2d-proportional-crop-v1':
        output = output / native.camera_policy
    if provider.rules.enriched:
        output = output / provider.rules.name / provider.rules.version
    existing_manifest = output / 'accepted_manifest.json'
    if existing_manifest.exists() and digest(json.loads(existing_manifest.read_text())['producer']) != digest(producer):
        raise AdapterError('prepared_output_collision', 'use a new output directory for a new producer')
    (output / 'tensors').mkdir(parents=True, exist_ok=True)
    accepted = {'train': [], 'validation': []}
    failed = []
    counts = Counter()
    town_stats = {}
    started = time.monotonic()
    journal = output / 'attempts.jsonl'
    with journal.open('a', buffering=1) as stream:
        for split in ('train', 'validation'):
            for entry in config['data'][split]:
                route = (ROOT / entry['route']).resolve()
                reader = None
                for frame in entry['frames']:
                    token = str(route) + '::' + str(frame)
                    phase = 'source_identity'
                    record = {'split': split, 'token': token, 'route_group': route_group(route)}
                    sample_started = time.monotonic()
                    try:
                        if reader is None:
                            reader = FrameReader(RouteIndex(route))
                        ref = reader.index.ref(frame)
                        town = re.search(r'_(Town\d+(?:HD)?)_', route.name).group(1)
                        map_identity = provider.map_identity(town)
                        sources = sample_sources(reader.index, ref, native)
                        identity = {'schema': SCHEMA, 'token': token, 'sources': sources,
                                    'producer_sha256': digest(producer), 'map': map_identity}
                        key = digest(identity)
                        tensor_path = output / 'tensors' / (key + '.pt')
                        metadata_path = tensor_path.with_suffix('.json')
                        phase = 'cached_tensor_read'
                        if tensor_path.exists() or metadata_path.exists():
                            row = json.loads(metadata_path.read_text())
                            # JSON turns tuples into lists; compare canonical identities.
                            if digest(row['identity']) != key or row['cache_key'] != key:
                                raise AdapterError('invalid_prepared_cache', str(metadata_path))
                            features, targets = load_artifact(row)
                            cache = 'hit'
                        else:
                            phase = 'full_adapter_build'
                            dataset = DiffusionDriveV1Dataset(reader, [ref], native, model,
                                                            config['lidar_decoder_python'], provider)
                            sample = dataset.build_sample(0)
                            features, targets = sample.features, sample.targets
                            validate_tensors(features, targets)
                            phase = 'source_stability'
                            if sources != sample_sources(reader.index, ref, native):
                                raise AdapterError('source_changed_during_build', token)
                            phase = 'atomic_tensor_write'
                            fd, temporary = tempfile.mkstemp(dir=tensor_path.parent, suffix='.pt')
                            try:
                                with os.fdopen(fd, 'wb') as tensor_stream:
                                    torch.save({'features': features, 'targets': targets}, tensor_stream)
                                row = dict(record, cache_key=key, identity=identity, tensor_path=str(tensor_path),
                                           tensor_sha256=file_digest(temporary), provenance=sample.metadata)
                                os.replace(temporary, tensor_path)
                                atomic_json(metadata_path, row)
                            finally:
                                if os.path.exists(temporary):
                                    os.unlink(temporary)
                            cache = 'generated'
                        counts[cache] += 1
                        accepted[split].append(row)
                        pixels = torch.bincount(targets['bev_semantic_map'].flatten(), minlength=7).tolist()
                        stat = town_stats.setdefault(town, {'accepted': 0, 'class_pixels': [0] * 7,
                                                           'class_present_frames': [0] * 7, 'map': map_identity})
                        stat['accepted'] += 1
                        for i, pixels_i in enumerate(pixels):
                            stat['class_pixels'][i] += pixels_i
                            stat['class_present_frames'][i] += int(pixels_i > 0)
                        record.update(status='accepted', cache=cache, cache_key=key, class_pixels=pixels,
                                      ego_map_alignment=row['provenance']['targets']['bev']['coverage'])
                    except Exception as exc:
                        record.update(status='rejected', phase=phase,
                                      error_code=getattr(exc, 'code', type(exc).__name__), error=str(exc))
                        failed.append(dict(record))
                    record['seconds'] = time.monotonic() - sample_started
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(json.dumps({k: record[k] for k in ('token', 'status', 'seconds')}, ensure_ascii=False), flush=True)
    manifest = {'schema': SCHEMA, 'producer': producer, 'requested_data_sha256': digest(config['data']),
                'map_paths_sha256': digest(config['map_paths']), 'accepted': accepted, 'failed_count': len(failed),
                'rejected_tokens': [row['token'] for row in failed]}
    # A repeat with unchanged sources produces byte-identical manifests, keeping resume valid.
    atomic_json(output / 'accepted_manifest.json', manifest)
    report = {'status': 'ready' if all(accepted.values()) else 'blocked_empty_split',
              'requested': sum(len(entry['frames']) for split in config['data'].values() for entry in split),
              'accepted': {key: len(rows) for key, rows in accepted.items()}, 'failed_count': len(failed),
              'failure_codes': dict(Counter(row['error_code'] for row in failed)), 'failures': failed,
              'tensor_cache': dict(counts), 'bev_cache': provider.cache_stats, 'towns': town_stats,
              'seconds': time.monotonic() - started, 'manifest_sha256': file_digest(output / 'accepted_manifest.json')}
    atomic_json(output / 'preparation_result.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'accepted', 'failed_count', 'failure_codes', 'tensor_cache')}, ensure_ascii=False))
    if report['status'] != 'ready':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
