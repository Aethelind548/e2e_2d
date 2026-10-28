"""Inspect a release profile without importing models or reading data payloads."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def small_text_sha256(path):
    """Hash only bounded source/configuration files, never large assets."""
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('source/configuration exceeds 8 MiB inspection limit')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_profile(root, manifest, profile, locations=None):
    """Return static checks; success never certifies training or Agent readiness."""
    root = Path(root).resolve()
    if manifest.get('schema') != 'diffusiondrive-v1-release-resources-v1':
        raise ValueError('unknown release resource manifest schema')
    spec = manifest['profiles'][profile]
    locations = dict(locations or {}, workspace=str(root))
    checks = []

    def record(name, status, detail):
        checks.append({'name': name, 'status': status, 'detail': detail})

    def source_path(relative):
        path = root / relative
        # Source/config paths are release-relative; external resources use the
        # explicit location templates below and are only inspected with stat.
        if Path(relative).is_absolute() or root not in path.resolve().parents:
            raise ValueError('source/configuration path escapes release root')
        return path

    for relative in spec.get('entry_files', []):
        path = source_path(relative)
        record(relative, 'PASS' if path.is_file() else 'BLOCKED', 'entry source exists' if path.is_file() else 'entry source missing')

    for relative, expected in spec.get('source_sha256', {}).items():
        path = source_path(relative)
        if not path.is_file():
            record(relative, 'BLOCKED', 'pinned source missing')
        else:
            actual = small_text_sha256(path)
            record(relative, 'PASS' if actual == expected else 'BLOCKED',
                   {'expected_sha256': expected, 'actual_sha256': actual})

    for relative in spec.get('config_files', []):
        path = source_path(relative)
        if not path.is_file():
            record(relative, 'BLOCKED', 'configuration missing')
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('configuration exceeds inspection limit')
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError('configuration must be a JSON object')
        record(relative, 'PASS', 'JSON object loaded; no dataset instantiated')
        for pinned, expected in config.get('execution_source_sha256', {}).items():
            source = source_path(pinned)
            actual = small_text_sha256(source) if source.is_file() else None
            if actual == expected:
                status = 'PASS'
            elif spec.get('fit_execution_contract', False):
                status = 'BLOCKED'
            else:
                # R2 does not execute fit.py's pin gate. Still expose inherited
                # mismatches; this tool cannot infer its checkpoint contract.
                status = 'UNVERIFIED'
            record(relative + ':execution_pin:' + pinned, status,
                   {'expected_sha256': expected, 'actual_sha256': actual,
                    'scope': 'fit execution contract' if spec.get('fit_execution_contract', False) else 'inherited fit pin; verify actual R2 checkpoint separately'})

    for resource in manifest.get('resources', []):
        if profile not in resource['profiles']:
            continue
        try:
            path = Path(resource['path'].format_map(locations))
        except KeyError as exc:
            record(resource['id'], 'BLOCKED', 'explicit location required: ' + str(exc))
            continue
        exists = path.is_dir() if resource['kind'] == 'directory' else path.is_file()
        record(resource['id'], 'PASS' if exists else 'BLOCKED',
               {'path': str(path), 'metadata_present': exists,
                'size_bytes': path.stat().st_size if exists and path.is_file() else None,
                'expected_sha256': resource.get('expected_sha256'),
                'payload_identity': 'NOT_VERIFIED; metadata only'})

    # Report portability blockers without rewriting any original configuration.
    for limitation in spec.get('unverified', []):
        record('acceptance', 'UNVERIFIED', limitation)
    blocked = any(item['status'] == 'BLOCKED' for item in checks)
    unverified = any(item['status'] == 'UNVERIFIED' for item in checks)
    return {'schema': 'diffusiondrive-v1-release-preflight-v1', 'profile': profile,
            'status': 'BLOCKED' if blocked else 'UNVERIFIED' if unverified else 'STATIC_CHECKS_PASS',
            'training_or_agent_acceptance': 'NOT_VERIFIED', 'checks': checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'tools/release/resources.json')
    parser.add_argument('--profile', required=True, choices=('baseline', 'r2', 'agent'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--experiment-root', type=Path)
    parser.add_argument('--map-root', type=Path)
    parser.add_argument('--location', action='append', default=[], metavar='KEY=PATH',
                        help='explicit payload-root location from the resource manifest; never remaps runtime configs')
    args = parser.parse_args(argv)
    locations = {name: str(value.resolve()) for name, value in
                 (('data_root', args.data_root), ('experiment_root', args.experiment_root),
                  ('map_root', args.map_root)) if value is not None}
    for value in args.location:
        key, separator, path = value.partition('=')
        if not separator or not key or not path or key == 'workspace' or key in locations:
            parser.error('--location requires a unique KEY=PATH; workspace cannot be overridden')
        locations[key] = str(Path(path).resolve())
    try:
        manifest = json.loads(args.manifest.read_text())
        report = inspect_profile(args.root, manifest, args.profile, locations)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Release preflight rejected: ' + str(exc) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'STATIC_CHECKS_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
