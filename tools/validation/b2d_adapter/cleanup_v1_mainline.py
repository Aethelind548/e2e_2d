"""One-time, explicit V1 mainline cleanup; plan/dry-run precede file deletion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'analysis/diffusiondrive_b2d_architecture/v1_validation/training_preparation'
# Enumerated after inspecting directory contents and V1 imports/configuration.
OBSOLETE_TREES = (
    'tools/bndp', 'tools/training/bndp', 'tools/validation/bndp', 'adapters',
    'outputs/bndp_pre3b', 'outputs/bndp_pre3b_sm120', 'outputs/bndp_3b',
    'outputs/bndp_3b3', 'outputs/bndp_full', 'outputs/bndp_full_v2',
    'analysis/bndp2_model_analysis', 'analysis/bndp3_curriculum',
    'analysis/bndp3_diagnosis', 'analysis/bndp3_original_il',
    'analysis/bndp3_overfit', 'analysis/bndp3_source_il',
    'analysis/bndp3b3_preparation', 'analysis/bndp3b3_results_review',
    'analysis/bndp3b_preparation', 'analysis/bndp3b_results_review',
    'analysis/bndp_full_preparation', 'analysis/bndp_full_v2_performance',
    'analysis/bndp_pre3b',
)
OBSOLETE_FILES = (
    'analysis/model_baseline.md',
    'analysis/data_interface_analysis/baseline_b_adapter_requirements.md',
    'analysis/data_interface_analysis/baseline_b_implementation_plan.md',
    'analysis/data_interface_analysis/baseline_b_training_design.md',
    'analysis/data_interface_analysis/training_strategy_options.md',
    'analysis/data_interface_analysis/active_model_interface.md',
    'analysis/data_interface_analysis/v2_supervised_reuse_analysis.md',
    'analysis/data_interface_analysis/diffusion_supervised_training_reference.md',
    'tools/validation/b2d_adapter/validate_level2.py',
    'tools/validation/b2d_adapter/validate_native_builders.py',
    'tools/validation/b2d_adapter/validate_engineering_builders.py',
    'tools/validation/b2d/backups/inspect_b2d_route.py.orig',
)
EXCEPTIONS = {
    'analysis/bndp3_original_il/sources/official_kmeans_navsim_traj_20.npy':
        'Official V1 anchor referenced by original configurations and resume identities.',
    'analysis/bndp3_original_il/sources/anchor_provenance.json': 'Official anchor URL, size and SHA.',
    'analysis/bndp3_original_il/sources/provenance.json': 'Official V1 upstream commit/file provenance.',
    'analysis/bndp3_original_il/sources/LICENSE': 'Official source license.',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def files_under(path):
    for parent, dirs, files in os.walk(path, followlinks=False):
        if any((Path(parent)/d).is_symlink() for d in dirs):
            raise RuntimeError('Symlink directory requires manual review: '+parent)
        for name in files:
            p = Path(parent)/name
            if not p.is_symlink() and not p.is_file():
                raise RuntimeError('Only regular files are eligible: '+str(p))
            yield p


def status(path):
    p = subprocess.run(['git', '-C', str(path), 'status', '--porcelain=v1'],
                       text=True, capture_output=True)
    head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                          text=True, capture_output=True)
    return {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr,
            'head': head.stdout.strip() if head.returncode == 0 else None}


def protected():
    roots = ('repositories/DiffusionDrive', 'repositories/DiffusionDriveV2',
             'tools/b2d_adapter', 'tools/training/diffusiondrive_v1',
             'outputs/diffusiondrive_v1')
    result = {}
    for base in roots:
        for p in files_under(ROOT/base):
            result[str(p.relative_to(ROOT))] = sha(p)
    for rel in EXCEPTIONS:
        result[rel] = sha(ROOT/rel)
    return result


def make_plan():
    candidates = {}
    for base in OBSOLETE_TREES:
        for p in files_under(ROOT/base):
            rel = str(p.relative_to(ROOT))
            if rel not in EXCEPTIONS:
                candidates[rel] = ('obsolete_research', 'Inspected BNDP model/Ranking/Rollout training or its generated artifacts; no V1 runtime dependency.')
    for rel in OBSOLETE_FILES:
        if (ROOT/rel).is_file():
            candidates[rel] = ('obsolete_entry_or_report', 'Superseded V2-only diagnostic, BNDP plan or obsolete duplicate; V1 uses independent entrypoints.')
    # Bytecode outside both original repositories is reproducible, not source.
    for rel in ('tools',):
        for p in files_under(ROOT/rel):
            if '__pycache__' in p.parts and p.suffix == '.pyc':
                candidates.setdefault(str(p.relative_to(ROOT)), ('generated_bytecode', 'Reproducible bytecode; all required .py sources retained.'))
    rows = []
    for rel, (category, reason) in sorted(candidates.items()):
        p = ROOT/rel
        link = os.readlink(p) if p.is_symlink() else None
        if link is not None:
            target = str(p.resolve().relative_to(ROOT))
            if not any(target.startswith(base+'/') for base in OBSOLETE_TREES):
                raise RuntimeError('Symlink target requires separate review: '+rel)
        if p.parent.resolve() != p.parent or (not p.is_file() and link is None) or rel.startswith('repositories/'):
            raise RuntimeError('Invalid deletion boundary: '+rel)
        st = p.lstat()
        rows.append({'path': rel, 'category': category, 'reason': reason,
                     'bytes': st.st_size, 'allocated_bytes': st.st_blocks*512,
                     'device': st.st_dev, 'inode': st.st_ino, 'nlink': st.st_nlink,
                     'mtime_ns': st.st_mtime_ns, 'symlink_target': link})
    before = {str(p.relative_to(ROOT)): status(p) for p in
              (ROOT, ROOT/'repositories/DiffusionDrive', ROOT/'repositories/DiffusionDriveV2')}
    save('git_before.json', before)
    hashes = protected()
    save('protected_before.json', hashes)
    plan = {'schema': 'v1-mainline-cleanup-v1', 'root': str(ROOT), 'files': rows,
            'preserved_exceptions': EXCEPTIONS, 'allowed_trees': OBSOLETE_TREES,
            'allowed_files': OBSOLETE_FILES, 'file_count': len(rows),
            'estimated_logical_bytes': sum(r['bytes'] for r in rows),
            'estimated_allocated_bytes': sum(r['allocated_bytes'] for r in rows),
            'external_deletions': [], 'original_data_deletions': []}
    save('deletion_plan.json', plan)
    save('classification.json', {
        'must_keep': ['both complete original model repositories, including Git histories',
                      'all V1 Adapter sources and complete-IL training modules',
                      'original Bench2Drive dataset and external maps/environments',
                      'all historical V1 tensors/checkpoints/manifests and M4/M5 code/evidence',
                      'official anchor/ImageNet initialization assets and provenance',
                      'shared coordinate/time/navigation/geometry validation evidence'],
        'delete': {'trees': OBSOLETE_TREES, 'individual_files': OBSOLETE_FILES,
                   'exceptions': EXCEPTIONS},
        'retained_pending_review': [
            {'path': 'repositories/NAVISIM_Agent_Info', 'reason': 'Empty auxiliary repository directory; origin/purpose not established.'},
            {'path': 'outputs/diffusiondrive_v1', 'reason': 'Historical/staging runs may be referenced by validation and resume; no duplicate V1 run removed without separate reference review.'},
            {'path': 'analysis/data_interface_analysis', 'reason': 'Remaining shared calibration/contract evidence supports inherited V1 transforms; selective obsolete plans only deleted.'}],
        'dependency_findings': ['No adapters.*, tools.bndp or tools.training.bndp imports in V1/official source.',
                                'Native shared builders contain a lazy historical V2 factory; V1 overrides it. Source retained to preserve producer hash and shared tests.',
                                'Official anchor under a mixed old directory is retained at its exact historical path.']})
    print(json.dumps({k: plan[k] for k in ('file_count', 'estimated_logical_bytes', 'estimated_allocated_bytes')}))


def apply_plan():
    plan = json.loads((REPORT/'deletion_plan.json').read_text())
    if plan['root'] != str(ROOT) or plan['allowed_trees'] != list(OBSOLETE_TREES):
        raise RuntimeError('Unexpected deletion plan identity')
    # Check all paths/stats before any mutation; reject source races.
    for row in plan['files']:
        rel = row['path']
        if rel in EXCEPTIONS or rel.startswith('repositories/'):
            raise RuntimeError('Protected path: '+rel)
        allowed = rel in OBSOLETE_FILES or any(rel.startswith(b+'/') for b in OBSOLETE_TREES)
        allowed |= rel.startswith('tools/') and '/__pycache__/' in rel and rel.endswith('.pyc')
        p = ROOT/rel
        if not allowed or p.parent.resolve() != p.parent:
            raise RuntimeError('Out of scope: '+rel)
        if (os.readlink(p) if p.is_symlink() else None) != row['symlink_target']:
            raise RuntimeError('Changed symlink since dry-run: '+rel)
        st = p.lstat()
        if (st.st_size, st.st_dev, st.st_ino, st.st_mtime_ns) != tuple(row[k] for k in ('bytes','device','inode','mtime_ns')):
            raise RuntimeError('Changed since dry-run: '+rel)
    save('dry_run.json', {'status': 'pass', 'checked_files': len(plan['files']),
                          'paths_exist': True, 'only_regular_files_or_reviewed_internal_checkpoint_links': True,
                          'workspace_boundary': str(ROOT), 'protected_paths_excluded': True})
    free_before = shutil.disk_usage(ROOT).free
    start = time.monotonic()
    with (REPORT/'deleted_files.jsonl').open('x') as log:
        for row in plan['files']:
            (ROOT/row['path']).unlink()
            log.write(json.dumps({'path': row['path'], 'bytes': row['bytes'],
                                  'allocated_bytes': row['allocated_bytes'], 'status': 'deleted'})+'\n')
    # Remove empty directories only, preserving mixed-directory exceptions.
    dirs = set()
    for row in plan['files']:
        p = (ROOT/row['path']).parent
        while p != ROOT:
            dirs.add(p)
            p = p.parent
    removed_dirs = []
    for p in sorted(dirs, key=lambda x: len(x.parts), reverse=True):
        try:
            p.rmdir()
            removed_dirs.append(str(p.relative_to(ROOT)))
        except OSError:
            pass
    after = protected()
    before = json.loads((REPORT/'protected_before.json').read_text())
    expected_bytecode = {r['path'] for r in plan['files'] if r['category'] == 'generated_bytecode'}
    differences = [p for p, h in before.items() if after.get(p) != h and p not in expected_bytecode]
    save('protected_after.json', after)
    save('git_after_cleanup.json', {str(p.relative_to(ROOT)): status(p) for p in
                                  (ROOT, ROOT/'repositories/DiffusionDrive', ROOT/'repositories/DiffusionDriveV2')})
    result = {'status': 'pass' if not differences else 'failed_integrity',
              'deleted_files': len(plan['files']), 'removed_directories': removed_dirs,
              'removed_logical_bytes': plan['estimated_logical_bytes'],
              'removed_allocated_bytes': plan['estimated_allocated_bytes'],
              'filesystem_free_delta_bytes': shutil.disk_usage(ROOT).free-free_before,
              'protected_file_count': len(before), 'protected_differences': differences,
              'expected_removed_protected_bytecode': sorted(set(before) & expected_bytecode),
              'seconds': time.monotonic()-start}
    save('cleanup_result.json', result)
    print(json.dumps(result))
    if differences:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    apply_plan() if args.apply else make_plan()
