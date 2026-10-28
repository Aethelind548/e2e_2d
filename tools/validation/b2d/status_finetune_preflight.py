"""Read-only preflight for a proposed Status-only warm-start experiment.

This does not create a training entry, load a model for inference, start an
optimizer, or touch the running Route Preview generation.  It verifies that the
draft config, frozen base config, checkpoint, panel, manifest and overlay have
the identities the experiment plan claims.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

DEFAULT_DRAFT = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_status_prep/status_finetune_draft.json'
DEFAULT_OVERLAY_PREFLIGHT = ROOT / 'outputs/diffusiondrive_v1/route_preview_v11/20261022_status_preflight/status_overlay_preflight.json'


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_identity(checks, name, path, expected):
    path = Path(path)
    actual = sha256(path) if path.is_file() else None
    checks[name] = {'status': 'PASS' if actual == expected else 'FAIL',
                    'path': str(path), 'expected_sha256': expected,
                    'actual_sha256': actual}


V2_SCHEMA = 'status-finetune-draft-v2'
V2_IDENTITY_FIELDS = ('base_config', 'training_entry', 'paired_manifest', 'pair_index',
                       'status_overlay', 'status_overlay_summary', 'reader_source',
                       'contract_source', 'start_checkpoint')


def first_candidate(path):
    """Return the first overlay row that carries a Status candidate, without loading the file."""
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('status8') is not None:
                return row
    raise ValueError('overlay_has_no_candidate_row')


def preflight_v2(args, draft):
    """Full-overlay warm-start preflight: identities, summary, admission probe, isolation."""
    import sqlite3
    from tools.training.diffusiondrive_v1 import status_overlay_contract as contract_module
    from tools.training.diffusiondrive_v1 import status_overlay_reader as reader_module
    from tools.training.diffusiondrive_v1.features_cache import FEATURE_SHAPES
    from tools.training.diffusiondrive_v1.status_overlay_contract import validate_row

    checks = {}
    for name in V2_IDENTITY_FIELDS:
        check_identity(checks, name, draft[name], draft[name + '_sha256'])

    shape = FEATURE_SHAPES.get('status_feature')
    checks['status_feature_shape_8'] = {'status': 'PASS' if shape == (8,) else 'FAIL',
                                        'observed': list(shape or ())}
    checks['status_contract_order'] = {
        'status': 'PASS' if draft['status_contract']['order'] == list(contract_module.STATUS_ORDER) else 'FAIL',
        'declared': draft['status_contract']['order'],
        'contract': list(contract_module.STATUS_ORDER)}

    summary = json.loads(Path(draft['status_overlay_summary']).read_text())
    failed = sum(item.get('failed', 0) for item in summary.get('checks', {}).values())
    overlay_ok = (summary.get('overlay_sha256') == draft['status_overlay_sha256']
                  and summary.get('manifest_sha256') == draft['paired_manifest_sha256']
                  and summary.get('pair_index_sha256') == draft['pair_index_sha256']
                  and summary.get('counts', {}).get('rows') == draft['overlay_rows']
                  and summary.get('counts', {}).get('training_admitted') == 0
                  and summary.get('counts', {}).get('status_validity_NOT_VERIFIED') == draft['expected_candidates']
                  and summary.get('counts', {}).get('status_validity_INVALID') == draft['expected_invalid']
                  and summary.get('mapping_rate') == 1.0
                  and failed == 0)
    checks['overlay_summary'] = {
        'status': 'PASS' if overlay_ok else 'FAIL',
        'rows': summary.get('counts', {}).get('rows'),
        'expected_rows': draft['overlay_rows'],
        'candidates': summary.get('counts', {}).get('status_validity_NOT_VERIFIED'),
        'invalid': summary.get('counts', {}).get('status_validity_INVALID'),
        'training_admitted': summary.get('counts', {}).get('training_admitted'),
        'check_failures': failed,
        'mapping_rate': summary.get('mapping_rate'),
        'summary_admission': summary.get('training_admission')}

    checks['reader_contract_version'] = {
        'status': 'PASS' if (draft['reader_version'] == reader_module.VERSION
                             and draft['reader_contract_schema'] == contract_module.READER_SCHEMA
                             and draft['overlay_schema'] == contract_module.SCHEMA) else 'FAIL',
        'reader_version': reader_module.VERSION,
        'reader_contract_schema': contract_module.READER_SCHEMA,
        'overlay_schema': contract_module.SCHEMA}

    pair_index = Path(draft['pair_index'])
    db = sqlite3.connect(pair_index.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        overlay_row = first_candidate(draft['status_overlay'])
        pair = db.execute('SELECT * FROM pairs WHERE sample_id=?', (overlay_row['sample_id'],)).fetchone()
    finally:
        db.close()
    pair = dict(pair) if pair is not None else None
    context = {'feature_producer_sha256': draft['feature_producer_sha256'],
               'target_producer_sha256': draft['target_producer_sha256'],
               'manifest_sha256': draft['paired_manifest_sha256'],
               'pair_index_sha256': draft['pair_index_sha256'],
               'allowed_overlay_producer_versions': draft['allowed_overlay_producer_versions'],
               'allowed_navigation_evidence_sha256': draft['allowed_navigation_evidence_sha256']}
    errors = validate_row(overlay_row, pair, context, require_admitted=True) if pair else ['pair_missing']
    checks['admission_probe'] = {
        'status': 'PASS' if errors else 'FAIL',
        'sample_id': overlay_row.get('sample_id'),
        'requirement': 'first candidate row must be rejected for training admission',
        'rejection_codes': errors,
        'row_status_validity': overlay_row.get('status_validity'),
        'row_certification': (overlay_row.get('navigation_source') or {}).get('certification'),
        'row_training_admitted': overlay_row.get('training_admitted')}

    target = Path(draft['output_dir'])
    paused = Path(draft['paused_run']['checkpoint_dir']).parent
    checks['fresh_output_dir'] = {'status': 'PASS' if not target.exists() else 'FAIL',
                                  'path': str(target), 'exists': target.exists()}
    checks['isolated_from_paused_run'] = {
        'status': 'PASS' if (target.resolve() != paused.resolve()
                             and paused.resolve() not in target.resolve().parents
                             and draft['start_checkpoint_sha256'] != draft['paused_run']['last_checkpoint_sha256'])
        else 'FAIL',
        'output_dir': str(target), 'paused_run': str(paused),
        'start_checkpoint_sha256': draft['start_checkpoint_sha256'],
        'paused_last_checkpoint_sha256': draft['paused_run']['last_checkpoint_sha256']}

    entry_text = Path(draft['training_entry']).read_text(errors='replace')
    checks['training_entry_has_no_overlay_import'] = {
        'status': 'PASS' if ('status_overlay_reader' not in entry_text
                             and 'StatusOverlayDataset' not in entry_text) else 'FAIL',
        'policy': 'OVERLAY_REQUIRES_SEPARATE_VERSIONED_ENTRY'}

    if args.load_checkpoint:
        try:
            import torch
            checkpoint = torch.load(draft['start_checkpoint'], map_location='cpu', weights_only=False)
            step = checkpoint.get('global_step')
            state = checkpoint.get('state_dict')
            ok = step == draft['start_global_step'] and isinstance(state, dict) and len(state) > 0
            checks['checkpoint_metadata'] = {'status': 'PASS' if ok else 'FAIL',
                                             'global_step': step,
                                             'expected_global_step': draft['start_global_step'],
                                             'state_dict_entries': len(state) if isinstance(state, dict) else None,
                                             'has_optimizer_states': 'optimizer_states' in checkpoint,
                                             'device': 'cpu', 'model_loaded': False}
            checkpoint_status = checks['checkpoint_metadata']['status']
        except Exception as exc:  # noqa: BLE001 - report, never retry
            checks['checkpoint_metadata'] = {'status': 'FAIL',
                                             'reason': type(exc).__name__ + ': ' + str(exc),
                                             'device': 'cpu', 'model_loaded': False}
            checkpoint_status = 'FAIL'
    else:
        checks['checkpoint_metadata'] = {'status': 'NOT_VERIFIED',
                                         'reason': 'Hash-only preflight; rerun with --load-checkpoint.',
                                         'device': 'cpu', 'model_loaded': False}
        checkpoint_status = 'NOT_VERIFIED'

    blocking = []
    if errors:
        blocking.append('overlay rows are not admissible for training')
    if summary.get('training_admission') == 'NO-GO':
        blocking.append('overlay summary declares NO-GO')
    if draft['admission']['external_blockers']:
        blocking.extend(draft['admission']['external_blockers'])
    checks['admission'] = {'status': 'NO_GO' if blocking else 'NOT_PROVEN',
                           'reasons': blocking or ['No exclusion has been proven sufficient.']}
    result = {
        'schema': 'status-finetune-preflight-v2',
        'draft': str(args.draft.resolve()),
        'draft_sha256': sha256(args.draft),
        'checks': checks,
        'all_identity_checks_pass': all(item.get('status') == 'PASS' for name, item in checks.items()
                                        if name not in ('checkpoint_metadata', 'admission')),
        'checkpoint_load_status': checkpoint_status,
        'training_started': False,
        'gpu_used': False,
        'overlay_reader_available': True,
        'admission': checks['admission']['status'],
        'source_boundary': 'Dry-run preflight only; no training, GPU, cache, model or generation writes.',
    }
    (args.output / 'status_finetune_preflight.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'all_identity_checks_pass': result['all_identity_checks_pass'],
                      'admission': result['admission'],
                      'checkpoint_load_status': result['checkpoint_load_status'],
                      'admission_probe_codes': errors,
                      'output': str(args.output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, default=DEFAULT_DRAFT)
    parser.add_argument('--overlay-preflight', type=Path, default=DEFAULT_OVERLAY_PREFLIGHT)
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='CPU-only checkpoint metadata/state_dict load check; no forward or optimizer')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh explicit output directory')
    draft_bytes = args.draft.read_bytes()
    if json.loads(draft_bytes).get('schema') == V2_SCHEMA:
        args.output.mkdir(parents=True)
        preflight_v2(args, json.loads(draft_bytes))
        return
    args.output.mkdir(parents=True)
    draft = json.loads(args.draft.read_text())
    checks = {}
    check_identity(checks, 'base_config', draft['base_config'], draft['base_config_sha256'])
    check_identity(checks, 'training_entry', draft['training_entry'], draft['training_entry_sha256'])
    check_identity(checks, 'paired_manifest', draft['paired_manifest'], draft['paired_manifest_sha256'])
    check_identity(checks, 'panel', draft['panel'], draft['panel_sha256'])
    check_identity(checks, 'start_checkpoint', draft['start_checkpoint'],
                   draft['start_checkpoint_sha256'])
    check_identity(checks, 'status_overlay', draft['status_overlay'], draft['status_overlay_sha256'])

    from tools.training.diffusiondrive_v1.features_cache import FEATURE_SHAPES
    shape_ok = FEATURE_SHAPES.get('status_feature') == (8,)
    checks['status_feature_shape_8'] = {'status': 'PASS' if shape_ok else 'FAIL',
                                        'observed': list(FEATURE_SHAPES.get('status_feature', ()))}

    preflight = json.loads(args.overlay_preflight.read_text()) if args.overlay_preflight.is_file() else None
    overlay_ok = (preflight is not None
                  and preflight.get('overlay_sha256') == draft['status_overlay_sha256']
                  and preflight.get('failure_count') == 0
                  and preflight.get('valid_count') == 0
                  and preflight.get('training_admitted_count') == 0
                  and preflight.get('reader_accepts_overlay') is False)
    checks['overlay_preflight'] = {
        'status': 'PASS' if overlay_ok else 'FAIL',
        'path': str(args.overlay_preflight),
        'overlay_rows': None if preflight is None else preflight.get('overlay_rows'),
        'failure_count': None if preflight is None else preflight.get('failure_count'),
        'valid_count': None if preflight is None else preflight.get('valid_count'),
        'training_admitted_count': None if preflight is None else preflight.get('training_admitted_count'),
        'reader_accepts_overlay': None if preflight is None else preflight.get('reader_accepts_overlay'),
    }

    reader_refs = 0
    for relative in ('tools/training/diffusiondrive_v1/paired_dataset.py',
                     'tools/training/diffusiondrive_v1/cache_only_reader.py'):
        reader_refs += Path(relative).read_text(errors='replace').lower().count('overlay')
    checks['reader_does_not_accept_overlay'] = {
        'status': 'PASS' if reader_refs == 0 else 'FAIL',
        'overlay_reference_count': reader_refs,
        'policy': 'NO_NATIVE_OVERLAY_READER',
    }
    target = Path(draft['output_dir'])
    try:
        target_exists = target.exists()
    except OSError as exc:
        target_exists = None
        target_error = str(exc)
    else:
        target_error = None
    checks['fresh_output_dir'] = {'status': 'PASS' if not target_exists else 'FAIL',
                                  'path': str(target), 'exists': target_exists,
                                  'error': target_error}
    if args.load_checkpoint:
        try:
            import torch
            checkpoint = torch.load(draft['start_checkpoint'], map_location='cpu', weights_only=False)
            step = checkpoint.get('global_step')
            state = checkpoint.get('state_dict')
            checks['checkpoint_metadata'] = {
                'status': 'PASS' if (step == draft['start_global_step'] and isinstance(state, dict)
                                     and len(state) > 0) else 'FAIL',
                'global_step': step, 'expected_global_step': draft['start_global_step'],
                'state_dict_entries': len(state) if isinstance(state, dict) else None,
                'device': 'cpu', 'model_loaded': False,
            }
            checkpoint_load_status = checks['checkpoint_metadata']['status']
        except Exception as exc:
            checks['checkpoint_metadata'] = {
                'status': 'FAIL', 'reason': type(exc).__name__ + ': ' + str(exc),
                'device': 'cpu', 'model_loaded': False,
            }
            checkpoint_load_status = 'FAIL'
    else:
        checks['checkpoint_metadata'] = {
            'status': 'NOT_VERIFIED',
            'reason': 'Hash-only preflight; rerun with --load-checkpoint after generation.',
        }
        checkpoint_load_status = 'NOT_VERIFIED'
    checks['admission'] = {
        'status': 'NO_GO' if (not overlay_ok or reader_refs == 0) else 'NOT_PROVEN',
        'reason': ('No native overlay reader and no admitted Status rows; '
                   'training cannot consume this overlay.') if (not overlay_ok or reader_refs == 0)
                  else 'Overlay reader existence is not training admission.',
    }
    result = {
        'schema': 'status-finetune-preflight-v1',
        'draft': str(args.draft.resolve()),
        'draft_sha256': sha256(args.draft),
        'checks': checks,
        'all_identity_checks_pass': all(item.get('status') == 'PASS' for name, item in checks.items()
                                        if name not in ('checkpoint_metadata', 'admission')),
        'checkpoint_load_status': checkpoint_load_status,
        'training_started': False,
        'gpu_used': False,
        'overlay_reader_supported': reader_refs > 0,
        'admission': checks['admission']['status'],
        'source_boundary': 'Dry-run preflight only; no training, GPU, cache, model or generation writes.',
    }
    (args.output / 'status_finetune_preflight.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'all_identity_checks_pass': result['all_identity_checks_pass'],
                      'admission': result['admission'],
                      'checkpoint_load_status': result['checkpoint_load_status'],
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
