"""Validate the certified Overlay v4 candidate with the independent reader rules.

This does not admit anything.  It feeds every v4 row and its frozen pair record
through the production ``status_overlay_contract.validate_row`` with
``require_admitted=True`` and reports the exact failure codes.  A healthy result
is "only approval-gate codes": the row structure, cache identity, Status layout
and A0 Motion4 identity must all pass, while admission stays blocked because the
contract is unapproved and the producer/evidence allowlists are empty.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from tools.training.diffusiondrive_v1 import status_overlay_contract as contract_rules

VERSION = 'certified-overlay-reader-check-v1'
GATE_CODES = {'training_not_admitted', 'overlay_producer_not_approved',
              'navigation_evidence_not_approved'}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--pair-index', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    context = {
        'feature_producer_sha256': manifest['feature_producer_sha256'],
        'target_producer_sha256': manifest['target_sources']['baseline']['producer_sha256'],
        'manifest_sha256': file_sha(args.manifest),
        'pair_index_sha256': file_sha(args.pair_index),
        'allowed_overlay_producer_versions': [],
        'allowed_navigation_evidence_sha256': [],
    }
    connection = sqlite3.connect(f'file:{args.pair_index}?mode=ro', uri=True)
    try:
        pairs = {}
        for row in connection.execute(
                'select sample_id, route, frame, split, feature_sha, target_sha from pairs'):
            pairs[row[0]] = {'sample_id': row[0], 'route': row[1], 'frame': row[2],
                             'split': row[3], 'feature_sha': row[4], 'target_sha': row[5]}
    finally:
        connection.close()

    codes = Counter()
    structural = Counter()
    per_row = []
    total = 0
    for line in args.overlay.open():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        pair = pairs.get(row['sample_id'])
        if pair is None:
            codes['pair_missing'] += 1
            structural['pair_missing'] += 1
            continue
        errors = contract_rules.validate_row(row, pair, context, require_admitted=True)
        for error in errors:
            codes[error] += 1
            if error not in GATE_CODES:
                structural[error] += 1
        per_row.append({'sample_id': row['sample_id'], 'errors': errors})

    summary = {
        'version': VERSION,
        'rows': total,
        'code_counts': dict(codes),
        'structural_failures': dict(structural),
        'only_gate_codes': bool(total and not structural),
        'observation': (
            'rows satisfy schema, cache identity, layout and A0 identity; admission is '
            'blocked only by the unapproved contract / empty allowlists'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (args.output.parent / 'reader_check_rows.json').write_text(
        json.dumps(per_row[:50], indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'rows', 'code_counts', 'structural_failures', 'only_gate_codes')}, sort_keys=True))


if __name__ == '__main__':
    main()
