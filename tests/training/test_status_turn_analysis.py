import json
import math

import pytest

from tools.training.diffusiondrive_v1 import status_turn_analysis as turns


def write_samples(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'samples.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def straight_target():
    return [[float(index), 0.0] for index in range(8)]


def quarter_turn_target():
    points = []
    for index in range(8):
        angle = math.pi / 2 * index / 7
        points.append([5.0 * math.sin(angle), 5.0 * (1 - math.cos(angle))])
    return points


def sample(sample_id, route, target, ade, fde, heading):
    return {'sample_id': sample_id, 'route': route, 'target': target,
            'ade_m': ade, 'fde_m': fde, 'heading_error_deg': heading,
            'prediction': target, 'command': 'Right'}


def test_cumulative_turn_matches_curvature_not_endpoint_angle():
    assert turns.cumulative_turn_deg(straight_target()) == 0.0
    quarter = turns.cumulative_turn_deg(quarter_turn_target())
    assert 60.0 < quarter < 120.0


def test_summarize_keeps_only_turning_frames():
    rows = [sample('a', 'route', quarter_turn_target(), 2.0, 4.0, 10.0),
            sample('b', 'route', straight_target(), 9.0, 9.0, 90.0)]
    for row in rows:
        row['turn_deg'] = turns.cumulative_turn_deg(row['target'])
    stats = turns.summarize(rows, min_turn=60.0)
    assert stats['frames'] == 1
    assert stats['ade_m'] == 2.0 and stats['fde_m'] == 4.0
    assert stats['heading_error_deg'] == 10.0


def test_load_samples_adds_turn_deg(tmp_path):
    write_samples(tmp_path / 'first', [sample('a', 'route', quarter_turn_target(), 1.0, 2.0, 3.0)])
    rows = turns.load_samples(tmp_path / 'first')
    assert rows[0]['turn_deg'] > 60.0


def test_ensure_same_frames_rejects_different_sample_ids(tmp_path):
    write_samples(tmp_path / 'first', [sample('a', 'route', quarter_turn_target(), 1.0, 2.0, 3.0)])
    write_samples(tmp_path / 'second', [sample('b', 'route', quarter_turn_target(), 1.0, 2.0, 3.0)])
    payload = {'first': turns.load_samples(tmp_path / 'first'),
               'second': turns.load_samples(tmp_path / 'second')}
    with pytest.raises(SystemExit) as error:
        turns.ensure_same_frames(payload)
    assert 'sample_id sets differ' in str(error.value)

    write_samples(tmp_path / 'third', [sample('a', 'route', quarter_turn_target(), 1.0, 2.0, 3.0)])
    matching = {'first': payload['first'], 'third': turns.load_samples(tmp_path / 'third')}
    turns.ensure_same_frames(matching)
