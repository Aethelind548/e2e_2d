"""Run one real Bench2Drive leaderboard route and capture its global plan.

The capture itself happens inside the leaderboard process
(``session_plan_agent.py``).  This driver only:

* starts a fresh headless CARLA server for the route's town,
* invokes the unmodified ``leaderboard_evaluator.py`` for exactly one route,
* verifies that a capture landed on disk and records the run outcome,
* stops the server again.

One route per server is deliberate: CARLA 0.9.15 crashes on repeated
``load_world`` calls in the same process (observed ``Signal=11``).

Read-only with respect to the dataset: no dataset file, overlay, checkpoint or
leaderboard source is modified.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CARLA_ROOT = Path('/home/aethelind/carla/CARLA_0.9.15')
B2D_ROOT = Path('/home/aethelind/work/environment-build_test/Bench2Drive')
LEADERBOARD = B2D_ROOT / 'leaderboard'
SCENARIO_RUNNER = B2D_ROOT / 'scenario_runner'
EVALUATOR = LEADERBOARD / 'leaderboard' / 'leaderboard_evaluator.py'
AGENT = ROOT / 'tools' / 'validation' / 'b2d' / 'session_plan_agent.py'
CARLA_PYTHON = Path('/home/aethelind/carla/client_env/bin/python')
VERSION = 'leaderboard-session-capture-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def repo_head(repo):
    try:
        return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def wait_for_port(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(1.0)
    return False


def start_server(port, log_path, res_x, res_y):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open('w')
    command = [
        str(CARLA_ROOT / 'CarlaUE4.sh'), '-RenderOffScreen', '-opengl',
        '-quality-level=Low', '-nosound', f'-carla-rpc-port={port}',
        f'-ResX={res_x}', f'-ResY={res_y}',
    ]
    process = subprocess.Popen(command, cwd=str(CARLA_ROOT), stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process, stream


def stop_server(process, stream):
    try:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
    finally:
        stream.close()


def build_env(args, capture_dir):
    pythonpath = os.pathsep.join([
        str(LEADERBOARD), str(SCENARIO_RUNNER), str(CARLA_ROOT / 'PythonAPI' / 'carla'), str(ROOT),
    ])
    env = dict(os.environ)
    env['PYTHONPATH'] = pythonpath
    env['CARLA_ROOT'] = str(CARLA_ROOT)
    env['B2D_CAPTURE_PLAN_DIR'] = str(capture_dir)
    env['B2D_CAPTURE_DRIVE'] = '1' if args.drive else '0'
    env['B2D_CAPTURE_ID'] = args.capture_id
    env['B2D_DATASET_ROUTE'] = args.dataset_route
    env['B2D_ROUTE_ID'] = args.xml_id
    env['B2D_TOWN'] = args.town
    env['B2D_WEATHER'] = args.weather
    env['B2D_XML_FILE'] = str(Path(args.xml).name)
    env['B2D_XML_ID'] = args.xml_id
    env['B2D_XML_SHA256'] = sha256(args.xml)
    env['B2D_EVALUATOR_SHA'] = sha256(EVALUATOR)
    env['B2D_ROUTE_SCENARIO_SHA'] = sha256(LEADERBOARD / 'leaderboard' / 'scenarios' / 'route_scenario.py')
    env['B2D_ROUTE_MANIPULATION_SHA'] = sha256(LEADERBOARD / 'leaderboard' / 'utils' / 'route_manipulation.py')
    env['B2D_REPO_HEAD'] = repo_head(B2D_ROOT) or ''
    env.setdefault('PYTHONUNBUFFERED', '1')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--town', required=True)
    parser.add_argument('--xml', required=True, type=Path)
    parser.add_argument('--xml-id', required=True)
    parser.add_argument('--dataset-route', required=True)
    parser.add_argument('--capture-id', required=True)
    parser.add_argument('--weather', default='')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=90.0)
    parser.add_argument('--server-timeout', type=float, default=180.0)
    parser.add_argument('--drive', action='store_true')
    parser.add_argument('--res-x', type=int, default=640)
    parser.add_argument('--res-y', type=int, default=480)
    parser.add_argument('--reuse-server', action='store_true',
                        help='assume a server is already running on --port')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.xml = args.xml.resolve()

    capture_dir = args.output / 'session_captures'
    runs_dir = args.output / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_path = capture_dir / f'{args.capture_id}.json'
    if capture_path.exists():
        print(json.dumps({'capture_id': args.capture_id, 'status': 'SKIPPED_EXISTING'}))
        return 0

    env = build_env(args, capture_dir)
    server_log = runs_dir / f'{args.capture_id}.server.log'
    evaluator_log = runs_dir / f'{args.capture_id}.evaluator.log'
    checkpoint = runs_dir / f'{args.capture_id}.checkpoint.json'

    server = stream = None
    started = time.time()
    result = {'capture_id': args.capture_id, 'dataset_route': args.dataset_route,
              'town': args.town, 'xml_id': args.xml_id, 'version': VERSION}
    try:
        if not args.reuse_server:
            server, stream = start_server(args.port, server_log, args.res_x, args.res_y)
            if not wait_for_port(args.host, args.port, args.server_timeout):
                raise RuntimeError(f'CARLA server did not open port {args.port}')
            time.sleep(3.0)

        command = [
            str(CARLA_PYTHON), str(EVALUATOR),
            '--host', args.host, '--port', str(args.port),
            '--traffic-manager-port', str(args.port + 5000),
            '--routes', str(args.xml), '--routes-subset', str(args.xml_id),
            '--repetitions', '1', '--track', 'MAP',
            '-a', str(AGENT), '--agent-config', '',
            '--timeout', str(args.timeout), '--debug', '0',
            '--checkpoint', str(checkpoint.resolve()),
        ]
        with evaluator_log.open('w') as log:
            completed = subprocess.run(command, cwd=str(B2D_ROOT), env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout + 300)
        result['exit_code'] = completed.returncode
    except subprocess.TimeoutExpired:
        result['exit_code'] = None
        result['error'] = 'evaluator_timeout'
    except Exception as error:
        result['exit_code'] = None
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if server is not None:
            stop_server(server, stream)

    result['wall_seconds'] = round(time.time() - started, 1)
    if capture_path.exists():
        record = json.loads(capture_path.read_text())
        result['status'] = 'CAPTURED'
        result['capture_sha256'] = record['sha256']
        result['raw_points'] = record['raw']['world_point_count']
        result['agent_points'] = record['agent_view']['point_count']
        result['road_option_counts'] = record['raw']['road_option_counts']
        result['map_name'] = record['metadata'].get('map_name')
        result['server_version'] = record['metadata'].get('carla_server_version')
    else:
        result['status'] = 'NO_CAPTURE'
        tail = ''
        if evaluator_log.exists():
            tail = '\n'.join(evaluator_log.read_text(errors='replace').splitlines()[-20:])
        result['evaluator_tail'] = tail

    index = args.output / f'{args.capture_id}.run.json'
    index.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'CAPTURED' else 1


if __name__ == '__main__':
    sys.exit(main())
