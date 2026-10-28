"""Observe an existing fit PID; never signal it or change model/cache contents."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

import psutil

from tools.training.diffusiondrive_v1.features_cache import active_mount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.output_dir:
        cfg['output_dir']=str(args.output_dir)
        cfg['output_storage']=dict(cfg['output_storage'],path=str(args.output_dir))
    process = psutil.Process(args.pid)
    assert 'tools.training.diffusiondrive_v1.fit' in process.cmdline()
    started = process.create_time()
    path = Path(cfg['output_dir']) / 'resource_monitor.jsonl'
    while process.is_running() and process.create_time() == started:
        # Refuse to create a file on tmpfs/system disk after losing the mount.
        info = active_mount(dict(cfg['output_storage'], mountpoint='/run/media/aethelind/PSSD'), writing=True)
        rss = read = written = 0
        for child in [process] + process.children(recursive=True):
            try:
                rss += child.memory_info().rss
                io = child.io_counters(); read += io.read_bytes; written += io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': args.pid,
                  'tree_rss_bytes': rss, 'tree_read_bytes': read, 'tree_write_bytes': written,
                  'pssd_free_bytes': info['free_bytes'], 'gpu_memory_mib_utilization_percent_temperature_c': gpu.stdout.strip()}
        with path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        time.sleep(30)


if __name__ == '__main__':
    main()
