"""Resume scheduling only; keep the frozen numeric/cache producer unchanged.

The legacy worker accepts its SQLite handle via features_cache.connect. This
adapter restricts only its initial ordered sweep to unfinished rows. All writes,
source checks, artifact checks for pending/orphan files and ordinary queries use
the original SQLite API. Completed-file auditing is an explicit separate action.
"""
import json
from pathlib import Path
import sqlite3

from tools.b2d_adapter.errors import AdapterError
from tools.training.diffusiondrive_v1 import features_cache as core

LEGACY_SWEEP = 'select * from samples order by route,frame'


class PendingConnection(sqlite3.Connection):
    def __init__(self, *args, retry_failed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_failed = retry_failed

    def execute(self, sql, parameters=()):
        if ' '.join(sql.strip().split()).lower() == LEGACY_SWEEP:
            states = dict(super().execute('SELECT state,COUNT(*) FROM samples GROUP BY state').fetchall())
            print(json.dumps({'event': 'resume_queue', 'completed_not_loaded': states.get('complete', 0),
                              'pending': states.get('pending', 0),
                              'failed_retry_enabled': self.retry_failed}), flush=True)
            predicate = "state IN ('pending','failed')" if self.retry_failed else "state='pending'"
            sql = 'SELECT * FROM samples WHERE ' + predicate + ' ORDER BY route,frame'
        return super().execute(sql, parameters)


def run_pending(config, state, limit=None, preferred=None, retry_failed=False):
    original_connect = core.connect
    def queue_connect(path):
        # Executed after the original worker acquires writer.lock. Reject a
        # pre-cutover config that was loaded just before relocation finished.
        from tools.training.diffusiondrive_v1.feature_migration import migration_preflight
        migration_preflight(state, config)
        # Reuse the existing schema and durability settings, not a second schema.
        bootstrap = original_connect(path)
        bootstrap.execute('CREATE INDEX IF NOT EXISTS pending_order ON samples(state,route,frame)')
        bootstrap.commit()
        bootstrap.close()
        db = sqlite3.connect(path, timeout=30,
                             factory=lambda *a, **kw: PendingConnection(*a, retry_failed=retry_failed, **kw))
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        return db
    # Single-threaded worker scope; no numeric function or producer is replaced.
    core.connect = queue_connect
    try:
        from tools.training.diffusiondrive_v1.feature_shared_storage import storage_scope
        with storage_scope(config):
            return core.run(config, state, limit, preferred, retry_failed)
    finally:
        core.connect = original_connect


def verify_completed(config, state, limit=None):
    """Audit completed blobs without changing their index or file contents.

    EIO/permission/mount/identity errors abort the audit, not invalidate the corpus.
    Raw JPG/LAZ files are not read. Dataset reads continue to verify all checksums.
    """
    import fcntl
    from tools.training.diffusiondrive_v1.feature_service import alive
    state = Path(state)
    if alive(state):
        raise core.StoragePause('Stop the worker before verifying completed files')
    with (state / 'writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise core.StoragePause('Another writer is active') from exc
        db = core.connect(state / 'index.sqlite')
        try:
            producer = core.feature_producer(config)
            pinned = db.execute("SELECT value FROM meta WHERE key='producer'").fetchone()
            if pinned is None:
                raise core.StoragePause('No pinned producer; initialize the job before verification')
            recorded = json.loads(pinned[0])
            if core.digest(recorded) != core.digest(producer):
                raise core.StoragePause('Verification producer differs from the pinned index')
            sql = "SELECT * FROM samples WHERE state='complete' ORDER BY route,frame"
            params = ()
            if limit is not None:
                if limit < 1: raise ValueError('Verification limit must be positive')
                sql += ' LIMIT ?'; params = (limit,)
            rows = db.execute(sql, params).fetchall()
            result = {'checked': 0, 'valid': 0, 'invalid': 0, 'samples': []}
            with (state / 'verification.jsonl').open('a', buffering=1) as journal:
                for row in rows:
                    reason = None
                    try:
                        core.load_feature_row(dict(row), config['cache_roots'], core.digest(producer))
                    except FileNotFoundError:
                        reason = 'missing_cache_file'
                    except AdapterError as exc:
                        if exc.code != 'invalid_feature_cache': raise
                        reason = str(exc)
                    except OSError as exc:
                        # Device faults must remain a safe pause, not a repair retry.
                        raise core.StoragePause('Cache verification I/O error: ' + str(exc)) from exc
                    result['checked'] += 1
                    if reason:
                        result['invalid'] += 1
                        event = {'sample_id': row['sample_id'], 'state': 'invalid', 'reason': reason}
                        result['samples'].append(event)
                    else:
                        result['valid'] += 1
                        event = {'sample_id': row['sample_id'], 'state': 'valid'}
                    journal.write(json.dumps(event) + '\n')
            core.durable_json(state / 'verification_result.json', result)
            return result
        finally:
            db.close()
