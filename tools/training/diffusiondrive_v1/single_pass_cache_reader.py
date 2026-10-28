"""One artifact open: SHA the bytes, then deserialize the identical buffer.

The frozen Feature/Target readers still perform every identity/value check.
Like target_storage.codec_scope, this adapter is scoped to the single-threaded
Dataset reader in each CPU worker. It never changes a numeric cache producer.
"""
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path

import torch

from tools.training.diffusiondrive_v1 import features_cache, target_cache, target_storage
from tools.training.diffusiondrive_v1.features_cache import StoragePause


def identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class VerifiedBufferReader:
    def __init__(self, loader):
        self.loader = loader
        self.pending = {}
        self.stats = {'artifact_opens': 0, 'artifact_bytes': 0, 'deserializations': 0}

    def digest(self, filename):
        path = Path(filename)
        if path in self.pending:
            raise StoragePause('Duplicate unconsumed artifact digest: ' + str(path))
        try:
            with path.open('rb') as stream:
                before = identity(os.fstat(stream.fileno()))
                data = stream.read()
                after = identity(os.fstat(stream.fileno()))
            if before != after or len(data) != before[2] or identity(path.stat()) != before:
                raise StoragePause('Artifact changed during single-pass read: ' + str(path))
            sha = hashlib.sha256(data).hexdigest()
            self.pending[path] = (io.BytesIO(data), before)
            self.stats['artifact_opens'] += 1
            self.stats['artifact_bytes'] += len(data)
            return sha
        except OSError as exc:
            raise StoragePause('Single-pass artifact read failed: ' + str(path)) from exc

    def load(self, filename, *args, **kwargs):
        path = Path(filename)
        if path not in self.pending:
            raise StoragePause('Deserialize requested without preceding SHA read: ' + str(path))
        buffer, expected = self.pending.pop(path)
        try:
            if identity(path.stat()) != expected:
                raise StoragePause('Artifact replaced between SHA and decode: ' + str(path))
            # No path is passed to torch.load, and no mmap/path reopen occurs.
            result = self.loader(buffer, *args, **kwargs)
            if identity(path.stat()) != expected:
                raise StoragePause('Artifact changed during decode: ' + str(path))
            self.stats['deserializations'] += 1
            return result
        except OSError as exc:
            raise StoragePause('Single-pass artifact decode failed: ' + str(path)) from exc
        finally:
            buffer.close()

    def close(self):
        for buffer, _ in self.pending.values():
            buffer.close()
        self.pending.clear()


@contextmanager
def single_pass_scope():
    # Use the codec's lock so its nested Target scope cannot race these bindings.
    # RAW_LOAD must also be redirected: codec_scope intentionally bypasses the
    # global torch.load to avoid recursive schema decoding.
    with target_storage.CODEC_LOCK:
        original = (features_cache.file_digest, target_cache.file_digest,
                    torch.load, target_storage.RAW_LOAD)
        reader = VerifiedBufferReader(original[2])
        features_cache.file_digest = reader.digest
        target_cache.file_digest = reader.digest
        torch.load = reader.load
        target_storage.RAW_LOAD = reader.load
        try:
            yield reader
            if reader.pending:
                raise StoragePause('Verified buffer was not consumed by the frozen cache loader')
        finally:
            features_cache.file_digest, target_cache.file_digest, torch.load, target_storage.RAW_LOAD = original
            reader.close()
