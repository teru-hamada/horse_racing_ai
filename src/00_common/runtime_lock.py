"""Cross-process, reentrant lock shared by runtime transfer and DB connections."""
from contextlib import contextmanager
from pathlib import Path
import os
import threading
import time

_states = {}
_states_lock = threading.Lock()


@contextmanager
def runtime_lock(root, *, allow_recovery=False, wait=False):
    root = Path(root).resolve()
    with _states_lock:
        state = _states.setdefault(str(root), {'mutex': threading.RLock(), 'depth': 0})
    acquired = state['mutex'].acquire(timeout=30) if wait else state['mutex'].acquire(blocking=False)
    if not acquired:
        raise RuntimeError('データを使用・同期しています。処理完了後に再実行してください。')
    stream = None
    try:
        if not allow_recovery and (root / '.private-runtime/activation-pending.json').exists():
            raise RuntimeError('前回の復元が中断されています。バックアップ復旧を実行してください。')
        if state['depth'] == 0:
            path = root / '.private-runtime/runtime.lock'
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a+b')
            if path.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            deadline = time.monotonic() + (30 if wait else 0)
            while True:
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        stream.close()
                        stream = None
                        raise RuntimeError('別のプロセスがデータを使用・同期しています。') from None
                    time.sleep(0.05)
        state['depth'] += 1
        try:
            yield
        finally:
            state['depth'] -= 1
    finally:
        if stream is not None:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        state['mutex'].release()


class LockedConnection:
    """Keep the runtime lock for the full lifetime of a database connection."""
    def __init__(self, connection, lease):
        self.connection, self.lease = connection, lease

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        return self.connection

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.lease is not None:
            try:
                self.connection.close()
            finally:
                lease, self.lease = self.lease, None
                lease.__exit__(None, None, None)


def connect_locked(root, database, factory):
    lease = runtime_lock(root, wait=True)
    lease.__enter__()
    try:
        return LockedConnection(factory(str(database)), lease)
    except BaseException:
        lease.__exit__(None, None, None)
        raise
