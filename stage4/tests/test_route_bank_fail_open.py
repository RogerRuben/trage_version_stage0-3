"""Two failure checks: optional cache loss never blocks dispatch/close."""
import sqlite3
from threading import current_thread
import time

from stage4.dispatch.route_bank_v3 import BufferedRouteBank


def test_writer_startup_failure_and_full_queue_close_are_bounded(tmp_path, monkeypatch):
    original = BufferedRouteBank._configure
    def fail_worker(self, db):
        if current_thread().name == "BoundedRawRouteWriter":
            raise sqlite3.OperationalError("injected writer startup failure")
        return original(self, db)
    monkeypatch.setattr(BufferedRouteBank, "_configure", fail_worker)
    bank = BufferedRouteBank(tmp_path / "cache.db", {"test": "frozen"}, limit_mib=16)
    assert bank._writer_done.wait(timeout=2.)
    assert bank.disabled_writes and "startup failure" in bank.write_error
    # Reproduce exactly the old dead-writer/full-queue close hazard.
    bank.queue.put_nowait({b"a": (1., 10.)})
    bank.queue.put_nowait({b"b": (2., 20.)})
    start = time.monotonic()
    bank.close(); bank.close()
    assert time.monotonic() - start < 1.
    assert bank.closed and not bank.diagnostics()["writer_alive"]
    assert bank.queue.empty()


def test_cache_open_and_read_failures_fall_back_to_empty_lookup(tmp_path, monkeypatch):
    connect = sqlite3.connect
    def fail_open(*args, **kwargs):
        raise sqlite3.OperationalError("injected optional cache open failure")
    monkeypatch.setattr(sqlite3, "connect", fail_open)
    bank = BufferedRouteBank(tmp_path / "cache.db", {"test": "frozen"}, limit_mib=16)
    assert bank.get_many([b"a"]) == {}
    bank.remember(b"a", 1., 10.)
    bank.close()
    assert bank.disabled_reads and bank.disabled_writes and bank.read_error
    monkeypatch.setattr(sqlite3, "connect", connect)
    healthy = BufferedRouteBank(tmp_path / "read.db", {"test": "frozen"}, limit_mib=16)
    healthy.reader.close()  # Force a read error without stopping the caller.
    assert healthy.get_many([b"missing"]) == {}
    assert healthy.disabled_reads and healthy.read_error
    healthy.close()
