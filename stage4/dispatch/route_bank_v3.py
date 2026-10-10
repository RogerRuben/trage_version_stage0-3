"""Read-mostly raw cache with bounded optional background writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from queue import Queue, Full, Empty
import sqlite3
from threading import Thread, Event
import time


class BufferedRouteBank:
    def __init__(self, path, context, *, limit_mib=512, max_entries=2_000_000):
        self.path = Path(path)
        self.context_sha256 = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.limit_bytes, self.max_entries = limit_mib * 2**20, int(max_entries)
        if not 16 <= limit_mib <= 512 or not 1 <= max_entries <= 2_000_000:
            raise ValueError("v3 route bank exceeds bounded storage")
        self.reader = None
        self.entry_count = 0
        self.pending = {}
        self.queue = Queue(maxsize=2)
        self.dropped_writes = 0
        self.write_time_s = 0.
        self.write_error = None
        self.read_error = None
        self.close_error = None
        self.disabled_writes = False
        self.disabled_reads = False
        self.closed = False
        self._closing = Event()
        self._writer_done = Event()
        self.thread = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.reader = sqlite3.connect(self.path, timeout=.1)
            self._configure(self.reader)
            self.reader.execute("CREATE TABLE IF NOT EXISTS raw_routes (key BLOB PRIMARY KEY, raw REAL NOT NULL, distance REAL NOT NULL) WITHOUT ROWID")
            self.reader.commit()
            self.entry_count = self.reader.execute("SELECT COUNT(*) FROM raw_routes").fetchone()[0]
        except (OSError, sqlite3.Error) as error:
            self.read_error = type(error).__name__ + ":" + str(error)
            self.disabled_reads = self.disabled_writes = True
            if self.reader is not None:
                self.reader.close()
                self.reader = None
            self._writer_done.set()
            return  # Optional cache failed; caller still computes every arc.
        try:
            self.thread = Thread(target=self._write_loop, name="BoundedRawRouteWriter", daemon=True)
            self.thread.start()
        except Exception as error:
            self._write_failed(error)
            self.thread = None
            self._writer_done.set()

    def _configure(self, db):
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA cache_size=-4096")
        db.execute("PRAGMA mmap_size=0")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA wal_autocheckpoint=256")
        db.execute("PRAGMA journal_size_limit=8388608")
        page = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={self.limit_bytes // page}")

    def key(self, payload):
        return hashlib.sha256((self.context_sha256 + "|" + "|".join(payload)).encode()).digest()

    def get_many(self, keys):
        keys = list(dict.fromkeys(keys))
        found = {k: self.pending[k] for k in keys if k in self.pending}
        if self.disabled_reads or self.reader is None or self.closed:
            return found
        missing = [k for k in keys if k not in found]
        for left in range(0, len(missing), 400):
            block = missing[left:left + 400]
            try:
                for key, raw, distance in self.reader.execute(
                        "SELECT key,raw,distance FROM raw_routes WHERE key IN (" + ",".join("?" for _ in block) + ")", block):
                    found[key] = (float(raw), float(distance))
            except sqlite3.Error as error:
                self.read_error = type(error).__name__ + ":" + str(error)
                self.disabled_reads = True
                break
        return found

    def remember(self, key, raw, distance):
        if self.disabled_writes or self.closed or self._writer_done.is_set():
            self.dropped_writes += 1
            return
        self.pending[key] = (float(raw), float(distance))
        if len(self.pending) >= 4096:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        block, self.pending = self.pending, {}
        if self.disabled_writes or self._writer_done.is_set():
            self.dropped_writes += len(block)
            return
        try:
            self.queue.put_nowait(block)
        except Full:
            # Cache loss is harmless: never drop an answer or candidate arc.
            self.dropped_writes += len(block)

    def _write_failed(self, error):
        self.write_error = type(error).__name__ + ":" + str(error)
        self.disabled_writes = True

    def _discard_queued(self):
        while True:
            try:
                block = self.queue.get_nowait()
            except Empty:
                break
            self.dropped_writes += len(block)
            self.queue.task_done()

    def _write_loop(self):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=2)
            self._configure(db)  # Startup errors must not escape the thread.
            while True:
                try:
                    block = self.queue.get(timeout=.1)
                except Empty:
                    if self._closing.is_set():
                        break
                    continue
                start = time.perf_counter()
                try:
                    room = max(0, self.max_entries - self.entry_count)
                    if not self.disabled_writes and room:
                        entries = list(block.items())[:room]
                        before = db.total_changes
                        db.executemany("INSERT OR IGNORE INTO raw_routes VALUES (?,?,?)",
                            ((key, raw, distance) for key, (raw, distance) in entries))
                        db.commit()
                        self.entry_count += db.total_changes - before
                        self.dropped_writes += len(block) - len(entries)
                    else:
                        self.dropped_writes += len(block)
                except sqlite3.Error as error:
                    self._write_failed(error)
                    self.dropped_writes += len(block)
                    try:
                        db.rollback()
                    except sqlite3.Error:
                        pass
                finally:
                    self.write_time_s += time.perf_counter() - start
                    self.queue.task_done()
            try:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as error:
                self._write_failed(error)
        except Exception as error:
            self._write_failed(error)
        finally:
            self._discard_queued()
            if db is not None:
                try:
                    db.close()
                except sqlite3.Error as error:
                    self._write_failed(error)
            self._writer_done.set()

    def close(self):
        if self.closed:
            return
        self.flush()
        self._closing.set()  # Never block on putting a sentinel into a full queue.
        if self.thread is not None:
            self.thread.join(timeout=5.)
            if self.thread.is_alive():
                self.close_error = "optional cache writer did not finish within 5 seconds"
                self.disabled_writes = True
        self._discard_queued()
        if self.reader is not None:
            try:
                self.reader.close()
            except sqlite3.Error as error:
                self.close_error = type(error).__name__ + ":" + str(error)
            self.reader = None
        self.closed = True

    def diagnostics(self):
        return dict(entry_count=self.entry_count, dropped_optional_writes=self.dropped_writes,
            write_time_s=self.write_time_s, read_error=self.read_error, write_error=self.write_error,
            close_error=self.close_error, disabled_reads=self.disabled_reads,
            disabled_writes=self.disabled_writes, closed=self.closed,
            writer_alive=self.thread is not None and self.thread.is_alive())
