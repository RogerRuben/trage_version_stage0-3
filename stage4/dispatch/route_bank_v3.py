"""Read-mostly raw cache with bounded optional background writes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from queue import Queue, Full
import sqlite3
from threading import Thread
import time


class BufferedRouteBank:
    def __init__(self, path, context, *, limit_mib=512, max_entries=2_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context_sha256 = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.limit_bytes, self.max_entries = limit_mib * 2**20, int(max_entries)
        if not 16 <= limit_mib <= 512 or not 1 <= max_entries <= 2_000_000:
            raise ValueError("v3 route bank exceeds bounded storage")
        self.reader = sqlite3.connect(self.path, timeout=.1)
        self._configure(self.reader)
        self.reader.execute("CREATE TABLE IF NOT EXISTS raw_routes (key BLOB PRIMARY KEY, raw REAL NOT NULL, distance REAL NOT NULL) WITHOUT ROWID")
        self.reader.commit()
        self.entry_count = self.reader.execute("SELECT COUNT(*) FROM raw_routes").fetchone()[0]
        self.pending = {}
        self.queue = Queue(maxsize=2)
        self.dropped_writes = 0
        self.write_time_s = 0.
        self.write_error = None
        self.disabled_writes = False
        self.thread = Thread(target=self._write_loop, name="BoundedRawRouteWriter", daemon=True)
        self.thread.start()

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
        missing = [k for k in keys if k not in found]
        for left in range(0, len(missing), 400):
            block = missing[left:left + 400]
            for key, raw, distance in self.reader.execute(
                    "SELECT key,raw,distance FROM raw_routes WHERE key IN (" + ",".join("?" for _ in block) + ")", block):
                found[key] = (float(raw), float(distance))
        return found

    def remember(self, key, raw, distance):
        if self.disabled_writes:
            self.dropped_writes += 1
            return
        self.pending[key] = (float(raw), float(distance))
        if len(self.pending) >= 4096:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        block, self.pending = self.pending, {}
        try:
            self.queue.put_nowait(block)
        except Full:
            # Cache loss is harmless: never drop an answer or candidate arc.
            self.dropped_writes += len(block)

    def _write_loop(self):
        db = sqlite3.connect(self.path, timeout=2)
        self._configure(db)
        while True:
            block = self.queue.get()
            if block is None:
                self.queue.task_done()
                break
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
                db.rollback()
                self.write_error = type(error).__name__ + ":" + str(error)
                self.disabled_writes = True
                self.dropped_writes += len(block)
            finally:
                self.write_time_s += time.perf_counter() - start
                self.queue.task_done()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()

    def close(self):
        self.flush()
        self.queue.put(None)
        self.thread.join()
        self.reader.close()
