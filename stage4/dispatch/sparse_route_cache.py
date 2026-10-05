"""Bounded, on-demand raw routing cache (not an all-pairs OD product)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time


class SparseRouteDiskCache:
    def __init__(self, path, context, *, limit_mib=512, max_entries=2_000_000):
        if not 16 <= limit_mib <= 1024 or not 1 <= max_entries <= 2_000_000:
            raise ValueError("disk route cache capacity is out of bounds")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context_sha256 = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.max_entries = int(max_entries)
        self.limit_bytes = int(limit_mib) * 2**20
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA journal_mode=TRUNCATE")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-8192")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("PRAGMA temp_store=FILE")
        page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
        self.db.execute(f"PRAGMA max_page_count={self.limit_bytes // page_size}")
        self.db.execute("CREATE TABLE IF NOT EXISTS routes (key BLOB PRIMARY KEY, raw REAL NOT NULL, distance REAL NOT NULL, used INTEGER NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS route_recency ON routes(used)")
        self.db.commit()
        self.count = int(self.db.execute("SELECT COUNT(*) FROM routes").fetchone()[0])
        self.pending = {}
        self.touched = set()
        self.evicted = 0
        self.disabled_writes = False

    def key(self, payload):
        # payload includes exact float.hex coordinates and minute, not beta.
        return hashlib.sha256((self.context_sha256 + "|" + "|".join(payload)).encode()).digest()

    def get_many(self, keys):
        keys = list(dict.fromkeys(keys))
        result = {k: self.pending[k] for k in keys if k in self.pending}
        missing = [k for k in keys if k not in result]
        for left in range(0, len(missing), 400):
            chunk = missing[left:left + 400]
            placeholders = ",".join("?" for _ in chunk)
            for key, raw, distance in self.db.execute(
                    f"SELECT key, raw, distance FROM routes WHERE key IN ({placeholders})", chunk):
                result[key] = (float(raw), float(distance))
                self.touched.add(key)
        if len(self.touched) >= 4096:
            self.flush()
        return result

    def remember(self, key, raw, distance):
        if self.disabled_writes:
            return
        self.pending[key] = (float(raw), float(distance))
        if len(self.pending) >= min(1024, self.max_entries):
            self.flush()

    def flush(self):
        if not self.pending and not self.touched:
            return
        stamp = time.time_ns()
        try:
            # Leave slack for indexes/journal; allocated pages never grow past
            # max_page_count. Deleted pages are reused, no automatic VACUUM.
            pages = self.db.execute("PRAGMA page_count").fetchone()[0]
            free_pages = self.db.execute("PRAGMA freelist_count").fetchone()[0]
            page_size = self.db.execute("PRAGMA page_size").fetchone()[0]
            excess = max(0, self.count + len(self.pending) - self.max_entries)
            if (pages - free_pages) * page_size > self.limit_bytes * .90:
                excess = max(excess, 8192)
            if excess:
                before = self.db.total_changes
                self.db.execute("DELETE FROM routes WHERE key IN (SELECT key FROM routes ORDER BY used LIMIT ?)", (excess,))
                deleted = self.db.total_changes - before
                self.count -= deleted
                self.evicted += deleted
            self.db.executemany("UPDATE routes SET used=? WHERE key=?", ((stamp, key) for key in self.touched))
            before = self.db.total_changes
            self.db.executemany("INSERT OR IGNORE INTO routes VALUES (?,?,?,?)",
                ((key, raw, distance, stamp) for key, (raw, distance) in self.pending.items()))
            self.count += self.db.total_changes - before
            self.db.commit()
        except sqlite3.OperationalError as error:
            self.db.rollback()
            if "full" not in str(error).casefold():
                raise
            # A cache is optional; capacity pressure cannot change an arc.
            self.disabled_writes = True
            self.count = int(self.db.execute("SELECT COUNT(*) FROM routes").fetchone()[0])
        self.pending.clear()
        self.touched.clear()

    def close(self):
        self.flush()
        self.db.close()
