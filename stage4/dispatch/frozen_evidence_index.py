"""Compact exact row lookup on immutable Arrow evidence, not a dense graph."""
from __future__ import annotations

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def _hash(value):
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


class FrozenColumnIndex:
    """Hash-sorted dictionary keys and integer row lists; collisions are checked.

    Returned rows retain source order, including duplicate boundary memberships.
    No Python dictionary/list per network edge is kept alive. Missing keys do
    not grow the index. The caller owns the immutable source table.
    """

    def __init__(self, table, column):
        if len(table) >= 2**31:
            raise ValueError("frozen evidence index exceeds compact row capacity")
        self.table, self.column = table, column
        encoded = pc.dictionary_encode(table[column].combine_chunks())
        if encoded.null_count:
            raise ValueError("frozen evidence keys must be nonnull")
        self.keys = encoded.dictionary
        codes = encoded.indices.to_numpy()
        hashes = np.fromiter((_hash(value) for value in self.keys.to_pylist()),
            dtype=np.uint64, count=len(self.keys))
        self.key_order = np.argsort(hashes, kind="stable").astype(np.int32)
        self.hashes = hashes[self.key_order]
        self.rows = np.argsort(codes, kind="stable").astype(np.int32)
        self.ends = np.cumsum(np.bincount(codes, minlength=len(self.keys)), dtype=np.int32)
        self.starts = np.r_[np.int32(0), self.ends[:-1]].astype(np.int32)
        self.nbytes = self.keys.nbytes + sum(a.nbytes for a in (
            self.key_order,self.hashes,self.rows,self.ends,self.starts))

    def select(self, values):
        pieces = []
        values = tuple(set(values))
        # uint64 query arrays avoid NumPy's Python-int/uint64 promotion, which
        # can otherwise recast the WHOLE index on every scalar lookup.
        hashes = np.fromiter((_hash(str(value)) for value in values),
            dtype=np.uint64,count=len(values))
        lefts = np.searchsorted(self.hashes, hashes, side="left")
        rights = np.searchsorted(self.hashes, hashes, side="right")
        for value,left,right in zip(values,lefts,rights):
            for position in range(left,right):
                code = int(self.key_order[position])
                if self.keys[code].as_py() == value:
                    pieces.append(self.rows[self.starts[code]:self.ends[code]])
        rows = np.sort(np.concatenate(pieces)) if pieces else np.empty(0,dtype=np.int32)
        return self.table.take(pa.array(rows))
