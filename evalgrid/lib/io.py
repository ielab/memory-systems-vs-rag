"""
evalgrid/lib/io.py
------------------
Atomic JSON writes, for the files several processes write to one directory.

A sharded ingest is N processes (65 for a_mem, 70 for hypermem) writing into one
``mem_dir``. Their PER-SAMPLE files are safe by construction — each sample is
owned by exactly one shard — but the directory-level ones are not: ``config.json``,
``manifest.json`` and ``ingest_usage.json`` are whole-file rewrites that every
shard performs, one at its start and one at its end. A plain ``open(path, "w")``
truncates first and writes after, so a reader (or a second writer) can see a
half-written file, and two writers can interleave into one that parses as
neither.

Writing to a temporary file in the same directory and ``os.replace``-ing it over
the target makes the swap atomic on POSIX: a reader sees either the old complete
file or the new complete one, and concurrent writers produce one intact winner
rather than a blend. Same directory matters — ``os.replace`` is only atomic
within a filesystem.

This does NOT make the writes ordered: the last shard to finish still wins. That
is correct for these three files, because each is derived from the directory's
own contents rather than from the writing process's state (see
runner._write_aggregate_ingest_usage, which re-reads every sample's usage.json).
"""

from __future__ import annotations

import json
import os
import tempfile

#: The mode a plain ``open(path, "w")`` would have produced. mkstemp creates at
#: 0600, so without this an atomically-written config.json would become
#: unreadable to everyone but its author — a silent permissions change on a
#: shared cluster filesystem, for files a collaborator is expected to read.
#: Read once at import, while the process is still single-threaded.
_UMASK = os.umask(0)
os.umask(_UMASK)
_FILE_MODE = 0o666 & ~_UMASK


def write_json_atomic(path: str, obj, **json_kwargs) -> None:
    """Serialise ``obj`` to ``path`` via a same-directory temp file + os.replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-",
                               suffix=os.path.basename(path))
    try:
        os.chmod(tmp, _FILE_MODE)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **json_kwargs)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave the temp file behind — a crashed shard would otherwise
        # litter mem_dir with files that scandir-based readers then walk.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
