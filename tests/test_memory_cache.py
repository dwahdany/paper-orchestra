from __future__ import annotations

import threading

import pytest

from paper_orchestra.memory import KeyValueCache


def test_round_trip_and_separate_files(tmp_path):
    cache = KeyValueCache(tmp_path / "cache")
    assert cache.get("scholar", "a title") is None and cache.count("scholar") == 0
    assert cache.put("scholar", "a title", {"year": 2020, "authors": ["Ada"]})
    assert cache.put("scholar", "another", [1, 2, 3])
    assert cache.get("scholar", "a title") == {"year": 2020, "authors": ["Ada"]}
    assert cache.get("scholar", "another") == [1, 2, 3]
    assert cache.get("other", "a title") is None
    assert cache.count("scholar") == 2
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 2
    assert not list((tmp_path / "cache").rglob("*.tmp"))
    cache.put("scholar", "a title", "replaced")
    assert cache.get("scholar", "a title") == "replaced" and cache.count("scholar") == 2


def test_bad_entries_and_names_are_handled(tmp_path, caplog):
    cache = KeyValueCache(tmp_path / "cache")
    cache.put("scholar", "key", 1)
    (file,) = (tmp_path / "cache").rglob("*.json")
    file.write_text("not json")
    assert cache.get("scholar", "key") is None and "unreadable cache entry" in caplog.text
    file.write_text('{"key": "a different key", "value": 2}')
    assert cache.get("scholar", "key") is None  # a digest collision is never served
    for namespace in ("", "../up", "With Space", "a/b"):
        with pytest.raises(ValueError):
            cache.put(namespace, "key", 1)
    assert not cache.put("scholar", "unserialisable", {1, 2})
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_parallel_writers_do_not_lose_entries(tmp_path):
    caches = [KeyValueCache(tmp_path / "cache") for _ in range(8)]

    def fill(index: int) -> None:
        for item in range(25):
            caches[index].put("scholar", f"writer {index} item {item}", item)
            caches[index].put("scholar", "shared", index)

    threads = [threading.Thread(target=fill, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert caches[0].count("scholar") == 8 * 25 + 1
    assert caches[0].get("scholar", "shared") in range(8)
    assert all(caches[0].get("scholar", f"writer {i} item 24") == 24 for i in range(8))
