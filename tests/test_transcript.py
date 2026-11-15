from __future__ import annotations

import json
import threading

from paper_orchestra.memory.transcript import MAX_FIELD_CHARS, TranscriptLog


def test_record_stamps_and_appends(tmp_path, fixed_clock):
    log = TranscriptLog(tmp_path, session_id="s1", clock=fixed_clock)
    first = log.record({"type": "stage_start", "stage": "outline"})
    second = log.record({"type": "stage_end", "stage": "outline", "seconds": 1.5})
    assert (first["seq"], second["seq"]) == (1, 2)
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [e["type"] for e in lines] == ["stage_start", "stage_end"]
    assert all(e["session"] == "s1" and e["ts"].endswith("Z") for e in lines)


def test_reopening_a_session_continues_the_sequence(tmp_path):
    TranscriptLog(tmp_path, session_id="s1").record({"type": "a"})
    reopened = TranscriptLog(tmp_path, session_id="s1")
    assert reopened.record({"type": "b"})["seq"] == 2


def test_long_fields_and_bytes_are_clipped(tmp_path):
    log = TranscriptLog(tmp_path, session_id="s1")
    event = log.record(
        {"type": "assistant", "text": "a" * (MAX_FIELD_CHARS * 3), "raw": b"\x00" * 9}
    )
    assert len(event["text"]) < MAX_FIELD_CHARS + 200
    assert "characters omitted" in event["text"]
    assert event["raw"] == "[9 bytes]"
    json.loads(log.path.read_text().splitlines()[0])


def test_concurrent_writers_do_not_interleave(tmp_path):
    log = TranscriptLog(tmp_path, session_id="s1")
    sink = log.sink(stage="plotting", agent="worker")

    def work(i: int) -> None:
        for j in range(25):
            sink({"type": "tool_result", "content": f"{i}-{j}"})

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = list(log.events())
    assert len(events) == 200
    assert sorted(e["seq"] for e in events) == list(range(1, 201))
    assert all(e["stage"] == "plotting" and e["agent"] == "worker" for e in events)


def test_search_filters_and_ranks(tmp_path, fixed_clock):
    old = TranscriptLog(tmp_path, session_id="a-old", clock=fixed_clock)
    old.record(
        {
            "type": "compile",
            "stage": "section_writing",
            "errors": ["Undefined control sequence \\cref"],
        }
    )
    new = TranscriptLog(tmp_path, session_id="b-new", clock=fixed_clock)
    new.record(
        {
            "type": "assistant",
            "stage": "refinement",
            "text": "fixed the undefined control sequence by loading cleveref",
        }
    )
    new.record({"type": "assistant", "stage": "refinement", "text": "unrelated"})

    hits = new.search("undefined control sequence cleveref")
    assert [h.session for h in hits] == ["b-new", "a-old"]
    assert "cleveref" in hits[0].snippet
    assert [h.session for h in new.search("undefined control", stage="section_writing")] == [
        "a-old"
    ]
    assert [h.type for h in new.search("undefined control", types=["compile"])] == ["compile"]
    assert new.search("undefined", session="b-new")[0].session == "b-new"
    assert new.search("zzzz") == [] and new.search("the") == []


def test_corrupt_lines_are_skipped(tmp_path):
    log = TranscriptLog(tmp_path, session_id="s1")
    log.record({"type": "a"})
    with log.path.open("a") as handle:
        handle.write("{not json\n")
    log.record({"type": "b"})
    assert [e["type"] for e in log.events()] == ["a", "b"]
