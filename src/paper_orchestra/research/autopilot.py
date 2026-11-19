"""Autopilot: a project that works without the person, up to a cap.

With ``settings.autonomy.mode: autopilot`` the person has said beforehand what is to
happen without them.  Their standing rule then decides in their place (what it
decides is in ``research.project`` and ``research.experiments``, each next to the
decision of the person it stands in for), and the worker goes on from phase to phase
by itself (``research.engine``).  This module holds what the two share and what whoever
shows the project asks for: the record of one unattended **stretch**, the reasons a
stretch ends, and the digest of what happened in it ("While you were away").

A stretch begins when the person starts the agents (Continue in the app, ``research
advance``) and ends at a checkpoint when a cap of ``settings.autonomy`` is reached
(rounds, dollars of model spend, hours), when the project's own limits stop the work,
when a phase fails, when the same experiment fails twice, when nothing more can
happen without the person, or when they press Stop.  Nothing starts a new one but the
person: an ended stretch stays ended, whatever an agent does.

The record of the stretch lies in the engine's ``state.json`` under ``autopilot`` (a
project that never ran unattended has no such key), and the events
``autopilot.started``, ``autopilot.continued`` and ``autopilot.stopped`` say the same in
the feed; the digest is counted from the project's events since the stretch began, so
it is on file, not in anybody's memory.

One thing no count here holds: what the code of an experiment spends on model calls of
its own.  A run with network access can call a model API with whatever credential it
finds (and with the worker's federation settings where model access for jobs is on);
the usage ledger counts the agents' sessions, not those calls.  The digest says so, and
gives the one cheap bound there is: how many runs had network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from paper_orchestra.research.model import AUTOPILOT, SECTIONS, parse_stamp

if TYPE_CHECKING:
    from paper_orchestra.research.project import Project

__all__ = [
    "CONTINUED",
    "REASONS",
    "STARTED",
    "STOPPED",
    "STRETCH_KEY",
    "UNCOUNTED",
    "caps_in_words",
    "digest",
    "grouped",
    "new_stretch",
    "read_stretch",
    "summary",
]

#: The key of ``state.json`` that holds the record of the latest stretch.
STRETCH_KEY = "autopilot"
STARTED = "autopilot.started"
CONTINUED = "autopilot.continued"
STOPPED = "autopilot.stopped"

#: Why a stretch ended, by the code its record carries.
REASONS: dict[str, str] = {
    "rounds": "the cap on rounds was reached",
    "model_usd": "the cap on model spend was reached",
    "hours": "the cap on hours was reached",
    "limits": "a limit of the project itself was reached",
    "needs_model": "the next phase needs a model that cannot be used",
    "failed": "a phase failed",
    "experiment_failed_twice": "the same experiment failed twice",
    "needs_you": "nothing more can happen without you",
    "stopped": "it was stopped",
    "switched_off": "autopilot was switched off",
    "worker_gone": "its worker ended without saying why",
}

#: Said wherever the spend of a stretch is said.
UNCOUNTED = (
    "What the code of an experiment spends on model calls of its own is not counted by "
    "anything here"
)


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def _plural(count: int, noun: str, many: str | None = None) -> str:
    return f"{count} {noun if count == 1 else many or noun + 's'}"


def _hours(hours: float) -> str:
    """``12 hours``, ``1 hour``, ``30 minutes``: a length of time as it is said."""
    if hours >= 1:
        hours = round(hours, 1)  # "5 hours", not "5.00002 hours"
        return f"{hours:g} hour{'' if hours == 1 else 's'}"
    minutes = hours * 60
    if minutes >= 1:
        return f"{minutes:.0f} minute{'' if round(minutes) == 1 else 's'}"
    return f"{hours * 3600:.0f} seconds"


def caps_in_words(caps: Mapping[str, Any]) -> str:
    """``at most 3 rounds, $200.00 on models or 12 hours, whichever comes first``."""
    rounds = int(_number(caps.get("rounds")))
    return (
        f"at most {_plural(rounds, 'round')}, {_money(_number(caps.get('model_usd')))} on "
        f"models or {_hours(_number(caps.get('hours')))}, whichever comes first"
    )


def new_stretch(
    caps: Mapping[str, Any],
    *,
    started: str,
    spent_usd: float,
    round: int,
    unpriced_usd: float = 0.0,
    experiment_usd: float = 0.0,
) -> dict[str, Any]:
    """The record of a stretch that begins now.

    ``caps`` are ``settings.autonomy`` without the mode, as they were when it began
    (the engine holds the stretch to the settings as they are, so that the person
    can tighten a cap while it runs; this copy is for whoever shows it).
    """
    return {
        "started": started,
        #: The ``seq`` of the event that announced it: the digest counts from there.
        "seq": 0,
        "caps": {name: caps.get(name) for name in ("rounds", "model_usd", "hours", "network")},
        "spent_at_start": round_usd(spent_usd),
        #: What the tokens on models without a price counted for when it began, and
        #: what the project's experiments had cost: the stretch counts from there.
        "unpriced_at_start": round_usd(unpriced_usd),
        "experiment_at_start": round_usd(experiment_usd),
        "round_at_start": round,
        #: Rounds whose analysis completed in this stretch, and passes of the engine.
        "rounds": 0,
        "passes": 0,
        #: What counts against the dollar cap: what the ledger prices, and
        #: ``unpriced_usd`` of it for the tokens of ``unpriced_models``, which have no
        #: price and are counted at the highest list price.
        "spent_usd": 0.0,
        "unpriced_usd": 0.0,
        "unpriced_models": [],
        #: What its experiments cost: no cap of autopilot counts that.
        "experiment_usd": 0.0,
        "ended": None,
        "reason": None,
        "said": None,
    }


def round_usd(amount: float) -> float:
    return round(float(amount), 4)


def read_stretch(value: Any) -> dict[str, Any] | None:
    """The record of a stretch as ``state.json`` holds it, every value of the type the
    engine and whoever shows it count on; ``None`` when there is none."""
    if not isinstance(value, Mapping) or not isinstance(value.get("started"), str):
        return None
    caps = value.get("caps") if isinstance(value.get("caps"), Mapping) else {}
    ended, reason, said = value.get("ended"), value.get("reason"), value.get("said")
    models = value.get("unpriced_models")
    return {
        "started": value["started"],
        "seq": int(_number(value.get("seq"))),
        "caps": {
            "rounds": int(_number(caps.get("rounds"))),
            "model_usd": _number(caps.get("model_usd")),
            "hours": _number(caps.get("hours")),
            "network": caps.get("network") is True,
        },
        "spent_at_start": _number(value.get("spent_at_start")),
        "unpriced_at_start": _number(value.get("unpriced_at_start")),
        "experiment_at_start": _number(value.get("experiment_at_start")),
        "round_at_start": int(_number(value.get("round_at_start"))),
        "rounds": int(_number(value.get("rounds"))),
        "passes": int(_number(value.get("passes"))),
        "spent_usd": _number(value.get("spent_usd")),
        "unpriced_usd": _number(value.get("unpriced_usd")),
        "unpriced_models": [str(m) for m in models] if isinstance(models, list) else [],
        "experiment_usd": _number(value.get("experiment_usd")),
        "ended": ended if isinstance(ended, str) else None,
        "reason": reason if isinstance(reason, str) else None,
        "said": said if isinstance(said, str) else None,
    }


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------


def _data(event: Mapping[str, Any]) -> Mapping[str, Any]:
    data = event.get("data")
    return data if isinstance(data, Mapping) else {}


def _seconds_between(start: str | None, end: str | None) -> float:
    first, last = parse_stamp(start), parse_stamp(end)
    if first is None or last is None:
        return 0.0
    return max(0.0, (last - first).total_seconds())


def digest(
    project: Project, state: Mapping[str, Any] | None = None, *, now: str | None = None
) -> dict[str, Any] | None:
    """What happened in the project's latest unattended stretch; ``None`` when it never
    had one.  ``state`` is the engine's state (read from ``state.json`` without it)::

        {"started", "ended", "running": bool, "reason", "why", "said",
         "rounds", "caps", "spent_usd", "unpriced_usd", "unpriced_models",
         "experiment_usd", "experiment_limit": {"spent", "allowed"}, "seconds",
         "accepted": [{"path", "kind", "id"}], "accepted_by_kind": {kind: n},
         "edits": [{"proposal", "path"}],
         "decisions": [{"decision", "page"}],
         "runs": [{"id", "path", "network", "outcome"}], "runs_with_network": n,
         "held": [{"id", "reason"}], "uncounted": sentence}

    ``accepted`` are the pages the standing rule accepted, ``edits`` the edit
    proposals it applied, ``decisions`` those it left at their default, ``runs`` the
    experiments it approved, each with how its run ended (``outcome``: the state the
    experiment reached since, or ``None`` when it has not run).  ``held`` is what
    the rule would not approve as the project stands, with the reason: what truly
    waits for the person.  Everything but ``held`` is counted from the project's
    events since the stretch began; nothing is written.

    ``spent_usd`` is what counts against the dollar cap; ``unpriced_usd`` of it stands
    for the tokens of ``unpriced_models``, which have no price here and are counted
    at the highest list price there is.  ``experiment_usd`` is what the experiments
    of the stretch cost, which no cap of autopilot counts: ``experiment_limit`` is
    what does, the project's own limit and what is spent of it in all.
    """
    if state is None:
        from paper_orchestra.research.engine import read_state

        state = read_state(project)
    stretch = read_stretch(state.get(STRETCH_KEY))
    if stretch is None:
        return None
    events = project.events(after=stretch["seq"]) if stretch["seq"] else []
    accepted: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    defaulted: list[dict[str, Any]] = []
    runs: dict[str, dict[str, Any]] = {}
    for event in events:
        kind, data = event.get("type"), _data(event)
        if kind == "page.created" and data.get("decided_by") == AUTOPILOT:
            accepted.append(
                {"path": data.get("path"), "kind": data.get("kind"), "id": data.get("id")}
            )
        elif kind == "review.decided" and data.get("by") == AUTOPILOT:
            if data.get("target") == "proposal":
                edits.append({"proposal": data.get("proposal"), "path": data.get("path")})
            elif data.get("decision") == "accept":
                accepted.append(
                    {"path": data.get("path"), "kind": data.get("kind"), "id": data.get("id")}
                )
        elif (
            kind == "decision.asked" and data.get("state") == "defaulted"
        ) or kind == "decision.defaulted":
            defaulted.append({"decision": data.get("decision"), "page": data.get("page")})
        elif kind == "experiment.state" and data.get("id"):
            name = str(data["id"])
            if data.get("by") == AUTOPILOT and data.get("state") == "approved":
                runs[name] = {
                    "id": name,
                    "path": data.get("path"),
                    "network": False,
                    "outcome": None,
                }
            elif name in runs and data.get("state") in ("succeeded", "failed", "cancelled"):
                runs[name]["outcome"] = data.get("state")
            elif name in runs and data.get("state") == "planned":
                runs[name]["outcome"] = runs[name]["outcome"] or "not run"
    by_kind: dict[str, int] = {}
    for item in accepted:
        by_kind[str(item.get("kind"))] = by_kind.get(str(item.get("kind")), 0) + 1
    for run in runs.values():
        try:
            run["network"] = project.page(str(run["path"])).meta.get("network") is True
        except Exception:  # the page is gone; nothing says it had the network
            run["network"] = False
    ended = stretch["ended"]
    spent = state.get("spent") if isinstance(state.get("spent"), Mapping) else {}
    spent_usd = stretch["spent_usd"]
    experiment_usd = stretch["experiment_usd"]
    if not ended:
        spent_usd = max(spent_usd, _number(spent.get("model_usd")) - stretch["spent_at_start"])
        experiment_usd = max(
            experiment_usd, _number(spent.get("experiment_usd")) - stretch["experiment_at_start"]
        )
    return {
        "started": stretch["started"],
        "ended": ended,
        "running": ended is None,
        "reason": stretch["reason"],
        "why": REASONS.get(str(stretch["reason"]), "") if stretch["reason"] else "",
        "said": stretch["said"],
        "rounds": stretch["rounds"],
        "caps": stretch["caps"],
        "spent_usd": round(max(0.0, spent_usd), 4),
        "unpriced_usd": stretch["unpriced_usd"],
        "unpriced_models": stretch["unpriced_models"],
        "experiment_usd": round(max(0.0, experiment_usd), 4),
        "experiment_limit": {
            "spent": _number(spent.get("experiment_usd")),
            "allowed": _number(project.settings.limits.experiment_usd),
        },
        "seconds": round(_seconds_between(stretch["started"], ended or now), 1),
        "accepted": accepted,
        "accepted_by_kind": by_kind,
        "edits": edits,
        "decisions": defaulted,
        "runs": list(runs.values()),
        "runs_with_network": sum(1 for run in runs.values() if run["network"]),
        "held": held(project),
        "uncounted": UNCOUNTED,
    }


def held(project: Project) -> list[dict[str, str]]:
    """The accepted experiments the standing rule does not approve as the project
    stands, each with the reason: what waits for the person although autopilot is on.
    Empty where it is off."""
    if not project.settings.autonomy.autopilot:
        return []
    from paper_orchestra.research import experiments
    from paper_orchestra.research.model import ProjectError

    found = []
    for page in project.pages(kind="experiment", review="accepted", state="planned"):
        try:
            reason = experiments.rule_holds(project, page.path)
        except (ProjectError, OSError, ValueError):
            continue
        if reason:
            found.append({"id": str(page.id or page.path), "reason": reason})
    return found


def summary(found: Mapping[str, Any]) -> str:
    """The digest in plain sentences: what was accepted, applied, defaulted, approved and
    spent, and what waits.  (Why the stretch ended is ``said``, which goes before it.)"""
    parts: list[str] = []
    kinds = found.get("accepted_by_kind") or {}
    if kinds:
        listed = ", ".join(
            _plural(count, kind, SECTIONS.get(kind)) for kind, count in sorted(kinds.items())
        )
        parts.append(f"{_plural(len(found['accepted']), 'page')} accepted ({listed})")
    if found.get("edits"):
        parts.append(f"{_plural(len(found['edits']), 'edit')} applied")
    if found.get("decisions"):
        count = len(found["decisions"])
        parts.append(
            f"{_plural(count, 'decision')} left at {'its' if count == 1 else 'their'} default"
        )
    runs = found.get("runs") or []
    if runs:
        how = ", ".join(f"{run['id']} {run['outcome'] or 'has not run'}" for run in runs)
        parts.append(f"{_plural(len(runs), 'run')} approved ({how})")
    said = [f"Without you: {'; '.join(parts)}." if parts else "Nothing was decided without you."]
    caps = found.get("caps") or {}
    said.append(
        f"{_money(_number(found.get('spent_usd')))} of {_money(_number(caps.get('model_usd')))} "
        f"went to the agents' models in {_hours(_number(found.get('seconds')) / 3600)}, over "
        f"{_plural(int(_number(found.get('rounds'))), 'completed round')}."
    )
    unpriced, models = _number(found.get("unpriced_usd")), found.get("unpriced_models") or []
    if unpriced and models:
        said.append(
            f"Of that, {_money(unpriced)} is not a cost but a count: no price is known for "
            f"{_listed(models)}, so against the cap {'its' if len(models) == 1 else 'their'} "
            "tokens were counted at the highest list price there is, and what they really "
            "cost is not known here."
        )
    limit = found.get("experiment_limit") or {}
    if runs or _number(found.get("experiment_usd")):
        said.append(
            f"{_money(_number(found.get('experiment_usd')))} went to the experiments of this "
            "stretch, which no cap of autopilot counts: the project's own limit holds it "
            f"({_money(_number(limit.get('spent')))} of {_money(_number(limit.get('allowed')))} "
            "spent in all, settings.limits.experiment_usd)."
        )
    networked = int(_number(found.get("runs_with_network")))
    said.append(
        f"{UNCOUNTED}; "
        + (
            f"{_plural(networked, 'run')} had network access and could make such calls."
            if networked
            else "no run of this stretch had network access."
        )
    )
    waiting = found.get("held") or []
    if waiting:
        said.append(
            f"Waiting for you: {grouped([(item['id'], item['reason']) for item in waiting])}."
        )
    return " ".join(said)


def _listed(names: Sequence[str]) -> str:
    names = [str(name) for name in names]
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else "".join(names)


def grouped(held: Sequence[tuple[str, str]], limit: int = 6) -> str:
    """``E1 and E2: <reason>; E3: <another>``: the experiments autopilot left, each reason
    said once for all that share it, and of more than ``limit`` the number."""
    by_reason: dict[str, list[str]] = {}
    for name, why in held[:limit]:
        by_reason.setdefault(why, []).append(name)
    parts = [
        f"{', '.join(names[:-1]) + ' and ' + names[-1] if len(names) > 1 else names[0]}: {why}"
        for why, names in by_reason.items()
    ]
    more = len(held) - limit
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
