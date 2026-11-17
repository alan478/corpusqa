"""The corpus the agent-tool tests run on: the mini corpus plus an invented source.

`benchwork` is a made-up workshop's repair orders (invented names AND field names) with
what the tools must handle: 80 documents (more than one page of 50), an enum with a
spelling the model may get wrong, a value with two spellings that fold to one loose form,
people written three ways, a list field, numbers, dates (created from `job_opened`), a
field that is present but empty in some documents and absent in others, an own key per
document and references to it (resolved, unresolved, inside a URL). `logbook` has two
entries whose one date field has a date inside the learned window (2024) and one far
outside it (2128): each date is judged on its own, so the first entry has a created date
and the second has none.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from corpusqa.config import CONTENT_KEY, TITLE_KEY, UUID_KEY, Settings
from corpusqa.ingest import Runtime, run
from corpusqa.llm.fakes import FakeEmbedder

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_corpus"
SOURCE = "benchwork"
MIXED_DATES_SOURCE = "logbook"
IN_WINDOW_STAMP, OUTSIDE_STAMP = "1717236000", "5000000000"  # epochs: 2024-06-01, 2128-06-11
N_ORDERS = 80
DIMS = 16
FIRST_OPENED = date(2024, 1, 1)
ITEMS = ("lathe", "kiln", "press", "loom", "bandsaw", "grinder", "drill", "router")
FIRST_NAMES = ("Ottilie", "Ferdy", "Magnus", "Saskia", "Bertil", "Yusra")
SURNAMES = ("Brandvold", "Quillam", "Oyelaran", "Tremlow", "Annaker")
# 30 people: too many for an enum, so the profile learns the field holds people.
PEOPLE = tuple(f"{first} {last}" for first in FIRST_NAMES for last in SURNAMES)
TEAMS = ("bay-north", "bay-south", "bay-east", "bay-west")


def state_of(n: int) -> str:
    return "In Repair" if n <= 55 else "Waiting Parts" if n <= 70 else "Done"


def rush_of(n: int) -> str:
    return "R1" if n % 3 == 0 else "R2"


def lead_of(n: int) -> str:
    """The crew lead; the first person is written three ways (orders 1, 31 and 61)."""
    person = PEOPLE[(n - 1) % len(PEOPLE)]
    if person == PEOPLE[0]:
        spellings = (person, f"{person} (Lead)", person.lower().replace(" ", "."))
        return spellings[(n - 1) // len(PEOPLE)]
    return person


def signer_of(n: int) -> str:
    return PEOPLE[(n + 7) % len(PEOPLE)]


def opened_of(n: int) -> date:
    return FIRST_OPENED + timedelta(days=3 * n)


def cost_of(n: int) -> int:
    return 50 + 10 * (n % 20)


def budget_of(n: int) -> str | None:
    """Two spellings of one band (orders 1-6 and 7-10), one other band, else none."""
    return "10k-20k" if n <= 6 else "$10k-$20k" if n <= 10 else "40k-60k" if n <= 14 else None


def tags_of(n: int) -> list[str]:
    return (
        ["hydraulics", "night-shift"] if n % 5 == 0 else ["hydraulics"] if n % 2 else ["electrics"]
    )


def waiver_of(n: int) -> str | None:
    """ "granted" in orders 10, 20, ...; empty in 11, 21, ...; absent in every other."""
    return "granted" if n % 10 == 0 else "" if n % 10 == 1 else None


def blocked_by_of(n: int) -> str | None:
    """Orders 2-6 refer to order 1 by its key, 7 to a key no order owns, 8 to order 1 by
    a URL, 9 to order 12 (a key that starts with order 1's key)."""
    if 2 <= n <= 6:
        return "WRK-1"
    return {7: "WRK-999", 8: "https://tracker.example/browse/WRK-1", 9: "WRK-12"}.get(n)


def order(n: int) -> dict[str, object]:
    item = ITEMS[n % len(ITEMS)]
    doc: dict[str, object] = {
        "job_label": f"Repair order {n}: the {item} in {TEAMS[n % 4]}",
        "job_notes": f"The {item} needs a new belt and a full service; order number {n}.",
        "job_code": f"WRK-{n}",
        "job_state": state_of(n),
        "job_rush": rush_of(n),
        "crew_lead": lead_of(n),
        "signed_off_by": signer_of(n),
        "bench_team": TEAMS[n % 4],
        "job_tags": tags_of(n),
        "job_cost": cost_of(n),
        "job_opened": opened_of(n).isoformat(),
    }
    if state_of(n) == "Done":
        doc["job_closed"] = (opened_of(n) + timedelta(days=5)).isoformat()
    optional = {"budget_band": budget_of(n), "job_waiver": waiver_of(n)}
    optional["blocked_by"] = blocked_by_of(n)
    doc.update({key: value for key, value in optional.items() if value is not None})
    doc.update({TITLE_KEY: "job_label", CONTENT_KEY: ["job_notes"], UUID_KEY: f"bw-{n:03d}"})
    return doc


def doc_id(n: int) -> str:
    return f"bw-{n:03d}"


def make_tool_corpus(target: Path) -> Path:
    """The mini corpus copied to `target`, plus the `benchwork` source."""
    shutil.copytree(FIXTURE, target)
    folder = target / SOURCE
    folder.mkdir()
    for n in range(1, N_ORDERS + 1):
        (folder / f"order-{n}.json").write_text(json.dumps(order(n)), encoding="utf-8")
    log = target / MIXED_DATES_SOURCE
    log.mkdir()
    for n, stamp in enumerate((IN_WINDOW_STAMP, OUTSIDE_STAMP)):
        entry = {
            "entry_name": f"Tide log {n}",
            "entry_text": f"High water at the north quay, reading {n}.",
            "stamp_at": stamp,
            TITLE_KEY: "entry_name",
            CONTENT_KEY: ["entry_text"],
            UUID_KEY: f"lg-{n}",
        }
        (log / f"tide-{n}.json").write_text(json.dumps(entry), encoding="utf-8")
    return target


@dataclass(frozen=True, slots=True)
class Built:
    runtime: Runtime
    embedder: FakeEmbedder


def build_tool_corpus(tmp: Path) -> Built:
    """Ingest the tool corpus (with fake vectors) into `tmp`."""
    corpus = make_tool_corpus(tmp / "corpus")
    embedder = FakeEmbedder(dims=DIMS)
    settings = Settings(
        corpus_dir=corpus.resolve(),
        port=8080,
        index_dir=tmp / "index",
        embed_cache_db=tmp / "embed-cache.sqlite",
    )
    return Built(runtime=run(settings, embedder), embedder=embedder)
