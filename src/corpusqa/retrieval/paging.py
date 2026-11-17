"""Pages of a list: every list a tool returns comes this way, with the next offset, so the
model can reach every item (user decision 2026-10-16: nothing is cut). The total is exact,
except on a search page whose ranking has not reached every chunk in scope yet: such a
page says only where the next page starts (a later page searches deeper).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from corpusqa.retrieval.errors import ToolInputError


@dataclass(frozen=True, slots=True)
class Page[T]:
    """Items offset..offset+len(items) of a list of `total` items. With `exact` False,
    `total` counts only the items found so far and more follow: there is a next page."""

    items: tuple[T, ...]
    offset: int
    total: int
    exact: bool = True

    @property
    def next_offset(self) -> int | None:
        """Where the next page starts; None on the last page."""
        end = self.offset + len(self.items)
        return end if end < self.total else None


def page_of[T](items: Sequence[T], offset: int, size: int, *, exact: bool = True) -> Page[T]:
    """The page of `items` that starts at `offset` (`exact`: `items` is the whole list). An
    offset past the end is the model's mistake: ToolInputError says where the list ends."""
    if offset < 0:
        raise ToolInputError(f"offset must be 0 or more, not {offset}.")
    if offset and offset >= len(items):
        raise ToolInputError(
            f"offset={offset} is past the end: there are {len(items)} in all "
            f"(offsets 0-{max(len(items) - 1, 0)})."
        )
    return Page(
        items=tuple(items[offset : offset + size]), offset=offset, total=len(items), exact=exact
    )
