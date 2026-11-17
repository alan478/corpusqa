"""Rules over what the profile stage (S8) learned about each field, shared by ingest (the
chunk header fields) and retrieval (the agent's field catalog). Both may import store.
"""

from corpusqa.store.interfaces import FieldProfileRow


def is_common_field(
    row: FieldProfileRow, *, min_presence: float, max_title_ratio: float, max_content_ratio: float
) -> bool:
    """In at least `min_presence` of its source's documents, and rarely the title or body
    text there (that text is shown or searched already). Picked by the profile only, never
    by the field's name."""
    return (
        row.presence >= min_presence
        and row.title_ratio < max_title_ratio
        and row.content_ratio < max_content_ratio
    )
