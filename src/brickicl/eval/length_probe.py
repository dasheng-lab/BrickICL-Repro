"""Shared-prefix helper for paired short and long target sequences."""


def shared_head(row, prime):
    """Return the first identical held-out parts, represented by two lines each."""
    short = row["target_short"].splitlines(True)
    long = row["target_long"].splitlines(True)
    head = short[: 2 * prime]
    if head != long[: 2 * prime]:
        raise ValueError("Short and long targets differ within the given first parts")
    return "".join(head)
