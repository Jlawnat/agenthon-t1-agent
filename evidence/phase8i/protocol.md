# Phase 8I Unseen Validation Protocol

Phase 8I begins from the frozen Phase 8H post-blind recovery 6/6 milestone.

The validation cohort was selected using task directory names only.

Before the Phase 8I blind evaluation:

- selected task instructions were not inspected;
- selected checker implementations were not inspected;
- selected task datasets were not inspected;
- selected reference outputs were not inspected;
- no task-specific modifications were made.

Previously seen or inspected task IDs were excluded conservatively.

The cohort was selected deterministically with Python:

random.Random(2026092008).sample(pool, 6)

The first evaluation of this cohort will be preserved as the Phase 8I
genuine blind result. Any subsequent inspection or modification will be
classified as post-blind analysis/recovery.
