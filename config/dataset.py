"""Dataset class metadata shared by active project workflows."""

from __future__ import annotations


CLASS_NAMES: tuple[str, ...] = (
    "person",
    "helmet",
    "vest",
    "gloves",
    "goggles",
)

PERSON_CLASS_NAME = "person"

PPE_CLASS_NAMES: tuple[str, ...] = (
    "helmet",
    "vest",
    "gloves",
    "goggles",
)

NUM_CLASSES = len(CLASS_NAMES)

# Compliance evaluates detected people. Only PPE classes are selectable
# requirements in the application menu.
COMPLIANCE_PPE_OPTIONS: tuple[str, ...] = PPE_CLASS_NAMES
