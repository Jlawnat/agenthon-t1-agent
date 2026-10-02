from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class XbrlFact:
    concept: str
    value: str | float
    context_ref: str
    unit_ref: str | None
    decimals: str | None


def parse_xbrl_facts(source: str | bytes) -> list[XbrlFact]:
    """Parse inline-XBRL and ordinary instance facts from one document."""
    root = ET.fromstring(source)
    rows: list[XbrlFact] = []
    for node in root.iter():
        attributes = {
            key.rsplit("}", 1)[-1].lower(): value
            for key, value in node.attrib.items()
        }
        context = attributes.get("contextref")
        if not context:
            continue
        concept = attributes.get("name") or node.tag.rsplit("}", 1)[-1]
        if not concept:
            continue
        text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
        if not text:
            continue
        unit = attributes.get("unitref")
        decimals = attributes.get("decimals")
        numeric = node.tag.lower().endswith("nonfraction") or unit is not None
        value: str | float = text
        if numeric:
            cleaned = text.replace(",", "").replace("$", "").strip()
            if cleaned in {"-", "—"}:
                cleaned = "0"
            try:
                number = float(cleaned)
                scale = int(attributes.get("scale", "0"))
                sign = -1.0 if attributes.get("sign") == "-" else 1.0
                value = float(sign * number * (10.0 ** scale))
            except ValueError:
                value = text
        rows.append(XbrlFact(str(concept), value, str(context), str(unit) if unit else None, str(decimals) if decimals else None))
    return rows


def load_xbrl_zip_facts(path: str | Path) -> list[XbrlFact]:
    """Collect facts from parseable filing documents inside a ZIP archive."""
    rows: list[XbrlFact] = []
    with zipfile.ZipFile(Path(path)) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".xml", ".xhtml", ".html", ".htm"}:
                continue
            data = archive.read(info)
            if b"contextRef" not in data and b"contextref" not in data:
                continue
            try:
                rows.extend(parse_xbrl_facts(data))
            except ET.ParseError:
                continue
    return rows


def facts_by_concept(facts: list[XbrlFact], concept: str) -> list[XbrlFact]:
    """Select facts by full QName or case-insensitive local concept name."""
    target = concept.lower().split(":")[-1]
    return [fact for fact in facts if fact.concept.lower().split(":")[-1] == target]
