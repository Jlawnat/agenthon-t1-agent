from __future__ import annotations

from datetime import datetime
from html import unescape
import re


_MONEY = (
    r"(?:\$?[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand)"
    r"|\$[\d,]+(?:\.\d+)?)"
)


def html_to_text(source: str) -> str:
    """Convert filing HTML to compact visible text without external parsers."""
    text = re.sub(r"(?is)<(?:script|style)\b.*?</(?:script|style)>", " ", source)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text).replace("\xa0", " ")).strip()


def extract_sec_event_date(text: str) -> str:
    """Extract and normalize a Date of Report cover-page date."""
    match = re.search(
        r"Date\s+of\s+Report\s*\([^)]*\)\s*:?\s*"
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Date of Report was not found")
    return datetime.strptime(match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")


def extract_sec_item_numbers(text: str) -> tuple[str, ...]:
    """Return unique SEC item numbers in numeric order."""
    values = {match.group(1) for match in re.finditer(r"\bItem\s+(\d+\.\d+)", text, re.I)}
    if not values:
        raise ValueError("no SEC item numbers were found")
    return tuple(sorted(values, key=lambda value: tuple(int(part) for part in value.split("."))))


def money_to_millions(value: str) -> float:
    """Parse a dollar expression and express it in millions."""
    cleaned = value.strip().lower().replace("$", "").replace(",", "")
    units = {"billion": 1000.0, "million": 1.0, "thousand": 0.001}
    for unit, multiplier in units.items():
        if unit in cleaned:
            return float(cleaned.replace(unit, "").strip()) * multiplier
    return float(cleaned) / 1_000_000.0


def revenue_guidance_midpoint_millions(text: str) -> float:
    """Extract a revenue-guidance range or point estimate in millions."""
    range_pattern = re.compile(
        rf"revenue.{{0,180}}?({_MONEY})\s*(?:to|through|[-–—])\s*({_MONEY})",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(text):
        context = text[max(0, match.start() - 250):match.end() + 120].lower()
        if any(term in context for term in ("guidance", "outlook", "forecast", "expect")):
            return (money_to_millions(match.group(1)) + money_to_millions(match.group(2))) / 2.0

    compact = re.compile(
        r"revenue.{0,100}?\$?(\d+(?:\.\d+)?)\s*[-–—]\s*"
        r"\$?(\d+(?:\.\d+)?)\s*(billion|million|thousand)",
        re.IGNORECASE,
    )
    match = compact.search(text)
    if match:
        unit = match.group(3)
        return (money_to_millions(f"{match.group(1)} {unit}") + money_to_millions(f"{match.group(2)} {unit}")) / 2.0

    patterns = (
        rf"revenue.{{0,120}}?(?:expected|forecast(?:ed|ing)?|guidance).{{0,100}}?({_MONEY})",
        rf"(?:forecast(?:ed|ing)?|guidance|outlook).{{0,220}}?revenue.{{0,100}}?({_MONEY})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return money_to_millions(match.group(1))
    raise ValueError("revenue guidance was not found")


def classify_executive_departure(text: str) -> str:
    """Classify explicit CEO or CFO resignation/step-down language."""
    if not re.search(r"\b(?:resign(?:ed|ation)|stepp?ed?\s+down)\b", text, re.I):
        raise ValueError("executive-departure language was not found")
    for role, phrase in (("ceo", "chief executive officer"), ("cfo", "chief financial officer")):
        if re.search(rf"(?:{phrase}.{{0,180}}?(?:resign|stepp?ed?\s+down)|(?:resign|stepp?ed?\s+down).{{0,180}}?{phrase})", text, re.I):
            return role
    raise ValueError("departing role was not identified as CEO or CFO")


def extract_debt_principal_millions(text: str) -> float:
    """Extract initial debt principal, excluding an included overallotment."""
    issue = re.search(
        rf"(?:issued|offering|pricing).{{0,180}}?({_MONEY})\s+(?:aggregate\s+)?principal\s+amount",
        text,
        re.IGNORECASE,
    )
    if issue is None:
        raise ValueError("debt principal amount was not found")
    primary = money_to_millions(issue.group(1))
    additional = re.search(
        rf"(?:up\s+to\s+an?\s+)?additional\s+({_MONEY})\s+(?:aggregate\s+)?principal\s+amount",
        text,
        re.IGNORECASE,
    )
    if additional and re.search(r"include.{0,300}?(?:full\s+exercise|exercise)", text, re.I):
        amount = money_to_millions(additional.group(1))
        if 0.0 < amount < primary:
            primary -= amount
    return float(primary)


def event_alpha_score(base_score: float, severity_multiplier: float, liquidity_multiplier: float) -> float:
    """Compose an event score from independently selected multipliers."""
    return float(base_score) * float(severity_multiplier) * float(liquidity_multiplier)
