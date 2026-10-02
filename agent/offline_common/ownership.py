from __future__ import annotations

from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Form4Transaction:
    transaction_date: str
    security_title: str
    transaction_code: str
    shares: float
    price_per_share: float
    shares_following: float
    ownership_form: str
    ownership_nature: str
    footnote_ids: tuple[str, ...]


def parse_price_range(text: str) -> tuple[float, float, float] | None:
    """Parse a reported execution-price range and its midpoint width in bps."""
    match = re.search(
        r"(?:ranging|range).{0,50}?\$([0-9]+(?:\.[0-9]+)?)\s+"
        r"(?:to|through|-)\s+\$([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    if high < low:
        raise ValueError("price range high is below low")
    midpoint = (low + high) / 2.0
    return low, high, float(10000.0 * (high - low) / midpoint)


def normalize_ownership_entity(
    ownership_form: str,
    nature: str,
    footnote_texts: tuple[str, ...] = (),
) -> str:
    """Resolve a useful ownership bucket from Form 4 ownership metadata."""
    if ownership_form.strip().upper() == "D":
        return "Direct holdings"
    cleaned = re.sub(r"\s+", " ", nature).strip()
    if cleaned and cleaned.lower() not in {"see footnote", "see footnotes"}:
        return cleaned
    joined = " ".join(footnote_texts)
    trustee = re.search(r"Trustee\s+of\s+The\s+(.+?)\s+u/a/d", joined, re.I)
    if trustee:
        return re.sub(r"\s+", " ", trustee.group(1)).strip()
    held = re.search(r"shares\s+held\s+by\s+(?:the\s+)?(.+?)(?:\.|,\s+over|\s+u/a/d)", joined, re.I)
    if held:
        return re.sub(r"\s+", " ", held.group(1)).strip()
    return cleaned or "Indirect holdings"


def parse_form4_transactions(xml_source: str, *, transaction_codes: tuple[str, ...] = ("S",)) -> list[Form4Transaction]:
    """Parse selected non-derivative Form 4 transactions from XML text."""
    root = ET.fromstring(xml_source)

    def value(element: ET.Element, path: str, default: str = "") -> str:
        node = element.find(path)
        if node is None:
            return default
        nested = node.find("value")
        raw = nested.text if nested is not None else node.text
        return raw.strip() if raw else default

    rows: list[Form4Transaction] = []
    for transaction in root.findall(".//nonDerivativeTransaction"):
        code = value(transaction, "transactionCoding/transactionCode")
        if code not in transaction_codes:
            continue
        ids = tuple(
            node.attrib["id"].strip()
            for node in transaction.findall(".//footnoteId")
            if node.attrib.get("id", "").strip()
        )
        rows.append(
            Form4Transaction(
                transaction_date=value(transaction, "transactionDate"),
                security_title=value(transaction, "securityTitle"),
                transaction_code=code,
                shares=float(value(transaction, "transactionAmounts/transactionShares", "0")),
                price_per_share=float(value(transaction, "transactionAmounts/transactionPricePerShare", "0")),
                shares_following=float(value(transaction, "postTransactionAmounts/sharesOwnedFollowingTransaction", "0")),
                ownership_form=value(transaction, "ownershipNature/directOrIndirectOwnership"),
                ownership_nature=value(transaction, "ownershipNature/natureOfOwnership"),
                footnote_ids=tuple(dict.fromkeys(ids)),
            )
        )
    return rows


def sale_pressure_ratios(
    shares_sold: float,
    gross_proceeds: float,
    inventory_before: float,
    adv20_shares: float,
    close_price: float,
) -> dict[str, float]:
    """Compute inventory and liquidity-scaled insider-sale pressure ratios."""
    inputs = (shares_sold, gross_proceeds, inventory_before, adv20_shares, close_price)
    if any(value < 0.0 for value in inputs) or adv20_shares == 0.0 or close_price == 0.0:
        raise ValueError("inputs must be non-negative and liquidity denominators positive")
    return {
        "pct_inventory_sold": float(shares_sold / inventory_before) if inventory_before else 0.0,
        "shares_sold_over_adv20": float(shares_sold / adv20_shares),
        "dollars_sold_over_adv20_dollar_volume": float(gross_proceeds / (adv20_shares * close_price)),
    }
