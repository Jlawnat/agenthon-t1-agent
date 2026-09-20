from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path


def _find_fx_data_dir(task_dir: Path) -> Path:
    required = {"spot_rates.csv", "deposit_rates.csv", "portfolio.csv", "historical_spots.csv", "params.json"}
    for directory in sorted({p.parent for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts}, key=str):
        names = {p.name for p in directory.iterdir() if p.is_file()}
        if required.issubset(names):
            return directory
    raise RuntimeError("Could not discover FX forward portfolio inputs.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _day_count_fraction(days: int, convention: str) -> float:
    if convention == "ACT/360":
        return days / 360.0
    if convention == "ACT/365":
        return days / 365.0
    raise RuntimeError(f"Unknown day-count convention: {convention}")


def _solve(task_dir: Path, out_dir: Path) -> None:
    data_dir = _find_fx_data_dir(task_dir)

    spots: dict[str, dict[str, str]] = {}
    for row in _read_csv(data_dir / "spot_rates.csv"):
        if float(row["bid"]) < float(row["ask"]):
            spots[row["pair"]] = row

    deposits = {row["currency"]: row for row in _read_csv(data_dir / "deposit_rates.csv")}
    portfolio = _read_csv(data_dir / "portfolio.csv")
    hist_spots = {row["pair"]: float(row["mid"]) for row in _read_csv(data_dir / "historical_spots.csv")}
    params = json.loads((data_dir / "params.json").read_text(encoding="utf-8"))
    tenor_days = {str(k): int(v) for k, v in params["tenor_days"].items()}
    pip_multipliers = {str(k): int(v) for k, v in params["pip_multipliers"].items()}

    pair_info = {}
    for pair in spots:
        base, quote = pair.split("/")
        pair_info[pair] = {"base": base, "quote": quote}

    forward_rates: dict[str, dict[str, dict[str, float]]] = {}
    forward_points: dict[str, dict[str, dict[str, float]]] = {}

    for pair, info in pair_info.items():
        base, quote = info["base"], info["quote"]
        s_bid, s_ask = float(spots[pair]["bid"]), float(spots[pair]["ask"])
        s_mid = (s_bid + s_ask) / 2.0
        pair_fwd, pair_pts = {}, {}

        for tenor, days in tenor_days.items():
            base_row, quote_row = deposits[base], deposits[quote]
            tb = _day_count_fraction(days, base_row["day_count"])
            tq = _day_count_fraction(days, quote_row["day_count"])
            rb = float(base_row[f"tenor_{tenor.lower()}"]) / 100.0
            rq = float(quote_row[f"tenor_{tenor.lower()}"]) / 100.0
            ratio = (1.0 + rq * tq) / (1.0 + rb * tb)
            bid, ask, mid = s_bid * ratio, s_ask * ratio, s_mid * ratio
            pair_fwd[tenor] = {"bid": round(bid, 6), "ask": round(ask, 6), "mid": round(mid, 6)}
            pm = pip_multipliers[pair]
            pair_pts[tenor] = {
                "bid": round((bid - s_bid) * pm, 2),
                "ask": round((ask - s_ask) * pm, 2),
                "mid": round((mid - s_mid) * pm, 2),
            }

        forward_rates[pair] = pair_fwd
        forward_points[pair] = pair_pts

    def usd_quote_from_quotes(quotes: dict[str, dict[str, float | str]], ccy: str) -> dict[str, float]:
        if ccy == "USD":
            return {"bid": 1.0, "ask": 1.0, "mid": 1.0}
        direct = f"{ccy}/USD"
        inverse = f"USD/{ccy}"
        if direct in quotes:
            row = quotes[direct]
            return {"bid": float(row["bid"]), "ask": float(row["ask"]), "mid": float(row["mid"])}
        if inverse in quotes:
            row = quotes[inverse]
            bid, ask, mid = float(row["bid"]), float(row["ask"]), float(row["mid"])
            return {"bid": 1.0 / ask, "ask": 1.0 / bid, "mid": 1.0 / mid}
        raise RuntimeError(f"No USD conversion leg for {ccy}.")

    spot_quote_map = {
        pair: {
            "bid": float(row["bid"]),
            "ask": float(row["ask"]),
            "mid": (float(row["bid"]) + float(row["ask"])) / 2.0,
        }
        for pair, row in spots.items()
    }

    def synth_cross(pair: str, quote_map: dict[str, dict[str, float | str]]) -> dict[str, float]:
        base, quote = pair.split("/")
        b = usd_quote_from_quotes(quote_map, base)
        q = usd_quote_from_quotes(quote_map, quote)
        return {
            "bid": b["bid"] / q["ask"],
            "ask": b["ask"] / q["bid"],
            "mid": (b["bid"] / q["ask"] + b["ask"] / q["bid"]) / 2.0,
        }

    cross_rates = {}
    for pair in params["cross_rates_to_compute"]:
        values = synth_cross(pair, spot_quote_map)
        cross_rates[pair] = {key: round(value, 4) for key, value in values.items()}

    fwd_quote_map_3m = {pair: values["3M"] for pair, values in forward_rates.items()}
    forward_cross_rates = {}
    for pair in params["cross_rates_to_compute"]:
        values = synth_cross(pair, fwd_quote_map_3m)
        forward_cross_rates[pair] = {
            "3M": {key: round(value, 4) for key, value in values.items()}
        }

    cip_deviations = {}
    for pair, info in pair_info.items():
        s_mid = spot_quote_map[pair]["mid"]
        f_mid = forward_rates[pair]["3M"]["mid"]
        days = tenor_days["3M"]
        tb = _day_count_fraction(days, deposits[info["base"]]["day_count"])
        tq = _day_count_fraction(days, deposits[info["quote"]]["day_count"])
        rb = float(deposits[info["base"]]["tenor_3m"]) / 100.0
        rq = float(deposits[info["quote"]]["tenor_3m"]) / 100.0
        implied_rq = ((f_mid / s_mid) * (1.0 + rb * tb) - 1.0) / tq
        cip_deviations[pair] = round((implied_rq - rq) * 10000.0, 4)

    def get_spot_mid(pair: str) -> float:
        if pair in spot_quote_map:
            return float(spot_quote_map[pair]["mid"])
        if pair in cross_rates:
            return float(cross_rates[pair]["mid"])
        return float(synth_cross(pair, spot_quote_map)["mid"])

    def fwd_quote_map(tenor: str) -> dict[str, dict[str, float | str]]:
        return {pair: values[tenor] for pair, values in forward_rates.items()}

    def get_forward_mid(pair: str, tenor: str) -> float:
        if pair in forward_rates:
            return float(forward_rates[pair][tenor]["mid"])
        if tenor == "3M" and pair in forward_cross_rates:
            return float(forward_cross_rates[pair]["3M"]["mid"])
        return float(synth_cross(pair, fwd_quote_map(tenor))["mid"])

    def to_usd(amount: float, ccy: str) -> float:
        if ccy == "USD":
            return amount
        return amount * usd_quote_from_quotes(spot_quote_map, ccy)["mid"]

    hist_quote_map: dict[str, dict[str, float | str]] = {
        pair: {"bid": mid, "ask": mid, "mid": mid} for pair, mid in hist_spots.items()
    }

    portfolio_mtm = {}
    pnl_attribution = {}
    currency_deltas: dict[str, float] = {}

    for trade in portfolio:
        tid, pair, tenor = trade["trade_id"], trade["pair"], trade["tenor"]
        direction = trade["direction"]
        notional = float(trade["notional_base"])
        trade_rate = float(trade["trade_rate"])
        base_ccy, quote_ccy = pair.split("/")
        sign = 1.0 if direction == "buy_base" else -1.0

        current_fwd = get_forward_mid(pair, tenor)
        pnl_quote = notional * (current_fwd - trade_rate) * sign
        pnl_usd = to_usd(pnl_quote, quote_ccy)

        portfolio_mtm[tid] = {
            "current_forward_mid": round(current_fwd, 6),
            "pnl_quote_ccy": round(pnl_quote, 2),
            "pnl_usd": round(pnl_usd, 2),
        }

        now_spot = get_spot_mid(pair)
        if pair in hist_spots:
            hist_spot = hist_spots[pair]
        else:
            hist_spot = synth_cross(pair, hist_quote_map)["mid"]

        spot_delta_usd = to_usd(notional * (now_spot - hist_spot) * sign, quote_ccy)
        pm = pip_multipliers.get(pair, 10000)
        carry_points = round((current_fwd - now_spot) * pm, 2)
        carry_usd = to_usd(notional * (carry_points / pm) * sign, quote_ccy)
        residual = pnl_usd - spot_delta_usd - carry_usd

        pnl_attribution[tid] = {
            "spot_delta_usd": round(spot_delta_usd, 2),
            "carry_usd": round(carry_usd, 2),
            "residual_usd": round(residual, 2),
            "total_usd": round(pnl_usd, 2),
        }

        base_usd = to_usd(notional * sign, base_ccy)
        quote_usd = to_usd(-notional * current_fwd * sign, quote_ccy)
        currency_deltas[base_ccy] = currency_deltas.get(base_ccy, 0.0) + base_usd
        currency_deltas[quote_ccy] = currency_deltas.get(quote_ccy, 0.0) + quote_usd

    results = {
        "forward_rates": forward_rates,
        "forward_points": forward_points,
        "cross_rates": cross_rates,
        "forward_cross_rates_3M": forward_cross_rates,
        "cip_deviations_bps": cip_deviations,
        "portfolio_mtm": portfolio_mtm,
        "pnl_attribution": pnl_attribution,
        "net_delta_usd": {key: round(value, 2) for key, value in sorted(currency_deltas.items())},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FxForwardCrossRateSkill:
    name: str = "fx-forward-cross-rate-domain"

    def matches(self, *, instruction: str, task_dir: Path) -> bool:
        lowered = instruction.lower()
        file_names = {p.name for p in task_dir.rglob("*") if p.is_file() and "checks" not in p.parts}
        file_hint = {"spot_rates.csv", "deposit_rates.csv", "portfolio.csv"}.issubset(file_names)
        semantic = "fx" in lowered and "forward" in lowered and ("cross rate" in lowered or "triangulat" in lowered)
        return bool(file_hint or semantic)

    def solve(self, *, instruction: str, task_dir: Path, out_dir: Path, seed: int) -> None:
        del instruction, seed
        _solve(task_dir, out_dir)
