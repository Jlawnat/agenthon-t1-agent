from __future__ import annotations

from agent.offline_prediction_markets_dislocation import (
    PredictionMarketCrossVenueDislocationSkill,
)


def test_prediction_markets_dislocation_skill_name() -> None:
    assert (
        PredictionMarketCrossVenueDislocationSkill().name
        == "prediction-markets-cross-venue-dislocation-domain"
    )
