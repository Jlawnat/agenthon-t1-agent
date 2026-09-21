from __future__ import annotations

from agent.offline_sentiment_factor import SentimentFactorAlphaSkill


def test_sentiment_factor_skill_name() -> None:
    assert SentimentFactorAlphaSkill().name == "sentiment-factor-alpha-domain"
