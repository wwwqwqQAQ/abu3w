"""
Probability Scorer — adjusts XGBoost probability with sentiment factors.

Merges technical prediction (XGBoost) with fundamental signal (news sentiment).
"""
from datetime import datetime


def adjusted_probability(prob_xgb, sentiment_score, sentiment_conf, risk_level):
    """
    Compute adjusted outperform probability by blending technical + sentiment signals.

    Args:
        prob_xgb: float 0-1, XGBoost predicted prob of outperforming the benchmark
        sentiment_score: float -1 to +1, from sentiment analysis
        sentiment_conf: float 0-1, confidence of sentiment analysis
        risk_level: str, '低'/'中'/'高'/'极高'

    Returns:
        dict with {prob_raw, prob_adjusted, sentiment_offset, blend_ratio, method}
    """
    # Sentiment offset: how much to shift the probability
    # Scale: -1 → shift -0.12, +1 → shift +0.12
    raw_offset = sentiment_score * 0.12

    # Adjust by confidence: low confidence → less impact
    sentiment_offset = raw_offset * min(1.0, sentiment_conf * 2.0)

    # Risk level modifier
    risk_mult = {"低": 1.0, "中": 0.85, "高": 0.6, "极高": 0.3}[risk_level]

    # Blend: weighted average of XGBoost + sentiment signal
    # Default: 70% XGBoost, 30% sentiment
    blend = 0.7
    sentiment_signal = 0.5 + sentiment_offset  # convert to prob space
    prob_adjusted = prob_xgb * blend + sentiment_signal * (1 - blend)

    # Clamp to [0, 1] with risk multiplier on upward moves
    if prob_adjusted > 0.5:
        prob_adjusted = 0.5 + (prob_adjusted - 0.5) * risk_mult
    prob_adjusted = max(0.01, min(0.99, prob_adjusted))

    # Determine final signal
    if prob_adjusted >= 0.60:
        signal = "强跑赢"
    elif prob_adjusted >= 0.55:
        signal = "小跑赢"
    elif prob_adjusted > 0.45:
        signal = "中性"
    elif prob_adjusted > 0.40:
        signal = "小跑输"
    else:
        signal = "强跑输"

    return {
        "prob_raw": round(prob_xgb, 4),
        "prob_adjusted": round(prob_adjusted, 4),
        "sentiment_offset": round(sentiment_offset, 4),
        "blend_ratio": blend,
        "risk_multiplier": risk_mult,
        "signal": signal,
        "adjusted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
