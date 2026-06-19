"""
Risk Alerter — combines sentiment + technical signals → risk level.

Risk levels: 低, 中, 高, 极高
Risk types: 利好出尽, 情绪底背离, 政策风险, 行业利空, 情绪过热, 无显著风险
"""
from datetime import datetime


def assess_risk(code, sentiment_data, ml_prediction, kline_features):
    """
    Assess risk level for a stock.

    Args:
        code: stock code
        sentiment_data: dict from sentiment.analyze_sentiment()
        ml_prediction: dict with {prob_up, signal, confidence}
        kline_features: dict with {rsi_14, trend_20, trend_60, volatility_20, ma20_dev, price_position}

    Returns:
        dict with {level, type, reasons, score, assessed_at}
    """
    sentiment_score = sentiment_data.get("score", 0)
    sentiment_conf = sentiment_data.get("confidence", 0)
    risk_flags = sentiment_data.get("risk_flags", [])

    rsi = kline_features.get("rsi_14", 50)
    trend_20 = kline_features.get("trend_20", 0)
    trend_60 = kline_features.get("trend_60", 0)
    vol = kline_features.get("volatility_20", 0.02)
    atr = kline_features.get("atr_14_pct", 0)
    ma20_dev = kline_features.get("ma20_dev", 0)
    ma60_dev = kline_features.get("ma60_dev", 0)
    price_position = kline_features.get("price_position", 0.5)
    prob_up = ml_prediction.get("outperform_prob", ml_prediction.get("prob_up", 0.5))
    class_probs = ml_prediction.get("class_probabilities", {}) or {}
    strong_underperform = class_probs.get("强跑输", 0)
    suggested_position = ml_prediction.get("suggested_position", {}) or {}
    news_quality = sentiment_data.get("news_quality", "unknown")

    reasons = []
    risk_score = 0  # 0-100, higher = more risk

    # 1. Check sentiment-based risks
    if sentiment_score < -0.3 and sentiment_conf > 0.4:
        reasons.append(f"新闻情绪偏空 (sentiment={sentiment_score:.2f})")
        risk_score += 20

    if sentiment_score < -0.5:
        reasons.append("显著利空信号")
        risk_score += 15

    # 2. 利好出尽: positive sentiment but overbought
    if sentiment_score > 0.3 and rsi > 70:
        reasons.append("利好出尽风险: 情绪积极但RSI超买")
        risk_score += 20

    # 3. 情绪底背离: negative sentiment but oversold
    if sentiment_score < -0.2 and rsi < 25:
        reasons.append("情绪底背离: 利空情绪但RSI极度超卖, 可能反弹")
        risk_score -= 10  # reduces risk (opportunity)

    # 4. Technical risk
    if rsi > 80:
        reasons.append("RSI极度超买 (>80)")
        risk_score += 15
    elif rsi > 70:
        reasons.append("RSI超买 (>70)")
        risk_score += 8

    if trend_20 < -10:
        reasons.append(f"短期趋势陡峭下跌 (trend_20={trend_20:.0f}°)")
        risk_score += 12

    if trend_60 < -12:
        reasons.append(f"中期趋势走弱 (trend_60={trend_60:.0f}°)")
        risk_score += 8

    if ma20_dev < -6:
        reasons.append(f"跌破MA20较多 ({ma20_dev:.1f}%)")
        risk_score += 8

    if ma60_dev < -10:
        reasons.append(f"深度跌破MA60 ({ma60_dev:.1f}%)")
        risk_score += 16
    elif ma60_dev < -4:
        reasons.append(f"跌破MA60 ({ma60_dev:.1f}%)")
        risk_score += 9

    if price_position < 0.15 and trend_20 < 0:
        reasons.append("价格处于60日低位且趋势偏弱")
        risk_score += 6

    if vol > 0.06:
        reasons.append(f"极高波动率 ({vol*100:.1f}%)")
        risk_score += 16
    elif vol > 0.05:
        reasons.append(f"高波动率 ({vol*100:.1f}%)")
        risk_score += 8

    if atr >= 8:
        reasons.append(f"ATR极高 ({atr:.1f}%)")
        risk_score += 14
    elif atr >= 5:
        reasons.append(f"ATR偏高 ({atr:.1f}%)")
        risk_score += 8

    # 5. ML disagreement risk
    if prob_up > 0.7 and sentiment_score < -0.3:
        reasons.append("技术面看涨 vs 基本面利空, 方向矛盾")
        risk_score += 10
    elif prob_up < 0.3 and sentiment_score > 0.3:
        reasons.append("技术面看跌 vs 基本面利好, 方向矛盾")
        risk_score += 10

    if prob_up < 0.35:
        reasons.append(f"模型跑赢概率偏低 ({prob_up*100:.0f}%)")
        risk_score += 12
    if prob_up < 0.25:
        reasons.append(f"模型强烈跑输预警 ({prob_up*100:.0f}%)")
        risk_score += 10
    if strong_underperform >= 0.35:
        reasons.append(f"强跑输概率较高 ({strong_underperform*100:.0f}%)")
        risk_score += 10
    if suggested_position.get("label") == "空仓":
        reasons.append("交易规则建议空仓")
        risk_score += 10

    if news_quality in ("none", "low") and sentiment_conf < 0.25:
        reasons.append("新闻新鲜度/相关性不足，舆情判断置信度低")
        risk_score += 5

    # 6. Named risk flags from sentiment analysis
    risk_type = "无显著风险"
    if risk_flags:
        risk_type = risk_flags[0]
        for flag in risk_flags:
            if flag in ("政策风险", "行业利空", "股东减持"):
                risk_score += 20
                risk_type = flag

    if "利好出尽" in risk_flags:
        risk_score += 15
        risk_type = "利好出尽"
    if "情绪底背离" in risk_flags:
        risk_score -= 8
        risk_type = "情绪底背离"

    if risk_type == "无显著风险":
        if prob_up < 0.35 or ma60_dev < -10:
            risk_type = "技术/模型走弱"
        elif news_quality in ("none", "low") and sentiment_conf < 0.25:
            risk_type = "新闻质量不足"

    # 7. Determine level
    if risk_score >= 50:
        level = "极高"
    elif risk_score >= 30:
        level = "高"
    elif risk_score >= 15:
        level = "中"
    else:
        level = "低"

    if not reasons:
        reasons.append("未检测到显著风险信号")

    return {
        "code": code,
        "level": level,
        "type": risk_type,
        "score": max(0, min(100, risk_score)),
        "reasons": reasons,
        "inputs": {
            "news_quality": news_quality,
            "sentiment_score": sentiment_score,
            "sentiment_confidence": sentiment_conf,
            "outperform_prob": prob_up,
            "rsi_14": rsi,
            "ma20_dev": ma20_dev,
            "ma60_dev": ma60_dev,
            "trend_20": trend_20,
            "trend_60": trend_60,
            "volatility_20": vol,
            "atr_14_pct": atr,
        },
        "assessed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
