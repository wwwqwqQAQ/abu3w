"""
Sentiment analysis using Claude API.

Converts raw news text into structured sentiment scores per stock.
Reuses the API key loading pattern from server.py.
"""
import json
import re
import time
import hashlib
from datetime import datetime


# In-memory cache: code -> {score, timestamp}
_SENTIMENT_CACHE = {}
_SENTIMENT_CACHE_TTL = 30 * 60  # 30 minutes


SENTIMENT_PROMPT = """你是一个A股市场情绪分析专家。下面是一些关于股票 {stock_name}({code}) 的最新新闻。

## 新闻列表
{news_text}

## 任务

分析上述新闻对该股票的总体情绪影响，输出JSON格式（只输出JSON）:

{{{{
  "score": 0.0,
  "confidence": 0.0,
  "direction": "bullish|neutral|bearish",
  "keywords": [],
  "summary": "一句话总结",
  "risk_flags": []
}}}}

规则:
- score: -1.0(极度利空) 到 +1.0(极度利好), 0=中性
- confidence: 0.0 到 1.0, 表示判断的把握
- direction: "bullish"(利好), "neutral"(中性), "bearish"(利空)
- keywords: 3-5个关键主题词
- summary: 不超过50字的中文总结
- risk_flags: 具体风险标签,如 "政策风险" "行业利空" "业绩低于预期" "股东减持" "利好出尽" "情绪底背离"

如果新闻不足或信息不足,confidence应低于0.3。
"""


def load_api_key():
    """Load Anthropic API key (reuses server.py pattern)."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    support_dir = os.path.expanduser("~/Library/Application Support/QuantDesk")
    paths = [
        os.path.join(support_dir, "config.json"),
        os.path.expanduser("~/.quantdesk.json"),
        os.path.join(support_dir, ".env"),
        os.path.join(support_dir, "anthropic_api_key.txt"),
        os.path.expanduser("~/.quantdesk.env"),
    ]
    for path in paths:
        try:
            if not os.path.exists(path):
                continue
            if path.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
                key = str(data.get("ANTHROPIC_API_KEY") or data.get("anthropic_api_key") or "").strip()
            else:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        name, value = line.split("=", 1)
                        if name.strip() == "ANTHROPIC_API_KEY":
                            key = value.strip().strip("\"'")
                            break
            if key:
                return key
        except Exception:
            continue
    return ""


def _news_cache_key(code, news_list):
    parts = []
    for item in (news_list or [])[:8]:
        parts.append(f"{item.get('time','')}|{item.get('source','')}|{item.get('title','')}")
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{code}:{digest}"


def _news_quality_meta(news_list):
    items = news_list or []
    fresh_count = sum(1 for item in items if item.get("fresh"))
    material_count = sum(1 for item in items if item.get("material"))
    if fresh_count >= 3:
        quality = "high"
    elif fresh_count >= 1:
        quality = "medium"
    elif items:
        quality = "low"
    else:
        quality = "none"
    return {
        "news_quality": quality,
        "fresh_count": fresh_count,
        "material_count": material_count,
        "news_count": len(items),
    }


def analyze_sentiment(news_list, code, stock_name=None):
    """
    Analyze news sentiment for a stock using Claude.

    Args:
        news_list: list of {title, content, source, time} dicts
        code: stock code
        stock_name: optional name

    Returns:
        dict with {score, confidence, direction, keywords, summary, risk_flags, analyzed_at}
    """
    name = stock_name or code

    # Check cache
    cache_key = _news_cache_key(code, news_list)
    cache_entry = _SENTIMENT_CACHE.get(cache_key)
    if cache_entry and time.time() - cache_entry["ts"] < _SENTIMENT_CACHE_TTL:
        return cache_entry["data"]

    if not news_list:
        return {
            "score": 0.0, "confidence": 0.0, "direction": "neutral",
            "keywords": [], "summary": "无新闻数据",
            "risk_flags": [], "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **_news_quality_meta(news_list),
        }

    # Format news for prompt
    news_text = ""
    for i, item in enumerate(news_list[:6]):
        news_text += f"新闻{i+1}: [{item.get('source','')}] {item.get('title','')}\n"
        content = item.get("content", "")[:300]
        if content and content != item.get("title", ""):
            news_text += f"  摘要: {content}\n"
        news_text += f"  时间: {item.get('time','')}\n\n"

    prompt = SENTIMENT_PROMPT.format(
        stock_name=name,
        code=code,
        news_text=news_text,
    )

    result = _call_claude_analysis(prompt, news_list, name)

    # Cache it
    result.update(_news_quality_meta(news_list))
    _SENTIMENT_CACHE[cache_key] = {"ts": time.time(), "data": result}
    return result


def _call_claude_analysis(prompt, news_list, stock_name):
    """Call Claude API for sentiment analysis."""
    api_key = load_api_key()

    if not api_key:
        # Fallback: rule-based sentiment from keywords
        return _rule_based_sentiment(news_list, stock_name)

    import requests

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "score": round(float(data.get("score", 0)), 3),
                "confidence": round(float(data.get("confidence", 0)), 3),
                "direction": data.get("direction", "neutral"),
                "keywords": data.get("keywords", []),
                "summary": data.get("summary", ""),
                "risk_flags": data.get("risk_flags", []),
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "_fallback": False,
            }
    except Exception:
        pass

    return _rule_based_sentiment(news_list, stock_name)


def _rule_based_sentiment(news_list, stock_name):
    """
    Fallback: weighted keyword sentiment when Claude is unavailable.
    """
    bullish_weights = {
        "超预期": 3, "扭亏": 3, "中标": 2, "签约": 2, "回购": 2, "增持": 2,
        "分红": 2, "涨停": 2, "获批": 2, "订单": 1.5, "增长": 1.5, "利好": 1.5,
        "突破": 1, "补贴": 1, "扩产": 1, "新高": 1, "创新高": 1,
        "持续向好": 2, "未来可期": 1.5, "稳进": 1, "释放强信号": 1.5,
        "改革": 1, "大宗交易成交": 0.8,
    }
    bearish_weights = {
        "退市": 4, "暴雷": 4, "造假": 4, "立案": 3, "调查": 3, "处罚": 3,
        "亏损": 3, "业绩下滑": 3, "减持": 2.5, "跌停": 2.5, "违约": 2.5,
        "诉讼": 2, "监管函": 2, "问询函": 2, "风险提示": 2, "爆仓": 2,
        "下滑": 1.5, "下降": 1.5, "利空": 1.5, "终止": 1.5, "延期": 1,
        "侵权": 2, "致歉": 1.5, "违规": 2, "声誉": 1.5, "损害": 1.5,
        "管控短板": 2, "乱象": 2, "招商乱象": 2,
    }
    risk_map = {
        "政策风险": ("监管", "政策", "处罚", "问询函", "监管函", "调查", "立案"),
        "业绩低于预期": ("亏损", "业绩下滑", "低于预期", "下降", "下滑"),
        "股东减持": ("减持", "清仓式减持"),
        "退市风险": ("退市", "ST", "*ST"),
        "诉讼风险": ("诉讼", "仲裁", "纠纷"),
        "信用风险": ("违约", "爆仓", "债务逾期"),
        "品牌/渠道风险": ("侵权", "声誉", "违规", "管控短板", "乱象", "招商乱象"),
    }

    all_text = " ".join(item.get("title", "") + " " + item.get("content", "")
                        for item in (news_list or []))

    bull_score = sum(weight for kw, weight in bullish_weights.items() if kw in all_text)
    bear_score = sum(weight for kw, weight in bearish_weights.items() if kw in all_text)
    raw = bull_score - bear_score

    if raw >= 2:
        score = min(0.75, 0.18 + raw * 0.08)
        direction = "bullish"
    elif raw <= -2:
        score = max(-0.75, -0.18 + raw * 0.08)
        direction = "bearish"
    else:
        score = 0.0
        direction = "neutral"

    keywords = []
    for kw in list(bullish_weights) + list(bearish_weights):
        if kw in all_text:
            keywords.append(kw)
    risk_flags = [
        label for label, kws in risk_map.items()
        if any(kw in all_text for kw in kws)
    ]
    hit_weight = bull_score + bear_score
    quality = _news_quality_meta(news_list)
    quality_mult = {"high": 1.0, "medium": 0.8, "low": 0.45, "none": 0.0}.get(quality["news_quality"], 0.5)
    confidence = min(0.7, (0.18 + hit_weight * 0.08) * quality_mult)

    return {
        "score": round(score, 3),
        "confidence": round(confidence, 3),
        "direction": direction,
        "keywords": keywords[:5],
        "summary": f"关键词规则分析: 利好权重{bull_score:.1f}, 利空权重{bear_score:.1f}",
        "risk_flags": risk_flags,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_fallback": True,
        "_fallback_reason": "anthropic_api_unavailable_or_failed",
    }
