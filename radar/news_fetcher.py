"""
News fetcher for Chinese A-share stocks.

Sources: 东方财富, 新浪财经, 雪球
Returns structured news items with title, content, source, and timestamp.
"""
import json
import re
import subprocess
import time
from datetime import datetime, timedelta


FRESH_DAYS = 30
MAX_NEWS_AGE_DAYS = 90
MATERIAL_KEYWORDS = (
    "业绩", "预告", "增长", "下滑", "亏损", "扭亏", "利润", "营收", "订单", "合同",
    "中标", "回购", "增持", "减持", "处罚", "调查", "诉讼", "分红", "重组", "并购",
    "停牌", "复牌", "风险", "监管", "评级", "目标价", "研报", "产能", "涨价", "降价",
    "股东大会", "股东会", "大宗交易", "侵权", "声誉", "违规", "招商", "渠道", "控价",
)
LOW_SIGNAL_ANN_KEYWORDS = (
    "集合资产管理计划", "基金合同", "基金份额", "托管协议", "招募说明书", "净值公告",
    "ETF", "LOF", "债券型证券投资基金", "交易型开放式指数证券投资基金",
)


def _curl(url, referer=None, timeout=8):
    """Minimal curl wrapper with browser-like headers."""
    cmd = ["curl", "-sS", "--connect-timeout", "3", "--max-time", str(timeout), "--noproxy", "*"]
    cmd += ["-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"]
    if referer:
        cmd += ["-H", f"Referer: {referer}"]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 2)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _parse_news_time(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts /= 1000
        try:
            return datetime.fromtimestamp(ts)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000)
    if re.fullmatch(r"\d{10}", text):
        return datetime.fromtimestamp(int(text))
    text = text.replace("T", " ").replace("/", "-")
    text = re.sub(r"\s+", " ", text)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text[:19] if "%H" in fmt else text[:10], fmt)
        except Exception:
            continue
    return None


def _clean_news_item(item, code, stock_name):
    title = (item.get("title") or "").strip()
    content = (item.get("content") or "").strip()
    source = (item.get("source") or "").strip() or "未知来源"
    dt = _parse_news_time(item.get("time"))
    now = datetime.now()
    age_days = None if dt is None else (now - dt).days
    text = f"{title} {content}"
    has_identity = code in text or (stock_name and stock_name in text)
    material = any(kw in text for kw in MATERIAL_KEYWORDS)
    low_signal_announcement = source.endswith("公告") and any(kw in text for kw in LOW_SIGNAL_ANN_KEYWORDS)

    if dt and dt > now + timedelta(days=1):
        return None, "future"
    if age_days is not None and age_days > MAX_NEWS_AGE_DAYS:
        return None, "stale"
    if low_signal_announcement and not has_identity and not material:
        return None, "irrelevant"
    if source.endswith("公告") and not has_identity and not material:
        return None, "irrelevant"
    if not source.endswith("公告") and not has_identity:
        return None, "irrelevant"

    cleaned = dict(item)
    cleaned["title"] = title
    cleaned["content"] = content[:500]
    cleaned["source"] = source
    if dt:
        cleaned["time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    cleaned["age_days"] = age_days
    cleaned["fresh"] = bool(age_days is not None and 0 <= age_days <= FRESH_DAYS)
    cleaned["material"] = bool(material)
    return cleaned, "kept"


def fetch_eastmoney_news(code, limit=8):
    """
    Fetch recent news for a stock from 东方财富.

    Args:
        code: 6-digit A-share code (e.g. '600519')
        limit: max news items to return

    Returns:
        list of dicts: [{title, content, source, time}, ...]
    """
    market = "1" if code.startswith(("6", "9")) else "2"
    stock_code = f"{market}{code}"

    url = (
        f"https://np-anotice-stock.eastmoney.com/api/security/ann"
        f"?page_size={limit}&page_index=1&stock_list={stock_code}"
    )
    try:
        resp = _curl(url, referer="https://www.eastmoney.com/")
        if resp:
            data = json.loads(resp)
            items = data.get("data", {}).get("list", [])
            results = []
            for item in items[:limit]:
                results.append({
                    "title": item.get("title", ""),
                    "content": (item.get("summary") or item.get("title", ""))[:500],
                    "source": "东方财富公告",
                    "time": item.get("notice_date", "") or datetime.now().strftime("%Y-%m-%d"),
                })
            return results
    except Exception:
        pass
    return []


def fetch_eastmoney_f10_news(code, limit=8):
    """
    Fetch stock-specific F10 news from 东方财富.
    """
    market = "SH" if code.startswith(("6", "9")) else "SZ"
    url = f"https://emweb.securities.eastmoney.com/PC_HSF10/NewsBulletin/PageAjax?code={market}{code}"
    try:
        resp = _curl(url, referer="https://emweb.securities.eastmoney.com/", timeout=6)
        if resp:
            data = json.loads(resp)
            blocks = []
            for key in ("gszx", "gg"):
                block = data.get(key, {})
                items = block.get("data", {}).get("items", [])
                blocks.extend(items)
            results = []
            for item in blocks[:limit]:
                results.append({
                    "title": item.get("title", ""),
                    "content": (item.get("summary") or item.get("title", ""))[:500],
                    "source": "东方财富F10",
                    "time": item.get("showDateTime") or item.get("publishDate") or item.get("updateTime") or "",
                    "url": item.get("url") or item.get("uniqueUrl") or "",
                })
            return results
    except Exception:
        pass
    return []


def fetch_sina_finance_news(keyword, limit=5):
    """
    Search 新浪财经 news by keyword.

    Args:
        keyword: search term (stock name, code, or topic)
        limit: max news items

    Returns:
        list of dicts
    """
    import urllib.parse
    encoded = urllib.parse.quote(keyword)
    url = f"https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num={limit}&page=1&r=0.5&callback=&coupon=&k={encoded}"
    try:
        resp = _curl(url, referer="https://finance.sina.com.cn/")
        if resp and "{" in resp:
            json_start = resp.index("{")
            data = json.loads(resp[json_start:])
            items = data.get("result", {}).get("data", [])
            results = []
            for item in items[:limit]:
                results.append({
                    "title": item.get("title", ""),
                    "content": (item.get("intro") or item.get("title", ""))[:500],
                    "source": "新浪财经",
                    "time": item.get("ctime", ""),
                })
            return results
    except Exception:
        pass
    return []


def fetch_xueqiu_hot(limit=5):
    """
    Fetch trending topics from 雪球.

    Returns:
        list of dicts with title, content, source='雪球', time
    """
    url = f"https://xueqiu.com/statuses/hots.json?count={limit}"
    try:
        resp = _curl(url, referer="https://xueqiu.com/", timeout=6)
        if resp:
            data = json.loads(resp)
            items = data if isinstance(data, list) else data.get("list", [])
            results = []
            for item in items[:limit]:
                text = item.get("text", "") or item.get("title", "")
                results.append({
                    "title": text[:100],
                    "content": text[:500],
                    "source": "雪球",
                    "time": item.get("created_at", ""),
                })
            return results
    except Exception:
        pass
    return []


def fetch_news_for_code(code, stock_name=None, limit=8):
    """
    Aggregate news from all sources for a given stock.

    Args:
        code: 6-digit A-share code
        stock_name: optional Chinese name for keyword search
        limit: max items per source

    Returns:
        dict with {code, news: [...], fetched_at}
    """
    all_news = []
    source_counts = {}

    f10_news = fetch_eastmoney_f10_news(code, limit=limit)
    all_news.extend(f10_news)

    em_news = fetch_eastmoney_news(code, limit=limit)
    all_news.extend(em_news)

    keyword = stock_name or code
    sina_news = fetch_sina_finance_news(keyword, limit=max(3, limit // 2))
    all_news.extend(sina_news)

    for item in all_news:
        source = item.get("source") or "未知来源"
        source_counts[source] = source_counts.get(source, 0) + 1

    seen_titles = set()
    deduped = []
    filtered = {"future": 0, "stale": 0, "irrelevant": 0, "duplicate": 0}
    for item in all_news:
        item, reason = _clean_news_item(item, code, stock_name or "")
        if reason != "kept":
            filtered[reason] = filtered.get(reason, 0) + 1
            continue
        key = item["title"][:40]
        if key not in seen_titles:
            seen_titles.add(key)
            deduped.append(item)
        else:
            filtered["duplicate"] += 1

    deduped.sort(key=lambda x: x.get("time", ""), reverse=True)
    selected = deduped[:limit]
    fresh_count = sum(1 for item in selected if item.get("fresh"))
    latest_valid_time = selected[0].get("time") if selected else ""
    if fresh_count >= 3:
        quality = "high"
    elif fresh_count >= 1:
        quality = "medium"
    elif selected:
        quality = "low"
    else:
        quality = "none"

    return {
        "code": code,
        "name": stock_name or code,
        "news": selected,
        "total": len(deduped),
        "news_quality": quality,
        "diagnostics": {
            "raw_count": len(all_news),
            "kept_count": len(deduped),
            "fresh_count": fresh_count,
            "source_counts": source_counts,
            "filtered": filtered,
            "latest_valid_time": latest_valid_time,
            "fresh_days": FRESH_DAYS,
            "max_age_days": MAX_NEWS_AGE_DAYS,
        },
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
