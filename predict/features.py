"""
Feature engineering for next-day direction prediction.

Takes OHLCV list-of-6-tuples: (date, open, close, high, low, volume)
Returns feature matrix and labels with NO look-ahead bias.

Every feature at bar index `i` uses only data from indices [0..i].
The label at bar `i` uses the close at bar `i+1` (next day).
"""
import math


NUMERIC_FEATURE_KEYS = [
    "rsi_14", "ma20_dev", "ma60_dev", "trend_20", "trend_60",
    "trend_accel", "volatility_20", "vol_ratio", "price_position",
    "ret_1d", "ret_5d", "ret_20d",
    "consec_up", "consec_down", "vol_price_div",
    "rel_strength_20", "rel_strength_5",
    "hl_ratio", "oc_ratio",
    "gap_shock", "abnormal_vol", "extreme_move",
    "vol_accel", "price_vol_corr", "high_vol_streak",
    "log_mv", "pe_ratio", "pb_ratio", "industry_cluster",
    "nb_net_flow", "nb_cum_flow_5d",
    "margin_balance", "margin_buy_ratio", "margin_net", "short_ratio",
    "ma5_dev", "ma10_dev", "trend_10",
    "ret_3d", "ret_10d", "ret_60d",
    "atr_14_pct", "hist_vol_60", "intraday_amp_20",
    "bench_above_ma20", "bench_ret_20", "bench_vol_20", "market_regime",
    "market_breadth", "industry_rel_strength_20", "industry_rank_pct",
    "liquidity_score", "limit_down_risk", "recent_surge",
    # 多周期趋势一致性 + 动量加速度（纯价格数据，永远可用）
    "trend_alignment", "rsi_momentum", "volume_trend", "ma_alignment",
    "bb_position", "turnover_5d",
]
DOW_FEATURE_KEYS = ["dow_0", "dow_1", "dow_2", "dow_3", "dow_4"]
FEATURE_NAMES = NUMERIC_FEATURE_KEYS + DOW_FEATURE_KEYS
LEGACY_FEATURE_NAMES = [
    "rsi_14", "ma20_dev", "ma60_dev", "trend_20", "trend_60",
    "trend_accel", "volatility_20", "vol_ratio", "price_position",
    "ret_1d", "ret_5d", "ret_20d",
    "consec_up", "consec_down", "vol_price_div",
    "rel_strength_20", "rel_strength_5",
    "hl_ratio", "oc_ratio",
    "gap_shock", "abnormal_vol", "extreme_move",
    "vol_accel", "price_vol_corr", "high_vol_streak",
    "log_mv", "pe_ratio", "pb_ratio", "industry_cluster",
    "nb_net_flow", "nb_cum_flow_5d",
    "margin_balance", "margin_buy_ratio", "margin_net", "short_ratio",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4",
]
# 核心特征：只用价格+基准数据，永远有值，不受外部数据影响
CORE_FEATURE_KEYS = [
    "rsi_14", "rsi_momentum",
    "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
    "trend_10", "trend_20", "trend_60", "trend_accel", "trend_alignment",
    "volatility_20", "hist_vol_60", "vol_ratio", "volume_trend", "vol_accel",
    "price_position", "bb_position",
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "consec_up", "consec_down",
    "rel_strength_5", "rel_strength_20",
    "atr_14_pct", "intraday_amp_20",
    "ma_alignment",
    "hl_ratio", "oc_ratio",
    "gap_shock", "abnormal_vol", "extreme_move", "recent_surge",
    "vol_price_div", "price_vol_corr", "high_vol_streak",
    "bench_above_ma20", "bench_ret_20", "bench_vol_20", "market_regime",
    "market_breadth",
    "liquidity_score", "limit_down_risk",
    "turnover_5d",
]
CORE_FEATURE_NAMES = CORE_FEATURE_KEYS + DOW_FEATURE_KEYS

TARGET_LABELS = {
    0: "强跑输",
    1: "小跑输",
    2: "小跑赢",
    3: "强跑赢",
}


def build_features(kl, benchmark_kl=None, lookahead=5, min_bars=60,
                   fundamentals=None, northbound_flow=None, margin_data=None,
                   market_breadth=None, industry_context=None,
                   target_mode="binary_outperform"):
    """
    Build feature vectors and labels from OHLCV data.

    Args:
        kl: list of (date, open, close, high, low, volume) tuples
        benchmark_kl: optional benchmark OHLCV for relative-strength features & label
        lookahead: forward days for label (default 5)
        min_bars: minimum bars before first feature row (default 60)

    Label:
        If benchmark_kl: y = 1 if (stock 5d return > benchmark 5d return) else 0
        Otherwise: y = 1 if close[t+lookahead] > close[t] else 0

    Returns:
        features: list of dicts, one per bar from min_bars to len(kl)-lookahead-1
        labels: list of ints
    """
    min_end = min_bars + lookahead
    if len(kl) < min_end + 1:
        return [], []

    n = len(kl)
    opens = [k[1] for k in kl]
    closes = [k[2] for k in kl]
    highs = [k[3] for k in kl]
    lows = [k[4] for k in kl]
    volumes = [k[5] for k in kl]
    dates = [k[0] for k in kl]

    # Benchmark data — align by DATE, not by index (critical fix)
    bench_by_date = {}
    bench_date_to_idx = {}
    bench_closes_aligned = []
    if benchmark_kl:
        for idx, k in enumerate(benchmark_kl):
            bench_by_date[k[0]] = (k[1], k[2], k[3], k[4], k[5])
            bench_date_to_idx[k[0]] = idx
        # Build benchmark closes aligned to stock dates
        bench_closes_aligned = [bench_by_date[d][1] if d in bench_by_date else None for d in dates]
        bench_closes_raw = [k[2] for k in benchmark_kl]

    # Precompute rolling RSI
    rsi_vals = _compute_rsi_series(closes, period=14)

    features = []
    labels = []

    for i in range(min_bars, n - lookahead):
        feat = {}
        feat["_date"] = dates[i]

        # -- RSI --
        feat["rsi_14"] = round(rsi_vals[i], 2)

        # -- MA deviation --
        ma5 = _rolling_mean(closes, 5, i)
        ma10 = _rolling_mean(closes, 10, i)
        ma20 = _rolling_mean(closes, 20, i)
        ma60 = _rolling_mean(closes, 60, i)
        feat["ma5_dev"] = round((closes[i] - ma5) / ma5 * 100, 2) if ma5 > 0 else 0.0
        feat["ma10_dev"] = round((closes[i] - ma10) / ma10 * 100, 2) if ma10 > 0 else 0.0
        feat["ma20_dev"] = round((closes[i] - ma20) / ma20 * 100, 2) if ma20 > 0 else 0.0
        feat["ma60_dev"] = round((closes[i] - ma60) / ma60 * 100, 2) if ma60 > 0 else 0.0

        # -- Trend angles --
        feat["trend_10"] = round(_trend_angle(closes, 10, i), 2)
        feat["trend_20"] = round(_trend_angle(closes, 20, i), 2)
        feat["trend_60"] = round(_trend_angle(closes, 60, i), 2)

        # -- Trend acceleration (change of trend_20 over 10 bars) --
        if i >= 10:
            trend_20_prev = _trend_angle(closes, 20, i - 10)
            feat["trend_accel"] = round(feat["trend_20"] - trend_20_prev, 2)
        else:
            feat["trend_accel"] = 0.0

        # -- Volatility --
        feat["volatility_20"] = round(_rolling_volatility(closes, 20, i), 4)
        feat["hist_vol_60"] = round(_rolling_volatility(closes, 60, i), 4)
        atr_14 = _rolling_atr(highs, lows, closes, 14, i)
        feat["atr_14_pct"] = round(atr_14 / closes[i] * 100, 3) if closes[i] > 0 else 0.0
        feat["intraday_amp_20"] = round(_rolling_mean(
            [(highs[j] - lows[j]) / closes[j] * 100 if closes[j] > 0 else 0.0 for j in range(len(closes))],
            20,
            i,
        ), 3)

        # -- Volume ratio --
        avg_vol_20 = _rolling_mean(volumes, 20, i)
        feat["vol_ratio"] = round(volumes[i] / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        # -- Price position --
        if i >= 60:
            h60 = max(highs[i - 59:i + 1])
            l60 = min(lows[i - 59:i + 1])
            feat["price_position"] = round((closes[i] - l60) / (h60 - l60), 3) if h60 > l60 else 0.5
        else:
            feat["price_position"] = 0.5

        # -- Prior returns --
        feat["ret_1d"] = round((closes[i] - closes[i - 1]) / closes[i - 1] * 100, 3) if i >= 1 else 0.0
        feat["ret_3d"] = round((closes[i] - closes[i - 3]) / closes[i - 3] * 100, 2) if i >= 3 else 0.0
        feat["ret_5d"] = round((closes[i] - closes[i - 5]) / closes[i - 5] * 100, 2) if i >= 5 else 0.0
        feat["ret_10d"] = round((closes[i] - closes[i - 10]) / closes[i - 10] * 100, 2) if i >= 10 else 0.0
        feat["ret_20d"] = round((closes[i] - closes[i - 20]) / closes[i - 20] * 100, 2) if i >= 20 else 0.0
        feat["ret_60d"] = round((closes[i] - closes[i - 60]) / closes[i - 60] * 100, 2) if i >= 60 else 0.0
        feat["recent_surge"] = 1 if feat["ret_5d"] >= 10 or feat["ret_20d"] >= 25 else 0

        # -- Consecutive up/down days --
        up_count = 0
        down_count = 0
        for j in range(i, max(0, i - 8), -1):
            if closes[j] > closes[j - 1]:
                if down_count == 0:
                    up_count += 1
                else:
                    break
            elif closes[j] < closes[j - 1]:
                if up_count == 0:
                    down_count += 1
                else:
                    break
            else:
                break
        feat["consec_up"] = min(up_count, 8)
        feat["consec_down"] = min(down_count, 8)

        # -- Volume-price divergence (last 5 days) --
        # vol up + price down → distribution (bearish); vol down + price up → weak rally
        if i >= 5:
            vol_change = (volumes[i] - _rolling_mean(volumes, 5, i - 1)) / max(1, _rolling_mean(volumes, 5, i - 1))
            price_change = (closes[i] - closes[i - 5]) / closes[i - 5]
            feat["vol_price_div"] = round(-vol_change * price_change * 100, 3)
        else:
            feat["vol_price_div"] = 0.0

        # -- Relative strength vs benchmark (date-aligned, not index-aligned) --
        bench_close = bench_closes_aligned[i] if i < len(bench_closes_aligned) else None
        if bench_close is not None:
            # Find benchmark's own index for this date to compute rolling stats (O(1) lookup)
            bench_idx = bench_date_to_idx.get(dates[i])
            if bench_idx is not None and bench_idx >= 20:
                feat["bench_above_ma20"] = 1.0 if bench_close > _rolling_mean(bench_closes_raw, 20, bench_idx) else 0.0
                feat["bench_vol_20"] = round(_rolling_volatility(bench_closes_raw, 20, bench_idx), 4)
                bench_ret_20 = (bench_close - bench_closes_raw[bench_idx-20]) / bench_closes_raw[bench_idx-20] * 100 if bench_idx >= 20 and bench_closes_raw[bench_idx-20] > 0 else 0
                feat["bench_ret_20"] = round(bench_ret_20, 2)
                feat["rel_strength_20"] = round(feat["ret_20d"] - bench_ret_20, 2)
                if bench_idx >= 5 and bench_closes_raw[bench_idx-5] > 0:
                    bench_ret_5 = (bench_close - bench_closes_raw[bench_idx-5]) / bench_closes_raw[bench_idx-5] * 100
                    feat["rel_strength_5"] = round(feat["ret_5d"] - bench_ret_5, 2)
                else:
                    feat["rel_strength_5"] = 0.0
                # Market regime from benchmark's own trend
                bench_trend_20 = _trend_angle(bench_closes_raw, 20, bench_idx)
                if bench_trend_20 > 2 and feat["bench_above_ma20"]:
                    feat["market_regime"] = 1.0
                elif bench_trend_20 < -2 and not feat["bench_above_ma20"]:
                    feat["market_regime"] = -1.0
                else:
                    feat["market_regime"] = 0.0
            else:
                feat["bench_above_ma20"] = 0.0
                feat["bench_ret_20"] = 0.0
                feat["bench_vol_20"] = 0.0
                feat["rel_strength_20"] = 0.0
                feat["rel_strength_5"] = 0.0
                feat["market_regime"] = 0.0
        else:
            feat["rel_strength_20"] = 0.0
            feat["rel_strength_5"] = 0.0
            feat["bench_above_ma20"] = 0.0
            feat["bench_ret_20"] = 0.0
            feat["bench_vol_20"] = 0.0
            feat["market_regime"] = 0.0

        breadth = (market_breadth or {}).get(dates[i])
        feat["market_breadth"] = round(float(breadth), 3) if breadth is not None else 0.5

        industry = (industry_context or {}).get(dates[i], {})
        feat["industry_rel_strength_20"] = round(industry.get("rel_strength_20", 0.0), 3)
        feat["industry_rank_pct"] = round(industry.get("rank_pct", 0.5), 3)

        # -- Day of week --
        feat["dow"] = _day_of_week(dates[i])

        # -- OHLCV ratios --
        feat["hl_ratio"] = round((highs[i] - lows[i]) / closes[i] * 100, 2) if closes[i] > 0 else 0.0
        feat["oc_ratio"] = round((closes[i] - opens[i]) / opens[i] * 100, 2) if opens[i] > 0 else 0.0

        # ── Sentiment proxy (no external data needed) ──
        # Gap shock: overnight gap magnitude (suggests news/events)
        if i >= 1 and closes[i - 1] > 0:
            feat["gap_shock"] = round(abs(opens[i] - closes[i - 1]) / closes[i - 1] * 100, 3)
        else:
            feat["gap_shock"] = 0.0

        # Abnormal volume day (volume > 2x 20-day average)
        feat["abnormal_vol"] = 1 if feat["vol_ratio"] > 2.0 else 0
        feat["liquidity_score"] = round(math.log10(max(volumes[i], 1)), 3)

        # Extreme price move (abs(1d return) > 5%)
        feat["extreme_move"] = 1 if abs(feat["ret_1d"]) > 5.0 else 0
        feat["limit_down_risk"] = 1 if feat["ret_1d"] <= -9.5 else 0

        # ── Capital flow proxy ──
        # Short-term volume acceleration (5d avg vol / 20d avg vol)
        if i >= 5:
            avg_vol_5 = _rolling_mean(volumes, 5, i)
            avg_vol_20_val = _rolling_mean(volumes, 20, i)
            feat["vol_accel"] = round(avg_vol_5 / avg_vol_20_val, 2) if avg_vol_20_val > 0 else 1.0
        else:
            feat["vol_accel"] = 1.0

        # Price-volume correlation (20-day rolling corr between return and volume)
        if i >= 20:
            rets = [(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(i - 18, i + 1) if closes[j - 1] > 0]
            vols = [volumes[j] for j in range(i - 18, i + 1)]
            if len(rets) >= 10:
                mean_r = sum(rets) / len(rets)
                mean_v = sum(vols) / len(vols)
                cov = sum((rets[j] - mean_r) * (vols[j] - mean_v) for j in range(min(len(rets), len(vols))))
                std_r = (sum((r - mean_r)**2 for r in rets) / len(rets))**0.5
                std_v = (sum((v - mean_v)**2 for v in vols) / len(vols))**0.5
                feat["price_vol_corr"] = round(cov / (std_r * std_v + 0.0001) / len(rets), 3)
            else:
                feat["price_vol_corr"] = 0.0
        else:
            feat["price_vol_corr"] = 0.0

        # High-volume streak (proportion of high-vol days in last 10)
        if i >= 10:
            avg_v10 = _rolling_mean(volumes, 10, i)
            high_days = sum(1 for j in range(max(0, i - 9), i + 1) if volumes[j] > avg_v10 * 1.5)
            feat["high_vol_streak"] = round(high_days / 10, 2)
        else:
            feat["high_vol_streak"] = 0.0

        # ── Fundamental features (static/slow-moving, no look-ahead) ──
        if fundamentals:
            mv = fundamentals.get("mv", 0)
            pe = fundamentals.get("pe", 0)
            pb = fundamentals.get("pb", 0)
            industry_hash = fundamentals.get("industry_hash", 0)
            feat["log_mv"] = round(math.log10(mv + 1), 3) if mv > 0 else 0.0
            feat["pe_ratio"] = round(pe, 2) if pe > 0 else 0.0
            feat["pb_ratio"] = round(pb, 2) if pb > 0 else 0.0
            feat["industry_cluster"] = industry_hash % 20
        else:
            feat["log_mv"] = 0.0
            feat["pe_ratio"] = 0.0
            feat["pb_ratio"] = 0.0
            feat["industry_cluster"] = 0

        # ── Macro northbound flow (applies to all stocks) ──
        if northbound_flow:
            date_key = dates[i]
            nb = northbound_flow.get(date_key, {})
            feat["nb_net_flow"] = round(nb.get("net_flow", 0), 2)  # net HK→SH flow in 亿元
            feat["nb_cum_flow_5d"] = round(nb.get("cum_5d", 0), 2)
        else:
            feat["nb_net_flow"] = 0.0
            feat["nb_cum_flow_5d"] = 0.0

        # ── Margin trading (融资融券) ──
        if margin_data:
            date_key = dates[i]
            margin = _closest_margin(margin_data, date_key)
            feat["margin_balance"] = round(margin.get("margin_balance", 0) / 1e8, 2) if margin.get("margin_balance", 0) > 0 else 0.0
            feat["margin_buy_ratio"] = round(margin.get("margin_buy", 0) / max(margin.get("margin_balance", 1), 1), 4)
            feat["margin_net"] = round((margin.get("margin_buy", 0) - margin.get("margin_repay", 0)) / max(margin.get("margin_balance", 1), 1), 4)
            feat["short_ratio"] = round(margin.get("short_vol", 0) / 10000, 2) if margin.get("short_vol", 0) > 0 else 0.0
        else:
            feat["margin_balance"] = 0.0
            feat["margin_buy_ratio"] = 0.0
            feat["margin_net"] = 0.0
            feat["short_ratio"] = 0.0

        # ── 多周期趋势一致性（纯价格数据，永远可用）──
        # trend_alignment: 短中长趋势方向一致 → +1(全多) / -1(全空) / 0(分歧)
        trend_signs = [1 if feat.get(k, 0) > 0 else (-1 if feat.get(k, 0) < 0 else 0)
                       for k in ["trend_10", "trend_20", "trend_60"]]
        if all(s == 1 for s in trend_signs):
            feat["trend_alignment"] = 1.0
        elif all(s == -1 for s in trend_signs):
            feat["trend_alignment"] = -1.0
        else:
            feat["trend_alignment"] = 0.0

        # rsi_momentum: RSI 5日变化量（加速/减速）
        if i >= 5:
            rsi_5d_ago = rsi_vals[i-5]
            feat["rsi_momentum"] = round(rsi_vals[i] - rsi_5d_ago, 1)
        else:
            feat["rsi_momentum"] = 0.0

        # volume_trend: 5日均量 / 20日均量（放量/缩量）
        avg_v5 = _rolling_mean(volumes, 5, i)
        avg_v20 = _rolling_mean(volumes, 20, i)
        feat["volume_trend"] = round(avg_v5 / avg_v20, 2) if avg_v20 > 0 else 1.0

        # ma_alignment: MA多头排列程度（ma5>ma10>ma20>ma60 → 4分满分）
        mas = [ma5, ma10, ma20, ma60]
        bullish_pairs = sum(1 for j in range(len(mas)-1) if mas[j] > mas[j+1])
        feat["ma_alignment"] = round(bullish_pairs / 3, 2)  # 0.0 ~ 1.0

        # bb_position: Bollinger Band 位置（0=下轨, 0.5=中轨, 1=上轨）
        bb_std = feat["volatility_20"] * closes[i] if closes[i] > 0 else 0.01 * closes[i]
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        feat["bb_position"] = round((closes[i] - bb_lower) / (bb_upper - bb_lower), 3) if bb_upper > bb_lower else 0.5

        # turnover_5d: 5日平均换手率（量/市值代理，无基本面数据时用 log 量代替）
        if fundamentals and fundamentals.get("mv", 0) > 0:
            mv_val = fundamentals.get("mv", 0)
            feat["turnover_5d"] = round(avg_v5 / mv_val * 100, 4) if mv_val > 0 else 0.0
        else:
            feat["turnover_5d"] = round(avg_v5 / 1e6, 4)  # 粗略代理

        features.append(feat)

        # ── Label: forward relative return (date-aligned) ──
        future_date = dates[i + lookahead] if i + lookahead < len(dates) else None
        if bench_close is not None and future_date and future_date in bench_by_date:
            stock_ret = (closes[i + lookahead] - closes[i]) / closes[i]
            bench_future_close = bench_by_date[future_date][1]
            bench_ret = (bench_future_close - bench_close) / bench_close
            excess_ret = stock_ret - bench_ret
            label = _classify_excess_return(excess_ret) if target_mode == "excess_4class" else int(excess_ret > 0)
        else:
            stock_ret = (closes[i + lookahead] - closes[i]) / closes[i] if i + lookahead < n else 0
            label = _classify_excess_return(stock_ret) if target_mode == "excess_4class" else int(stock_ret > 0)
        labels.append(label)

    return features, labels


def features_to_matrix(features, feature_names=None):
    """
    Convert list of feature dicts to a list of numeric vectors (for ML input).
    One-hot encodes 'dow' into 5 columns.

    Returns:
        X: list of lists (float values)
        feature_names: list of strings
    """
    feature_names = feature_names or FEATURE_NAMES
    X = []
    for feat in features:
        row = []
        dow = int(feat.get("dow", 0))
        for key in feature_names:
            if key.startswith("dow_"):
                try:
                    day = int(key.split("_", 1)[1])
                except (ValueError, IndexError):
                    day = -1
                row.append(1.0 if day == dow else 0.0)
            else:
                row.append(feat.get(key, 0.0))

        X.append(row)

    return X, feature_names


# ── Internal helpers ──────────────────────────────────────────


def _rolling_mean(values, period, end_idx):
    """Compute rolling mean of `values` over `period` ending at `end_idx` (inclusive)."""
    start = max(0, end_idx - period + 1)
    window = values[start:end_idx + 1]
    return sum(window) / len(window)


def _rolling_volatility(closes, period, end_idx):
    """Compute std of daily returns over `period` bars ending at `end_idx`."""
    if end_idx < 2:
        return 0.0
    start = max(1, end_idx - period + 1)
    returns = []
    for j in range(start, end_idx + 1):
        if closes[j - 1] > 0:
            returns.append((closes[j] - closes[j - 1]) / closes[j - 1])
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)


def _rolling_atr(highs, lows, closes, period, end_idx):
    if end_idx < 1:
        return 0.0
    start = max(1, end_idx - period + 1)
    trs = []
    for j in range(start, end_idx + 1):
        high_low = highs[j] - lows[j]
        high_close = abs(highs[j] - closes[j - 1])
        low_close = abs(lows[j] - closes[j - 1])
        trs.append(max(high_low, high_close, low_close))
    return sum(trs) / len(trs) if trs else 0.0


def _compute_rsi_series(closes, period=14):
    """Compute RSI for every bar. First `period` values default to 50."""
    rsi = []
    for i in range(len(closes)):
        if i < period:
            rsi.append(50.0)
            continue
        gains = []
        losses = []
        for j in range(i - period + 1, i + 1):
            diff = closes[j] - closes[j - 1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / (avg_loss + 0.0001)
            rsi.append(100.0 - 100.0 / (1.0 + rs))
    return rsi


def _trend_angle(prices, period, end_idx):
    """Linear regression slope → angle in degrees, using `period` bars ending at `end_idx`."""
    if end_idx < period - 1:
        return 0.0
    y = prices[end_idx - period + 1:end_idx + 1]
    x = list(range(period))
    n = period
    sxy = sum(x[j] * y[j] for j in range(n))
    sx = sum(x)
    sy = sum(y)
    sxx = sum(x[j] * x[j] for j in range(n))
    denom = n * sxx - sx * sx
    if abs(denom) < 0.0001:
        return 0.0
    slope = (n * sxy - sx * sy) / denom
    return math.degrees(math.atan(slope))


def _closest_margin(margin_data, date_str):
    """Find closest margin data point ≤ date_str (monthly snapshots)."""
    if not margin_data:
        return {}
    target = str(date_str).replace("-", "")
    available = sorted(margin_data.keys(), reverse=True)
    for d in available:
        if str(d).replace("-", "") <= target:
            return margin_data[d]
    return {}


def _day_of_week(date_str):
    """Parse YYYY-MM-DD string → weekday (0=Mon, 4=Fri, None=unknown)."""
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday()  # 0=Monday
    except (ValueError, TypeError):
        return 0


def _classify_excess_return(excess_ret):
    if excess_ret >= 0.02:
        return 3
    if excess_ret >= 0:
        return 2
    if excess_ret > -0.02:
        return 1
    return 0
