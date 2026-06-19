"""
Walk-forward backtest — rigorous out-of-sample validation.

Expanding window: train on data up to cutoff, predict next period,
measure actual returns vs benchmark, roll forward.

This is the GOLD STANDARD for evaluating any quantitative strategy.
"""
import json
import math
import os
import time
import bisect
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np

MODEL_DIR = os.path.expanduser("~/Library/Application Support/QuantDesk/models")
WF_RESULTS_PATH = os.path.join(MODEL_DIR, "walkforward_results.json")
WF_CACHE_DIR = os.path.join(MODEL_DIR, "walkforward")
WF_POLICY_VERSION = "selective_alpha_v7_scaled_clean_signals"


def _normalize_params(params):
    normalized = dict(params or {})
    normalized["policy_version"] = WF_POLICY_VERSION
    if "codes" in normalized:
        normalized["codes"] = [str(code) for code in normalized.get("codes") or []]
    for key in ("train_years", "step_months", "lookahead_days", "top_n", "max_bars", "max_train_samples", "n_estimators"):
        if key in normalized:
            normalized[key] = int(normalized[key])
    return normalized


def _params_cache_key(params):
    raw = json.dumps(_normalize_params(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _wf_cache_path(params):
    return os.path.join(WF_CACHE_DIR, f"{_params_cache_key(params)}.json")


def _date_ordinal(date_str):
    try:
        return int(str(date_str).replace("-", ""))
    except (TypeError, ValueError):
        return 0


def run_walkforward(fetch_kline_fn, codes, benchmark_code="000300",
                    train_years=3, step_months=3, lookahead_days=5,
                    top_n=5, max_bars=2500, commission=0.0003,
                    slippage=0.0005, stamp_tax=0.0005, spread=0.0002,
                    max_train_samples=0, n_estimators=200, progress_cb=None):
    """
    Run expanding-window walk-forward backtest.

    Args:
        fetch_kline_fn: callable(code, days) -> list of 6-tuples
        codes: list of stock codes
        benchmark_code: benchmark for alpha calculation
        train_years: initial training window in years (~250 trading days/year)
        step_months: roll-forward step size in months (~21 trading days/month)
        lookahead_days: prediction horizon (5 = 5-day forward)
        top_n: number of top-ranked stocks to "invest" in each period
        max_bars: max historical bars to fetch

    Returns:
        dict with equity curves, annual stats, and summary metrics
    """
    from sklearn.metrics import roc_auc_score

    started = time.time()
    if progress_cb:
        progress_cb("读取基准与股票K线")

    # ── 1. Fetch all data ──────────────────────────────
    benchmark_kl = fetch_kline_fn(benchmark_code, max_bars)
    if not benchmark_kl or len(benchmark_kl) < 500:
        return {"error": f"基准 {benchmark_code} 数据不足"}

    stock_data = {}
    def load_one(code):
        kl = fetch_kline_fn(code, max_bars)
        return code, kl
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(codes)))) as pool:
        for future in as_completed([pool.submit(load_one, code) for code in codes]):
            code, kl = future.result()
            if kl and len(kl) >= 500:
                stock_data[code] = kl

    if len(stock_data) < 10:
        return {"error": f"数据充足的股票不足: {len(stock_data)} 只"}

    # ── 2. Set up walk-forward windows ──────────────────
    benchmark_dates = [b[0] for b in benchmark_kl]
    benchmark_closes = [b[2] for b in benchmark_kl]
    total_bars = len(benchmark_kl)

    # Initial training needs at least train_years * 250 bars
    train_bars = train_years * 250
    step_bars = step_months * 21

    # First cutoff: after initial training
    # We need lookahead_days + min_bars for prediction
    min_pred_bars = 65  # 60 min_bars + 5 lookahead
    first_cutoff = train_bars
    last_cutoff = total_bars - step_bars - lookahead_days - 5

    cutoffs = list(range(first_cutoff, last_cutoff, step_bars))
    if len(cutoffs) < 3:
        # Adjust: fewer steps
        cutoffs = list(range(first_cutoff, last_cutoff, max(step_bars, 21)))
    if len(cutoffs) < 2:
        return {"error": "数据不足以做滚动回测"}

    # ── 3. Walk-forward loop ────────────────────────────
    periods = []
    equity_model = 1.0
    equity_benchmark = 1.0
    equity_equal = 1.0  # equal-weight baseline
    all_model_returns = []
    all_bench_returns = []

    from predict.train import (
        _build_industry_contexts,
        _build_market_breadth,
        _load_feature_context,
        _require_packages,
        _training_sample_weights_from_labels,
    )
    from predict.features import build_features, features_to_matrix, CORE_FEATURE_NAMES
    import xgboost as xgb
    _require_packages()
    params = _normalize_params({
        "codes": list(codes),
        "benchmark_code": benchmark_code,
        "train_years": train_years,
        "step_months": step_months,
        "lookahead_days": lookahead_days,
        "top_n": top_n,
        "max_bars": max_bars,
        "commission": commission,
        "slippage": slippage,
        "stamp_tax": stamp_tax,
        "spread": spread,
        "max_train_samples": max_train_samples,
        "n_estimators": n_estimators,
    })
    cache_key = _params_cache_key(params)
    fundamentals, northbound_flow, margin_all, data_quality = _load_feature_context(stock_data.keys())
    market_breadth = _build_market_breadth(stock_data)
    industry_contexts = _build_industry_contexts(stock_data, fundamentals)

    print(f"Walk-forward: {len(cutoffs)} steps, {len(stock_data)} stocks, "
          f"{len(fundamentals)} with fundamentals, {len(northbound_flow)} NB days, "
          f"{len(margin_all)} with margin data")

    if progress_cb:
        progress_cb("预计算完整特征矩阵")

    feature_cache = {}
    for code, kl in stock_data.items():
        benchmark_slice = benchmark_kl[:len(kl)] if len(benchmark_kl) >= len(kl) else benchmark_kl
        date_to_idx = {row[0]: idx for idx, row in enumerate(kl)}
        feats, labels = build_features(
            kl,
            benchmark_kl=benchmark_slice,
            lookahead=lookahead_days,
            min_bars=60,
            fundamentals=fundamentals.get(code),
            northbound_flow=northbound_flow,
            margin_data=margin_all.get(code),
            market_breadth=market_breadth,
            industry_context=industry_contexts.get(code),
            target_mode="excess_4class",
        )
        if len(feats) < 30:
            continue
        X, feature_names = features_to_matrix(feats, feature_names=CORE_FEATURE_NAMES)
        train_indices = []
        train_dates = []
        train_ordinals = []
        train_X = []
        train_y = []
        for feat, row, label in zip(feats, X, labels):
            date = feat.get("_date", "")
            idx = date_to_idx.get(date)
            if idx is None:
                continue
            train_indices.append(idx)
            train_dates.append(date)
            train_ordinals.append(_date_ordinal(date))
            train_X.append(row)
            train_y.append(label)

        pred_feats, _ = build_features(
            kl,
            benchmark_kl=benchmark_slice,
            lookahead=0,
            min_bars=60,
            fundamentals=fundamentals.get(code),
            northbound_flow=northbound_flow,
            margin_data=margin_all.get(code),
            market_breadth=market_breadth,
            industry_context=industry_contexts.get(code),
            target_mode="excess_4class",
        )
        pred_X, _ = features_to_matrix(pred_feats, feature_names=feature_names)
        pred_by_idx = {}
        for feat, row in zip(pred_feats, pred_X):
            idx = date_to_idx.get(feat.get("_date"))
            if idx is not None:
                pred_by_idx[idx] = (feat, row)

        feature_cache[code] = {
            "n": len(kl),
            "train_indices": train_indices,
            "train_dates": train_dates,
            "train_ordinals": np.array(train_ordinals, dtype=np.int32),
            "train_X": np.array(train_X, dtype=np.float64),
            "train_y": np.array(train_y, dtype=np.int32),
            "pred_by_idx": pred_by_idx,
            "pred_indices": sorted(pred_by_idx),
        }

    for step_idx, cutoff in enumerate(cutoffs):
        cutoff_date = benchmark_dates[cutoff]
        step_started = time.time()
        if progress_cb:
            progress_cb(f"训练滚动窗口 {step_idx + 1}/{len(cutoffs)}: {cutoff_date}")

        X_parts = []
        y_parts = []
        ordinal_parts = []
        stocks_used = 0
        for code, cached in feature_cache.items():
            stock_cutoff = min(cutoff, cached["n"] - lookahead_days - 5)
            if stock_cutoff < 70:
                continue
            max_train_idx = stock_cutoff - lookahead_days - 1
            selected_end = bisect.bisect_right(cached["train_indices"], max_train_idx)
            if selected_end < 30:
                continue
            X_parts.append(cached["train_X"][:selected_end])
            y_parts.append(cached["train_y"][:selected_end])
            ordinal_parts.append(cached["train_ordinals"][:selected_end])
            stocks_used += 1

        if not X_parts or stocks_used < 5:
            continue

        X_arr = np.concatenate(X_parts)
        y_arr = np.concatenate(y_parts)
        ordinal_arr = np.concatenate(ordinal_parts)
        if len(X_arr) < 200:
            continue
        order = np.argsort(ordinal_arr, kind="mergesort")
        if max_train_samples and len(order) > max_train_samples:
            order = order[-int(max_train_samples):]
        X_arr = X_arr[order]
        y_arr = y_arr[order]
        sample_weights = np.array(_training_sample_weights_from_labels(y_arr.tolist()), dtype=np.float64)

        # Train XGBoost
        model = xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", num_class=4,
            max_depth=4, n_estimators=int(n_estimators), learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        model.fit(X_arr, y_arr, sample_weight=sample_weights)

        # ── Predict on cutoff date ──
        predictions = []
        for code, cached in feature_cache.items():
            stock_cutoff = min(cutoff, cached["n"] - lookahead_days - 5)
            if stock_cutoff < 65:
                continue
            pred_pos = bisect.bisect_right(cached["pred_indices"], stock_cutoff) - 1
            if pred_pos < 0:
                continue
            pred_idx = cached["pred_indices"][pred_pos]
            latest, row = cached["pred_by_idx"][pred_idx]
            Xp = np.array([row], dtype=np.float64)

            rp = model.predict_proba(Xp)[0]
            prob_up = float(rp[2] + rp[3])
            edge_score = float((rp[3] * 2.0 + rp[2] * 0.5) - (rp[1] * 0.5 + rp[0] * 2.0))

            predictions.append({
                "code": code,
                "outperform_prob": prob_up,
                "prob_up": prob_up,
                "strong_outperform_prob": float(rp[3]),
                "strong_underperform_prob": float(rp[0]),
                "edge_score": edge_score,
                "features": latest,
            })

        # ── 市场择时: 200日均线法则 (Faber 2007) ──
        # 先算基准收益
        bench_start = benchmark_kl[cutoff][2] if cutoff < len(benchmark_kl) else 0
        bench_end = benchmark_kl[min(cutoff + lookahead_days, len(benchmark_kl) - 1)][2]
        bench_ret = (bench_end - bench_start) / bench_start if bench_start > 0 else 0

        bench_200ma = statistics.mean(benchmark_closes[max(0, cutoff-199):cutoff+1]) if len(benchmark_closes) > cutoff else benchmark_closes[cutoff] if cutoff < len(benchmark_closes) else 0
        bench_price = benchmark_closes[cutoff] if cutoff < len(benchmark_closes) else 0
        bear_market = bench_price < bench_200ma and bench_200ma > 0

        if bear_market:
            # 熊市持币避险 (Faber 2007)
            periods.append({
                "step": step_idx, "cutoff_date": cutoff_date,
                "stocks_trained": stocks_used, "samples": len(X_arr),
                "model_return": 0.0, "benchmark_return": round(bench_ret * 100, 3),
                "alpha_bps": round(-bench_ret * 10000, 0),
                "active_exposure": 0.0, "eligible": 0,
                "signal_gate": "bear_cash", "top_picks": [],
                "step_elapsed": round(time.time() - step_started, 1),
            })
            all_model_returns.append(0.0)
            all_bench_returns.append(bench_ret)
            equity_model *= (1 + 0.0)
            equity_benchmark *= (1 + bench_ret)
            continue

        # ── Jegadeesh-Titman (1993) 截面动量 + Moskowitz (2012) 波动率缩放 ──
        # 交叉截面排序: 永远选 top-N (不设绝对收益门槛)
        # 评分 = (ret_60d*0.5 + ret_20d*0.3 - ret_5d*0.2) / vol_20
        # ret_5d负权重 = 排除短期反转效应 (Jegadeesh 1990)
        eligible = []
        for pred in predictions:
            feat = pred.get("features", {})
            ret_60d = feat.get("ret_60d", 0) or 0
            ret_20d = feat.get("ret_20d", 0) or 0
            ret_5d = feat.get("ret_5d", 0) or 0
            vol_20 = max(feat.get("volatility_20", 0.02) or 0.02, 0.005)
            atr_pct = feat.get("atr_14_pct", 3) or 3

            # 流动性+极端过滤
            if atr_pct > 8 or feat.get("vol_ratio", 1) < 0.3 or feat.get("recent_surge", 0):
                continue

            # 截面动量评分 (ret/vol信息比率)
            momentum = ret_60d * 0.5 + ret_20d * 0.3 - ret_5d * 0.2
            score = momentum / (vol_20 * 100)
            pred["combined_score"] = score
            eligible.append(pred)

        eligible.sort(key=lambda p: p["combined_score"], reverse=True)

        # 动态仓位：信号越强仓位越高
        if not eligible:
            effective_top_n = 0
            active_exposure = 0.0
            signal_gate = "no_picks"
        else:
            # 用top-5平均评分和信号离散度决定仓位
            top5_scores = [p["combined_score"] for p in eligible[:5]]
            avg_top5 = statistics.mean(top5_scores) if top5_scores else 0
            # 信号离散度：前5名评分差异大=信号清晰=加仓
            score_spread = (top5_scores[0] - top5_scores[-1]) / max(abs(avg_top5), 0.01) if len(top5_scores) >= 2 else 0
            # 仓位 = 基础60% + 信号强度加成(最多到100%)
            base_exposure = 0.60
            signal_bonus = min(0.40, max(0, avg_top5 * 0.5 + score_spread * 0.1))
            active_exposure = base_exposure + signal_bonus
            effective_top_n = min(top_n, max(3, len(eligible) // 5))
            signal_gate = f"top{effective_top_n}_{active_exposure:.0%}"

        # ── Measure actual returns (with stop-loss) ──
        top_picks = eligible[:effective_top_n]
        model_ret_sum = 0.0
        n_valid = 0

        for pick in top_picks:
            code = pick["code"]
            kl = stock_data.get(code)
            if not kl or cutoff + lookahead_days >= len(kl):
                continue
            start_price = kl[cutoff][2] if cutoff < len(kl) else None
            if not start_price or start_price <= 0:
                continue

            # 动态止损+止盈：基于ATR的智能出场
            feat = pick.get("features", {})
            atr_pct = max(0.02, min(0.08, (feat.get("atr_14_pct", 3) or 3) / 100))
            stop_loss_pct = max(0.04, atr_pct * 1.2)  # 止损: max(4%, 1.2x ATR)
            take_profit_pct = atr_pct * 3.0  # 止盈: 3x ATR
            end_idx = min(cutoff + lookahead_days, len(kl) - 1)
            exit_price = kl[end_idx][2]
            exit_reason = "hold"

            for j in range(cutoff + 1, end_idx + 1):
                high_price = kl[j][3]
                low_price = kl[j][4]
                # 止损检查
                if (low_price - start_price) / start_price < -stop_loss_pct:
                    exit_price = start_price * (1 - stop_loss_pct)
                    exit_reason = "stop_loss"
                    break
                # 止盈检查
                if (high_price - start_price) / start_price > take_profit_pct:
                    exit_price = start_price * (1 + take_profit_pct)
                    exit_reason = "take_profit"
                    break

            roundtrip_cost = commission * 2 + slippage * 2 + spread * 2 + stamp_tax
            ret = (exit_price - start_price) / start_price - roundtrip_cost
            # 记录每笔交易的信号强度，用于加权
            weight = 1.0 + pick.get("combined_score", 0) * 0.5
            model_ret_sum += ret * weight
            n_valid += weight

        # (bench_ret 已在市场择时部分计算)
        selected_avg_ret = model_ret_sum / max(1, n_valid) if n_valid > 0 else bench_ret
        model_avg_ret = bench_ret + active_exposure * (selected_avg_ret - bench_ret)

        # Update equity curves
        equity_model *= (1 + model_avg_ret)
        equity_benchmark *= (1 + bench_ret)

        all_model_returns.append(model_avg_ret)
        all_bench_returns.append(bench_ret)

        alpha = model_avg_ret - bench_ret

        periods.append({
            "step": step_idx,
            "cutoff_date": cutoff_date,
            "stocks_trained": stocks_used,
            "samples": len(X_arr),
            "model_return": round(model_avg_ret * 100, 3),
            "benchmark_return": round(bench_ret * 100, 3),
            "alpha_bps": round(alpha * 10000, 0),  # basis points
            "active_exposure": round(active_exposure, 2),
            "eligible": len(eligible),
            "signal_gate": signal_gate,
            "top_picks": [{
                "code": p["code"],
                "outperform_prob": round(p["outperform_prob"], 4),
                "edge_score": round(p.get("edge_score", 0), 4),
                "market_regime": p.get("features", {}).get("market_regime"),
                "bench_ret_20": p.get("features", {}).get("bench_ret_20"),
                "market_breadth": p.get("features", {}).get("market_breadth"),
                "ma60_dev": p.get("features", {}).get("ma60_dev"),
                "atr_14_pct": p.get("features", {}).get("atr_14_pct"),
                "ret_20d": p.get("features", {}).get("ret_20d"),
                "rel_strength_20": p.get("features", {}).get("rel_strength_20"),
                "industry_rel_strength_20": p.get("features", {}).get("industry_rel_strength_20"),
            } for p in top_picks[:3]],
            "step_elapsed": round(time.time() - step_started, 1),
        })

    if len(periods) < 2:
        return {"error": "有效回测期不足"}

    # ── 4. Summary statistics ───────────────────────────
    model_rets = np.array(all_model_returns)
    bench_rets = np.array(all_bench_returns)
    alpha_series = model_rets - bench_rets

    # Annualized metrics (assuming ~21 steps/year with 3-month steps)
    steps_per_year = 12 / step_months
    annual_model_ret = (equity_model ** (1 / (len(periods) / steps_per_year)) - 1) if equity_model > 0 else 0
    annual_bench_ret = (equity_benchmark ** (1 / (len(periods) / steps_per_year)) - 1) if equity_benchmark > 0 else 0
    annual_alpha = annual_model_ret - annual_bench_ret

    # Sharpe of alpha
    alpha_mean = float(np.mean(alpha_series))
    alpha_std = float(np.std(alpha_series)) if len(alpha_series) > 1 else 0.01
    sharpe = alpha_mean / alpha_std * math.sqrt(steps_per_year) if alpha_std > 0 else 0

    # Win rate
    wins = int(np.sum(alpha_series > 0))
    win_rate = wins / len(alpha_series)

    # Max drawdown of cumulative alpha
    cum_alpha = np.cumprod(1 + alpha_series)
    peak = np.maximum.accumulate(cum_alpha)
    drawdown = (cum_alpha - peak) / peak
    max_dd = float(np.min(drawdown))

    # Annual breakdown
    annual = {}
    for p in periods:
        year = p["cutoff_date"][:4]
        if year not in annual:
            annual[year] = {"model_ret": 0, "bench_ret": 0, "periods": 0, "wins": 0}
        annual[year]["model_ret"] += p["model_return"]
        annual[year]["bench_ret"] += p["benchmark_return"]
        annual[year]["periods"] += 1
        if p["model_return"] > p["benchmark_return"]:
            annual[year]["wins"] += 1

    annual_summary = {}
    for year, stats in sorted(annual.items()):
        annual_summary[year] = {
            "model_cum_return": round(stats["model_ret"], 2),
            "bench_cum_return": round(stats["bench_ret"], 2),
            "alpha": round(stats["model_ret"] - stats["bench_ret"], 2),
            "periods": stats["periods"],
            "win_rate": round(stats["wins"] / stats["periods"] * 100, 1),
        }

    # 分离择时收益 vs 选股收益
    timing_alpha_sum = 0.0; stock_alpha_sum = 0.0
    timing_periods = 0; stock_periods = 0
    for p in periods:
        if p.get("signal_gate") == "bear_cash":
            timing_alpha_sum += p.get("alpha_bps", 0) / 10000
            timing_periods += 1
        elif p.get("active_exposure", 0) > 0:
            stock_alpha_sum += (p.get("model_return", 0) - p.get("benchmark_return", 0)) / 100
            stock_periods += 1

    summary = {
        "total_periods": len(periods),
        "date_range": f"{periods[0]['cutoff_date']} → {periods[-1]['cutoff_date']}",
        # 绝对收益（你实际赚到的钱）
        "equity_model": round(equity_model, 4),
        "model_total_return_pct": round((equity_model - 1.0) * 100, 2),
        "equity_benchmark": round(equity_benchmark, 4),
        "bench_total_return_pct": round((equity_benchmark - 1.0) * 100, 2),
        "excess_return": round((equity_model - equity_benchmark) * 100, 2),
        # Alpha拆分
        "annual_model_return_pct": round(annual_model_ret * 100, 2),
        "annual_bench_return_pct": round(annual_bench_ret * 100, 2),
        "annual_alpha_pct": round(annual_alpha * 100, 2),
        "timing_alpha_pct": round(timing_alpha_sum / max(1, timing_periods) * 100, 2) if timing_periods else 0,
        "stock_alpha_pct": round(stock_alpha_sum / max(1, stock_periods) * 100, 2) if stock_periods else 0,
        "timing_periods": timing_periods,
        "stock_periods": stock_periods,
        # 风险指标
        "sharpe_of_alpha": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_stocks": len(stock_data),
        "top_n": top_n,
        "active_periods": stock_periods,
        "avg_active_exposure": round(sum(p.get("active_exposure", 0) for p in periods) / max(1, len(periods)), 3),
        "policy_version": WF_POLICY_VERSION,
        "lookahead_days": lookahead_days,
        "transaction_costs": {
            "commission_per_side": commission,
            "slippage_per_side": slippage,
            "stamp_tax_sell_side": stamp_tax,
            "spread_per_side": spread,
            "roundtrip_total": round(commission * 2 + slippage * 2 + stamp_tax + spread * 2, 5),
        },
        "total_elapsed": round(time.time() - started, 1),
    }

    result = {
        "params": params,
        "cache_key": cache_key,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "annual": annual_summary,
        "periods": periods,
    }

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(WF_CACHE_DIR, exist_ok=True)
    with open(WF_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(_wf_cache_path(params), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def get_wf_status(params=None):
    """Check if walk-forward results exist. Returns full data for frontend."""
    path = _wf_cache_path(params) if params else WF_RESULTS_PATH
    if not os.path.exists(path):
        return {"available": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"available": True, "summary": data.get("summary", {}),
                "annual": data.get("annual", {}),
                "periods": data.get("periods", []),
                "params": data.get("params", {}),
                "cache_key": data.get("cache_key"),
                "generated_at": data.get("generated_at"),
                "n_periods": len(data.get("periods", []))}
    except Exception:
        return {"available": False}
