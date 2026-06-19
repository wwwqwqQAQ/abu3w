"""
XGBoost training pipeline for next-day direction prediction.

Uses strict TimeSeriesSplit (no random shuffle) to avoid look-ahead bias.
"""
import json
import hashlib
import math
import os
import time
import warnings

import numpy as np

from predict.features import (
    FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    TARGET_LABELS,
    build_features,
    features_to_matrix,
)

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.expanduser("~/Library/Application Support/QuantDesk/models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_predictor.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
APP_SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/QuantDesk")
CACHE_DIR = os.path.join(APP_SUPPORT_DIR, "cache")
DEFAULT_LOOKAHEAD_DAYS = 5
DEFAULT_TARGET_MODE = "excess_4class"
_MODEL_CACHE = {"mtime": None, "model": None}


def _require_packages():
    """Lazy import + helpful error if packages are missing."""
    missing = []
    try:
        import xgboost  # noqa: F401
    except ImportError:
        missing.append("xgboost")
    try:
        import sklearn  # noqa: F401
    except ImportError:
        missing.append("scikit-learn")
    try:
        import joblib  # noqa: F401
    except ImportError:
        missing.append("joblib")
    if missing:
        raise ImportError(
            f"缺少 Python 包: {', '.join(missing)}。"
            f"请运行: pip install {' '.join(missing)}"
        )


def _load_json(path, default=None):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _load_first_json(paths, default=None):
    for path in paths:
        data = _load_json(path, None)
        if data is not None:
            return data, path
    return default, ""


def _industry_cluster(industry):
    text = str(industry or "").strip()
    if not text or text == "-":
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 20


def _normalize_fundamentals(raw):
    fundamentals = {}
    for code, data in (raw or {}).items():
        industry = data.get("industry", "")
        fundamentals[str(code)] = {
            "mv": data.get("mv", 0) or 0,
            "pe": data.get("pe", 0) or 0,
            "pb": data.get("pb", 0) or 0,
            "industry": industry,
            "industry_hash": _industry_cluster(industry),
        }
    return fundamentals


def _load_northbound_flow():
    raw, path = _load_first_json([
        os.path.join(APP_SUPPORT_DIR, "northbound_flow.json"),
        os.path.join(CACHE_DIR, "northbound_flow.json"),
    ], [])
    flow = {}
    for entry in raw or []:
        date = entry.get("date")
        if not date:
            continue
        flow[date] = {
            "net_flow": entry.get("net_flow", 0) or 0,
            "cum_5d": entry.get("cum_5d", 0) or 0,
        }
    return flow, path


def _load_margin_data(codes=None):
    margin_dir = os.path.join(CACHE_DIR, "margin")
    if not os.path.isdir(margin_dir):
        return {}, margin_dir
    wanted = {str(code) for code in codes} if codes else None
    margin = {}
    for fname in os.listdir(margin_dir):
        if not fname.endswith(".json"):
            continue
        code = fname[:-5]
        if wanted is not None and code not in wanted:
            continue
        data = _load_json(os.path.join(margin_dir, fname), None)
        if data:
            margin[code] = data
    return margin, margin_dir


def _load_feature_context(codes=None):
    raw_fund, fund_path = _load_first_json([
        os.path.join(APP_SUPPORT_DIR, "fundamentals.json"),
        os.path.join(CACHE_DIR, "fundamentals.json"),
    ], {})
    fundamentals = _normalize_fundamentals(raw_fund)
    northbound_flow, northbound_path = _load_northbound_flow()
    margin_data, margin_path = _load_margin_data(codes)
    quality = {
        "fundamentals": len(fundamentals),
        "northbound_days": len(northbound_flow),
        "margin_stocks": len(margin_data),
        "fundamentals_path": fund_path,
        "northbound_path": northbound_path,
        "margin_path": margin_path,
    }
    return fundamentals, northbound_flow, margin_data, quality


def _model_lookahead_days():
    metrics = get_status()
    return int(metrics.get("lookahead_days") or DEFAULT_LOOKAHEAD_DAYS)


def _model_target_mode(model=None):
    if isinstance(model, dict):
        return model.get("target_mode") or get_status().get("target_mode") or "binary_outperform"
    return get_status().get("target_mode") or "binary_outperform"


def _model_feature_names(model):
    if isinstance(model, dict) and model.get("feature_names"):
        return model["feature_names"]
    return CORE_FEATURE_NAMES


def _class_distribution(labels):
    total = len(labels)
    counts = {str(k): int(sum(1 for y in labels if int(y) == k)) for k in range(4)}
    return {
        TARGET_LABELS[k]: {"count": counts[str(k)], "pct": round(counts[str(k)] / total * 100, 2) if total else 0}
        for k in range(4)
    }


def _xgb_candidate_params():
    return [
        {
            "max_depth": 2,
            "n_estimators": 260,
            "learning_rate": 0.035,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 8,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
        {
            "max_depth": 3,
            "n_estimators": 220,
            "learning_rate": 0.045,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 6,
            "reg_lambda": 4.0,
            "reg_alpha": 0.1,
        },
        {
            "max_depth": 4,
            "n_estimators": 180,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
        {
            "max_depth": 2,
            "n_estimators": 420,
            "learning_rate": 0.025,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 10,
            "reg_lambda": 8.0,
            "reg_alpha": 0.5,
        },
    ]


def _safe_auc(roc_auc_score, y_true, y_score):
    try:
        if len(set(int(y) for y in y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _fit_xgb(xgb, X, y, weights, objective, params, num_class=None):
    model_params = {
        **params,
        "objective": objective,
        "eval_metric": "mlogloss" if objective == "multi:softprob" else "logloss",
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }
    if num_class is not None:
        model_params["num_class"] = int(num_class)
    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y, sample_weight=weights)
    return model


def _build_market_breadth(stock_data):
    up_counts = {}
    total_counts = {}
    for kl in stock_data.values():
        for i in range(1, len(kl)):
            date = kl[i][0]
            total_counts[date] = total_counts.get(date, 0) + 1
            if kl[i][2] > kl[i - 1][2]:
                up_counts[date] = up_counts.get(date, 0) + 1
    return {
        date: up_counts.get(date, 0) / total
        for date, total in total_counts.items()
        if total > 0
    }


def _build_industry_contexts(stock_data, fundamentals):
    returns_by_code = {}
    industry_by_code = {}
    for code, kl in stock_data.items():
        industry = (fundamentals.get(code) or {}).get("industry") or ""
        if not industry or industry == "-":
            continue
        industry_by_code[code] = industry
        by_date = {}
        for i in range(20, len(kl)):
            prev = kl[i - 20][2]
            if prev > 0:
                by_date[kl[i][0]] = (kl[i][2] - prev) / prev * 100
        returns_by_code[code] = by_date

    grouped = {}
    for code, by_date in returns_by_code.items():
        industry = industry_by_code[code]
        for date, ret in by_date.items():
            grouped.setdefault(date, {}).setdefault(industry, []).append((code, ret))

    contexts = {code: {} for code in returns_by_code}
    for date, industries in grouped.items():
        for industry, rows in industries.items():
            if not rows:
                continue
            avg_ret = sum(ret for _, ret in rows) / len(rows)
            ranked = sorted(rows, key=lambda item: item[1])
            n = max(1, len(ranked) - 1)
            for rank, (code, ret) in enumerate(ranked):
                contexts[code][date] = {
                    "rel_strength_20": ret - avg_ret,
                    "rank_pct": rank / n if n else 0.5,
                }
    return contexts


def train_all_stocks(fetch_kline_fn, codes, days=500, cv_splits=5, benchmark_code=None,
                     lookahead_days=DEFAULT_LOOKAHEAD_DAYS, target_mode=DEFAULT_TARGET_MODE):
    """
    Train XGBoost classifier on all specified stocks.

    Args:
        fetch_kline_fn: callable(code, days) -> list of 6-tuples
        codes: list of stock code strings
        days: historical window per stock
        cv_splits: number of TimeSeriesSplit folds
        benchmark_code: optional benchmark stock code (e.g. '000300' for CSI 300)

    Returns:
        dict with keys: accuracy, auc, precision, recall, f1,
                        feature_importance, n_samples, n_features,
                        n_stocks_used, elapsed, best_params, calibration
    """
    _require_packages()
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    import joblib

    started = time.time()

    # Fetch benchmark if specified
    benchmark_kl = None
    if benchmark_code:
        benchmark_kl = fetch_kline_fn(benchmark_code, days)

    stock_data = {}
    for code in codes:
        if benchmark_code and code == benchmark_code:
            continue
        kl = fetch_kline_fn(code, days)
        if kl and len(kl) >= 80:
            stock_data[code] = kl

    # ── 1. Collect all features ──────────────────────────
    all_X = []
    all_y = []
    all_rows = []
    stocks_used = 0
    fundamentals, northbound_flow, margin_data, data_quality = _load_feature_context(codes)
    market_breadth = _build_market_breadth(stock_data)
    industry_contexts = _build_industry_contexts(stock_data, fundamentals)

    for code, kl in stock_data.items():
        feats, labels = build_features(
            kl,
            benchmark_kl=benchmark_kl,
            lookahead=lookahead_days,
            min_bars=60,
            fundamentals=fundamentals.get(code),
            northbound_flow=northbound_flow,
            margin_data=margin_data.get(code),
            market_breadth=market_breadth,
            industry_context=industry_contexts.get(code),
            target_mode=target_mode,
        )
        if len(feats) < 30:
            continue
        X, _ = features_to_matrix(feats, feature_names=CORE_FEATURE_NAMES)
        for feat, row, label in zip(feats, X, labels):
            all_rows.append((feat.get("_date", ""), row, label))
        stocks_used += 1

    all_rows.sort(key=lambda item: item[0])
    all_X = [row for _, row, _ in all_rows]
    all_y = [label for _, _, label in all_rows]

    if len(all_X) < 200:
        return {
            "error": f"训练样本不足: {len(all_X)} 行 (需要至少 200 行)。请确认 K 线数据已缓存。",
            "n_samples": len(all_X),
        }

    X_arr = np.array(all_X, dtype=np.float64)
    y_arr = np.array(all_y, dtype=np.int32)
    feature_names_list = CORE_FEATURE_NAMES
    if target_mode == "excess_4class" and len(set(y_arr.tolist())) < 2:
        return {"error": "四分类训练样本类别不足，无法训练。", "class_distribution": _class_distribution(y_arr.tolist())}

    sample_weights_all = np.array(_training_sample_weights(all_rows), dtype=np.float64)
    y_out_arr = (y_arr >= 2).astype(np.int32) if target_mode == "excess_4class" else y_arr

    # Strict chronological train / validation / final holdout.
    train_end = int(len(X_arr) * 0.64)
    val_end = int(len(X_arr) * 0.82)
    X_train, X_val, X_eval = X_arr[:train_end], X_arr[train_end:val_end], X_arr[val_end:]
    y_train, y_val, y_eval = y_arr[:train_end], y_arr[train_end:val_end], y_arr[val_end:]
    y_out_train = y_out_arr[:train_end]
    y_out_val = y_out_arr[train_end:val_end]
    y_out_eval = y_out_arr[val_end:]
    w_train = sample_weights_all[:train_end]
    X_dev, y_dev = X_arr[:val_end], y_arr[:val_end]
    y_out_dev = y_out_arr[:val_end]
    w_dev = sample_weights_all[:val_end]

    candidates = _xgb_candidate_params()
    candidate_report = []
    best_multi = {"score": -1, "params": candidates[0]}
    best_binary = {"score": -1, "params": candidates[0]}

    for idx, params in enumerate(candidates):
        multi = _fit_xgb(xgb, X_train, y_train, w_train, "multi:softprob", params, num_class=4)
        val_proba_multi = multi.predict_proba(X_val)
        val_pred = np.argmax(val_proba_multi, axis=1)
        val_auc_multi = _safe_auc(roc_auc_score, y_out_val, val_proba_multi[:, 2] + val_proba_multi[:, 3])
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        multi_score = (val_auc_multi or 0.5) * 0.70 + val_f1 * 0.30

        binary = _fit_xgb(xgb, X_train, y_out_train, w_train, "binary:logistic", params)
        val_proba_binary = binary.predict_proba(X_val)[:, 1]
        val_auc_binary = _safe_auc(roc_auc_score, y_out_val, val_proba_binary)
        binary_score = val_auc_binary or 0.0

        candidate_report.append({
            "idx": idx,
            "params": params,
            "multi_val_auc": round(val_auc_multi, 4) if val_auc_multi is not None else None,
            "multi_val_macro_f1": round(val_f1, 4),
            "binary_val_auc": round(val_auc_binary, 4) if val_auc_binary is not None else None,
        })
        if multi_score > best_multi["score"]:
            best_multi = {"score": multi_score, "params": params}
        if binary_score > best_binary["score"]:
            best_binary = {"score": binary_score, "params": params}

    eval_multi = _fit_xgb(xgb, X_dev, y_dev, w_dev, "multi:softprob", best_multi["params"], num_class=4)
    eval_binary = _fit_xgb(xgb, X_dev, y_out_dev, w_dev, "binary:logistic", best_binary["params"])

    production_multi = _fit_xgb(xgb, X_arr, y_arr, sample_weights_all, "multi:softprob", best_multi["params"], num_class=4)
    production_binary = _fit_xgb(xgb, X_arr, y_out_arr, sample_weights_all, "binary:logistic", best_binary["params"])

    final_model = {
        "base": production_multi,
        "outperform_model": production_binary,
        "target_mode": target_mode,
        "class_labels": TARGET_LABELS,
        "feature_names": feature_names_list,
        "lookahead_days": lookahead_days,
        "model_architecture": "multi_class_plus_binary_outperform_head",
        "best_multi_params": best_multi["params"],
        "best_binary_params": best_binary["params"],
    }

    # ── 5. Evaluate on untouched final chronological holdout ─────────
    y_proba = eval_multi.predict_proba(X_eval)
    y_pred = np.argmax(y_proba, axis=1) if target_mode == "excess_4class" else eval_binary.predict(X_eval)
    proba_out_multi = y_proba[:, 2] + y_proba[:, 3] if target_mode == "excess_4class" else y_proba[:, 1]
    proba_out = eval_binary.predict_proba(X_eval)[:, 1]
    outperform_auc_raw = _safe_auc(roc_auc_score, y_out_eval, proba_out)
    multiclass_outperform_auc_raw = _safe_auc(roc_auc_score, y_out_eval, proba_out_multi)
    outperform_auc = round(outperform_auc_raw, 4) if outperform_auc_raw is not None else None
    multiclass_outperform_auc = round(multiclass_outperform_auc_raw, 4) if multiclass_outperform_auc_raw is not None else None

    top_n_eval = max(20, int(len(y_out_eval) * 0.10))
    top_idx = np.argsort(proba_out)[-top_n_eval:]
    base_outperform_rate = float(np.mean(y_out_eval)) if len(y_out_eval) else 0.0
    top_decile_hit_rate = float(np.mean(y_out_eval[top_idx])) if len(top_idx) else 0.0
    top_decile_lift = top_decile_hit_rate - base_outperform_rate
    model_health = {
        "tradable": bool(
            outperform_auc_raw is not None
            and outperform_auc_raw >= 0.52
            and top_decile_lift >= 0.03
        ),
        "reason": "",
        "auc_threshold": 0.52,
        "top_decile_lift_threshold": 0.03,
        "base_outperform_rate": round(base_outperform_rate * 100, 2),
        "top_decile_hit_rate": round(top_decile_hit_rate * 100, 2),
        "top_decile_lift": round(top_decile_lift * 100, 2),
    }
    if not model_health["tradable"]:
        model_health["reason"] = "样本外AUC或Top10%命中率未过交易门槛"
    else:
        model_health["reason"] = "样本外AUC与Top10%命中率通过交易门槛"

    metrics = {
        "target_mode": target_mode,
        "target_description": "未来5天个股收益 - 沪深300收益，四分类: 强跑输/小跑输/小跑赢/强跑赢",
        "primary_metric": "out_of_sample_macro_f1",
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_eval, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, y_pred, average="macro", zero_division=0)), 4),
        "outperform_auc": outperform_auc,
        "multiclass_outperform_auc": multiclass_outperform_auc,
        "model_health": model_health,
        "n_samples": len(X_arr),
        "n_features": X_arr.shape[1],
        "n_stocks_used": stocks_used,
        "n_eval": len(y_eval),
        "lookahead_days": lookahead_days,
        "class_labels": TARGET_LABELS,
        "class_distribution": _class_distribution(y_arr.tolist()),
        "eval_class_distribution": _class_distribution(y_eval.tolist()),
        "data_quality": data_quality,
        "elapsed": round(time.time() - started, 1),
        "best_params": {
            "multi": best_multi["params"],
            "binary_outperform": best_binary["params"],
        },
        "candidate_report": candidate_report,
    }

    # Feature importance from the binary outperform head used for ranking.
    importances = production_binary.feature_importances_
    fi_sorted = sorted(
        zip(feature_names_list, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    metrics["feature_importance"] = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in fi_sorted
    ]

    # ── 6. Save ──────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    _MODEL_CACHE["mtime"] = os.path.getmtime(MODEL_PATH)
    _MODEL_CACHE["model"] = final_model
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def load_model():
    """Load trained model. Returns None if not trained yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    mtime = os.path.getmtime(MODEL_PATH)
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["mtime"] == mtime:
        return _MODEL_CACHE["model"]
    import joblib
    model = joblib.load(MODEL_PATH)
    _MODEL_CACHE["mtime"] = mtime
    _MODEL_CACHE["model"] = model
    return model


def _predict_proba(model, X_arr):
    """Get calibrated probabilities from model dict or raw XGBoost."""
    import numpy as np

    if isinstance(model, dict) and "base" in model:
        base = model["base"]
        if model.get("target_mode") == "excess_4class" or "calibrator" not in model:
            return base.predict_proba(X_arr)
        cal = model["calibrator"]
        raw_proba = base.predict_proba(X_arr)[:, 1]
        eps = 1e-12
        log_odds = np.log((raw_proba + eps) / (1 - raw_proba + eps))
        alpha = cal["alpha"]
        beta = cal["beta"]
        cal_log_odds = alpha * log_odds + beta
        cal_proba = 1.0 / (1.0 + np.exp(-cal_log_odds))
        return np.column_stack([1 - cal_proba, cal_proba])
    else:
        # Raw XGBoost model
        return model.predict_proba(X_arr)


def _predict_outperform_proba(model, X_arr):
    if isinstance(model, dict) and model.get("outperform_model") is not None:
        return model["outperform_model"].predict_proba(X_arr)[:, 1]
    proba = _predict_proba(model, X_arr)
    if proba.shape[1] >= 4:
        return proba[:, 2] + proba[:, 3]
    return proba[:, 1]


def get_status():
    """Check if model exists and return last training metrics."""
    if not os.path.exists(METRICS_PATH):
        return {"trained": False, "model_exists": False}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["trained"] = True
        metrics["model_exists"] = os.path.exists(MODEL_PATH)
        return metrics
    except (json.JSONDecodeError, IOError):
        return {"trained": False, "model_exists": os.path.exists(MODEL_PATH)}


def _confidence_from_probs(probs):
    top = max(probs)
    ordered = sorted(probs, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    if top >= 0.55 or margin >= 0.25:
        return "高"
    if top >= 0.40 or margin >= 0.12:
        return "中"
    return "低"


def _training_sample_weights_from_labels(labels):
    if labels is None or len(labels) == 0:
        return []
    labels = [int(label) for label in labels]
    counts = {label: max(1, labels.count(label)) for label in set(labels)}
    n_classes = max(1, len(counts))
    total = len(labels)
    weights = []
    denom = max(1, len(labels) - 1)
    for idx, label in enumerate(labels):
        recency = 0.75 + 0.55 * (idx / denom)
        class_balance = math.sqrt(total / (n_classes * counts[label]))
        class_strength = 1.20 if label in (0, 3) else 1.0
        weights.append(round(recency * class_balance * class_strength, 6))
    return weights


def _training_sample_weights(rows):
    """Time-decayed, class-balanced sample weights for outperformance learning."""
    if not rows:
        return []
    return _training_sample_weights_from_labels([int(item[2]) for item in rows])


def _rule_flags(feat):
    flags = []
    ma60_dev = feat.get("ma60_dev") or 0
    trend_20 = feat.get("trend_20") or 0
    if ma60_dev <= -10 and trend_20 < -20:
        flags.append({"type": "strong_reduce", "code": "deep_below_ma60", "message": "深度跌破MA60且短线下行，只允许极轻试仓"})
    elif ma60_dev < 0:
        flags.append({"type": "reduce", "code": "below_ma60", "message": "跌破MA60，降低仓位"})
    vol_ratio = feat.get("vol_ratio") or 1
    if vol_ratio < 0.25:
        flags.append({"type": "no_buy", "code": "dead_volume", "message": "成交量极低，无法交易"})
    elif vol_ratio < 0.5:
        flags.append({"type": "strong_reduce", "code": "low_volume", "message": "成交量偏低，只允许小仓"})
    if feat.get("recent_surge"):
        flags.append({"type": "reduce", "code": "recent_surge", "message": "近期暴涨过不追"})
    if (feat.get("bench_ret_20") or 0) <= -5:
        flags.append({"type": "reduce", "code": "market_drop", "message": "大盘大跌时降低仓位"})
    if feat.get("limit_down_risk"):
        flags.append({"type": "no_buy", "code": "limit_down_risk", "message": "存在跌停/极端下跌风险"})
    if (feat.get("atr_14_pct") or 0) >= 6:
        flags.append({"type": "reduce", "code": "high_atr", "message": "ATR过高，降低仓位"})
    if (feat.get("rel_strength_20") or 0) < -5:
        flags.append({"type": "reduce", "code": "weak_vs_benchmark", "message": "20日相对沪深300明显走弱"})
    if (feat.get("industry_rel_strength_20") or 0) < -5:
        flags.append({"type": "reduce", "code": "weak_vs_industry", "message": "相对行业明显走弱"})
    return flags


def _suggest_position(outperform_prob, confidence, feat, flags):
    reasons = [flag["message"] for flag in flags]
    if any(flag["type"] == "no_buy" for flag in flags):
        return {"label": "空仓", "weight": 0.0, "reasons": reasons}

    edge = outperform_prob - 0.5
    if outperform_prob >= 0.66:
        weight = 0.5
        label = "中仓"
    elif outperform_prob >= 0.60:
        weight = 0.3
        label = "轻仓"
    elif outperform_prob >= 0.55:
        weight = 0.2
        label = "试仓"
    elif outperform_prob >= 0.52 and ((feat.get("trend_20") or 0) > 0 or (feat.get("rel_strength_20") or 0) > 0):
        weight = 0.1
        label = "观察试仓"
    else:
        weight = 0.0
        label = "观望"
        if edge <= 0.08:
            reasons.append("跑赢边际不足，不偏离沪深300")

    if confidence == "低" and outperform_prob < 0.6:
        weight = min(weight, 0.12)
        reasons.append("四分类置信度低，只保留小仓观察")
    if (feat.get("market_regime") or 0) < 0:
        weight *= 0.5
        reasons.append("熊市/弱市场状态")
    if (feat.get("bench_ret_20") or 0) <= -5:
        weight *= 0.5
        reasons.append("沪深300 20日走弱，降低主动风险")
    reduce_count = sum(1 for flag in flags if flag["type"] == "reduce")
    strong_reduce_count = sum(1 for flag in flags if flag["type"] == "strong_reduce")
    weight *= 0.75 ** reduce_count
    weight *= 0.45 ** strong_reduce_count
    if (feat.get("hist_vol_60") or 0) >= 0.04:
        weight *= 0.7
        reasons.append("历史波动率偏高")

    weight = round(max(0.0, min(0.8, weight)), 2)
    if weight == 0 and label != "空仓":
        label = "观望"
    elif weight <= 0.2 and label not in ("空仓", "观望"):
        label = "试仓"
    elif weight <= 0.4 and label not in ("空仓", "观望"):
        label = "轻仓"
    return {"label": label, "weight": weight, "reasons": reasons}


def predict_single(fetch_kline_fn, code, benchmark_code=None, model=None, feature_context=None,
                   horizon_days=None, stock_kl=None, benchmark_kl=None,
                   market_breadth=None, industry_context=None):
    """
    Predict 5-day direction vs benchmark for a single stock.

    Returns:
        dict with code, name, prob_up, signal, confidence,
        and the features used for this prediction.
    """
    if model is None:
        model = load_model()
    if model is None:
        return {"error": "模型未训练，请先调用 /api/predict/train"}

    kl = stock_kl if stock_kl is not None else fetch_kline_fn(code, 500)
    if not kl or len(kl) < 80:
        return {"error": f"{code} 数据不足"}

    if benchmark_kl is None and benchmark_code and benchmark_code != code:
        benchmark_kl = fetch_kline_fn(benchmark_code, 500)

    if feature_context is None:
        feature_context = _load_feature_context([code])
    fundamentals, northbound_flow, margin_data, _ = feature_context

    feats, _ = build_features(
        kl,
        benchmark_kl=benchmark_kl,
        lookahead=0,
        min_bars=60,
        fundamentals=fundamentals.get(code),
        northbound_flow=northbound_flow,
        margin_data=margin_data.get(code),
        market_breadth=market_breadth,
        industry_context=industry_context,
    )
    if not feats:
        return {"error": f"{code} 无法提取特征"}

    # Use the most recent bar's features
    latest_feat = feats[-1]
    feature_names = _model_feature_names(model)
    X, _ = features_to_matrix([latest_feat], feature_names=feature_names)
    X_arr = np.array(X, dtype=np.float64)

    proba = _predict_proba(model, X_arr)[0]
    outperform_head_prob = float(_predict_outperform_proba(model, X_arr)[0])
    target_mode = _model_target_mode(model)
    if target_mode == "excess_4class" or len(proba) >= 4:
        probs = [float(p) for p in proba[:4]]
        outperform_prob = round(outperform_head_prob, 4)
        top_class = int(np.argmax(probs))
        signal = TARGET_LABELS.get(top_class, "中性")
        confidence = _confidence_from_probs(probs)
        class_probabilities = {
            TARGET_LABELS[i]: round(probs[i], 4)
            for i in range(4)
        }
        pred = int(outperform_prob >= 0.5)
    else:
        outperform_prob = round(float(proba[1]), 4)
        signal = "小跑赢" if outperform_prob >= 0.55 else "小跑输" if outperform_prob <= 0.45 else "中性"
        confidence = "高" if abs(outperform_prob - 0.5) >= 0.15 else "中" if abs(outperform_prob - 0.5) >= 0.07 else "低"
        class_probabilities = {
            "跑输": round(float(proba[0]), 4),
            "跑赢": outperform_prob,
        }
        pred = int(outperform_prob >= 0.5)

    flags = _rule_flags(latest_feat)
    suggested_position = _suggest_position(outperform_prob, confidence, latest_feat, flags)
    model_health = (get_status() or {}).get("model_health", {})
    if model_health and model_health.get("tradable") is False:
        suggested_position = {
            "label": "观望",
            "weight": 0.0,
            "reasons": [model_health.get("reason") or "模型健康度未过交易门槛"],
        }

    return {
        "code": code,
        "outperform_prob": outperform_prob,
        "prob_up": outperform_prob,
        "prob_down": round(1 - outperform_prob, 4),
        "class_probabilities": class_probabilities,
        "signal": signal,
        "confidence": confidence,
        "prediction": pred,
        "feature_date": latest_feat.get("_date"),
        "prediction_horizon_days": int(horizon_days or _model_lookahead_days()),
        "target": f"未来{int(horizon_days or _model_lookahead_days())}天跑赢沪深300概率",
        "suggested_position": suggested_position,
        "model_health": model_health,
        "rule_flags": flags,
        "features_used": {
            "rsi_14": latest_feat.get("rsi_14"),
            "ma60_dev": latest_feat.get("ma60_dev"),
            "trend_20": latest_feat.get("trend_20"),
            "trend_60": latest_feat.get("trend_60"),
            "volatility_20": latest_feat.get("volatility_20"),
            "hist_vol_60": latest_feat.get("hist_vol_60"),
            "atr_14_pct": latest_feat.get("atr_14_pct"),
            "ma20_dev": latest_feat.get("ma20_dev"),
            "price_position": latest_feat.get("price_position"),
            "vol_ratio": latest_feat.get("vol_ratio"),
            "margin_balance": latest_feat.get("margin_balance"),
            "margin_buy_ratio": latest_feat.get("margin_buy_ratio"),
            "margin_net": latest_feat.get("margin_net"),
            "nb_net_flow": latest_feat.get("nb_net_flow"),
            "bench_above_ma20": latest_feat.get("bench_above_ma20"),
            "bench_ret_20": latest_feat.get("bench_ret_20"),
            "market_regime": latest_feat.get("market_regime"),
            "market_breadth": latest_feat.get("market_breadth"),
            "industry_rel_strength_20": latest_feat.get("industry_rel_strength_20"),
            "industry_rank_pct": latest_feat.get("industry_rank_pct"),
        },
    }


def predict_all(fetch_kline_fn, codes, name_map=None, benchmark_code=None, model=None, inference_days=500):
    """
    Predict for all stocks, sorted by confidence / prob_up.

    Returns:
        list of prediction dicts sorted by |prob_up - 0.5| descending.
    """
    results = []
    if model is None:
        model = load_model()
    if model is None:
        return results
    feature_context = _load_feature_context(codes)
    fundamentals = feature_context[0]
    horizon_days = _model_lookahead_days()
    stock_data = {}
    days = max(500, int(inference_days or 500))
    for code in codes:
        kl = fetch_kline_fn(code, days)
        if kl and len(kl) >= 80:
            stock_data[code] = kl
    benchmark_kl = fetch_kline_fn(benchmark_code, days) if benchmark_code else None
    market_breadth = _build_market_breadth(stock_data)
    industry_contexts = _build_industry_contexts(stock_data, fundamentals)
    for code in codes:
        pred = predict_single(
            fetch_kline_fn,
            code,
            benchmark_code=benchmark_code,
            model=model,
            feature_context=feature_context,
            horizon_days=horizon_days,
            stock_kl=stock_data.get(code),
            benchmark_kl=benchmark_kl,
            market_breadth=market_breadth,
            industry_context=industry_contexts.get(code),
        )
        if "error" in pred:
            continue
        if name_map:
            pred["name"] = name_map.get(code, code)
        results.append(pred)

    # Sort actionable candidates first.
    results.sort(
        key=lambda r: (
            (r.get("suggested_position") or {}).get("weight", 0),
            r.get("outperform_prob", 0),
        ),
        reverse=True,
    )
    return results
