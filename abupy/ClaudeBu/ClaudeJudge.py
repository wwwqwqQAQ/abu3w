# -*- encoding:utf-8 -*-
"""
    ClaudeJudge — Claude 驱动的交易裁判

    两种模式:
    - rule:  让 Claude 分析历史交易 → 生成 Python 判断规则 → 本地高速执行
    - direct: 每笔/每批交易直调 Claude API → 精确但慢

    兼容 AbuUmpManager.append_user_ump() 接口，
    实现 class_unique_id(), is_buy_ump(), predict(**ml_feature_dict)
"""
from __future__ import absolute_import, print_function, division

import copy
import hashlib
import json
import logging
import re
import textwrap

import numpy as np
import pandas as pd

try:
    from .ClaudeConfig import ClaudeConfig
except ImportError:
    from ClaudeConfig import ClaudeConfig

__author__ = 'ClaudeBu'
__all__ = ['ClaudeJudge']


# ──────────────────────────────────────────────────────────
# 不依赖 abu 的 EEdgeType 枚举（避免导入整个 abu 时缺依赖报错）
# 值含义: -1 = 拦截, 0 = 放行, 1 = 强烈推荐
# ──────────────────────────────────────────────────────────
class _EEdgeType:
    E_EEdge_TOP_LOSS = -1      # 拦截
    E_EEdge_NORMAL = 0          # 放行
    E_EEdge_TOP_WIN = 1         # 推荐

# ──────────────────────────────────────────────────────────
# 特征说明文档 — 发给 Claude 帮助理解 ml_feature_dict
# ──────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = """
每个交易信号的特征字典 (ml_feature_dict) 包含以下指标:

【趋势角度】值域 -90~90, 正值=上涨趋势, 负值=下跌趋势
- buy_deg_ang21:  21天线性回归角度
- buy_deg_ang42:  42天线性回归角度
- buy_deg_ang60:  60天线性回归角度
- buy_deg_ang252: 252天(1年)线性回归角度

【价格位置】值域 0~1, 越接近1说明价格越接近N天高点
- buy_price_rank60:  当前价在60天内的百分位
- buy_price_rank90:  当前价在90天内的百分位
- buy_price_rank120: 当前价在120天内的百分位
- buy_price_rank252: 当前价在252天内的百分位

【波动率特征】
- buy_wave_score1: 42天滚动窗口波动率得分
- buy_wave_score2: 84天滚动窗口波动率得分
- buy_wave_score3: 126天滚动窗口波动率得分
- buy_atr_std:     42天ATR标准化值 (越高波动越大)

【跳空缺口】
- buy_jump_down_power: 最近向下跳空的力度 (负值表示缺口向下)
- buy_diff_down_days:  距最近向下跳空的天数
- buy_jump_up_power:   最近向上跳空的力度 (正值表示缺口向上)
- buy_diff_up_days:    距最近向上跳空的天数

【卖出信号对应特征】(前缀 sell_ 替代 buy_，含义相同)
- sell_deg_ang21, sell_deg_ang42, sell_deg_ang60, sell_deg_ang252
- sell_price_rank60, sell_price_rank90, sell_price_rank120, sell_price_rank252
- sell_wave_score1, sell_wave_score2, sell_wave_score3
- sell_jump_down_power, sell_diff_down_days, sell_jump_up_power, sell_diff_up_days

【结果字段】
- result: 1=盈利, -1=亏损, 0=持仓中
- profit: 盈亏金额
- profit_cg: 盈亏百分比
"""

RULE_EXTRACTION_PROMPT = """你是一个量化交易策略分析师。下面是一批历史交易的特征数据和实际结果。

{feature_descriptions}

## 历史交易数据

{training_data}

## 任务

分析上述交易中盈利和亏损交易的规律，输出一个 Python 函数 `judge_trade(features)` 用于判断新交易信号是否应该被拦截。

函数规范:
- 参数 `features` 是一个 dict，key 如 'buy_deg_ang42', 'buy_price_rank90' 等
- 返回 `True` 表示拦截（不执行这笔交易），`False` 表示放行
- 只拦截那些特征模式明显倾向于亏损的交易
- 如果特征不足以判断，宁可放行（返回 False）
- 规则要基于上面数据中观察到的规律，不要编造

输出格式:
```python
def judge_trade(features):
    \"\"\"
    基于历史数据训练的拦截规则。
    自动生成，请验证后再用于实盘。
    \"\"\"
    # 你的判断逻辑
    ...
```

只输出 Python 函数代码，不要输出其他内容。
"""

DIRECT_JUDGE_PROMPT = """你是一个量化交易裁判。判断以下交易信号是否应该执行。

{feature_descriptions}

## 待判断的交易信号

{signal_text}

## 任务

判断这笔交易是否应该放行。考虑:
1. 趋势方向是否健康（角度是否合适、是否过度延伸）
2. 价格位置是否合理（是否追高/抄底）
3. 波动率是否异常
4. 跳空缺口是否构成风险

输出 JSON 格式（不要输出其他内容）:
{{"approve": true/false, "confidence": 0.0-1.0, "reason": "一句话说明判断理由"}}
"""

BATCH_JUDGE_PROMPT = """你是一个量化交易裁判。判断以下 {count} 笔交易信号是否应该执行。

{feature_descriptions}

## 待判断的交易信号

{signals_text}

## 任务

对每一笔交易判断是否应该放行。输出 JSON 数组（不要输出其他内容）:
[
  {{"id": 0, "approve": true/false, "confidence": 0.0-1.0, "reason": "..."}},
  {{"id": 1, "approve": true/false, "confidence": 0.0-1.0, "reason": "..."}},
  ...
]
"""

STRATEGY_ANALYSIS_PROMPT = """你是一个量化交易策略分析师。下面是一个交易策略的回测结果。

## 策略概况
- 初始资金: {read_cash}
- 总交易笔数: {total_trades}
- 胜率: {win_rate:.1%}
- 总收益率: {total_return:.1%}
- 夏普比率: {sharpe_ratio}
- 最大回撤: {max_drawdown:.1%}

## 交易明细（前50笔）

{orders_summary}

## 任务

请分析：
1. **总体评价**: 这个策略表现如何？优缺点是什么？
2. **亏损共性**: 亏损交易有什么共同特征？是追高、抄底、还是波动率问题？
3. **盈利共性**: 盈利交易有什么共同特征？
4. **参数建议**: 买入/卖出因子参数如何优化？
5. **下一步**: 建议尝试什么改进方向？

输出 JSON 格式:
{{"summary": "...", "loss_patterns": "...", "win_patterns": "...",
  "parameter_suggestions": "...", "next_steps": "..."}}
"""


# ──────────────────────────────────────────────────────────
# ClaudeJudge
# ──────────────────────────────────────────────────────────

class ClaudeJudge(object):
    """
    Claude 驱动的交易裁判。

    Usage:
        # 模式1: 规则提取（适合全市场回测）
        judge = ClaudeJudge(mode="rule")
        judge.fit(orders_pd)  # 自动调 Claude 生成判断规则
        AbuUmpManager.append_user_ump(judge)
        AbuUmpManager.g_enable_user_ump = True

        # 模式2: 直接裁判（适合单股/小批量分析）
        judge = ClaudeJudge(mode="direct")
        AbuUmpManager.append_user_ump(judge)
        AbuUmpManager.g_enable_user_ump = True
    """

    def __init__(self, mode="rule", config=None):
        """
        :param mode: "rule" (提取规则本地执行) 或 "direct" (直调API)
        :param config: ClaudeConfig 实例，默认用环境变量
        """
        if mode not in ("rule", "direct"):
            raise ValueError("mode must be 'rule' or 'direct', got '{}'".format(mode))

        self.mode = mode
        self.config = config or ClaudeConfig()
        self._buy_ump = True
        self._is_fitted = False

        # 模式1: 存储 Claude 生成的判断规则
        self._rule_func = None       # Python 函数对象
        self._rule_code = None       # 规则源代码

        # 模式2: 特征哈希 → 判断结果 缓存
        self._cache = {}

        # 训练数据的摘要，用于 prompt
        self._training_summary = None

        # rule 模式立即加载默认规则（不需要 fit 也能用）
        if self.mode == "rule":
            self._set_default_rules()

    # ── UMP 兼容接口 ──────────────────────────────────

    @classmethod
    def class_unique_id(cls):
        """AbuUmpManager 要求: 返回唯一标识"""
        return "claude_judge"

    def is_buy_ump(self):
        """AbuUmpManager 要求: 是否用于买入裁判"""
        return self._buy_ump

    def predict(self, **ml_feature_dict):
        """
        AbuUmpManager 要求: 边裁接口。
        返回 EEdgeType.E_EEdge_TOP_LOSS (-1) 表示拦截，
        返回 EEdgeType.E_EEdge_NORMAL (0) 表示放行。
        """
        if not ml_feature_dict:
            return _EEdgeType.E_EEdge_NORMAL

        should_block = self._judge(ml_feature_dict)

        if should_block:
            return _EEdgeType.E_EEdge_TOP_LOSS
        return _EEdgeType.E_EEdge_NORMAL

    # ── 训练 / 预热 ──────────────────────────────────

    def fit(self, orders_pd, action_pd=None):
        """
        训练裁判:
        - mode="rule": 发送历史交易给 Claude → 生成判断规则代码 → 本地执行
        - mode="direct": 预热缓存（可选）
        """
        if orders_pd is None or len(orders_pd) == 0:
            logging.warning('ClaudeJudge.fit: orders_pd is empty, skip')
            return self

        if self.mode == "rule":
            self._extract_rules(orders_pd)
        elif self.mode == "direct":
            # 预热: 对前20条交易做批量判断，填充缓存
            self._warm_cache(orders_pd.head(20))

        self._is_fitted = True
        return self

    # ── 核心判断逻辑 ──────────────────────────────────

    def _judge(self, ml_feature_dict):
        """
        判断单笔交易是否应被拦截。
        返回 True = 拦截, False = 放行。
        """
        if self.mode == "rule" and self._rule_func is not None:
            return self._apply_rules(ml_feature_dict)
        elif self.mode == "direct":
            return self._judge_direct(ml_feature_dict)
        # 未训练 → 全部放行
        return False

    def _apply_rules(self, features):
        """
        用 Claude 生成的 Python 规则本地判断。
        安全执行，异常时放行。
        """
        if self._rule_func is None:
            return False
        try:
            result = self._rule_func(features)
            return bool(result)
        except Exception as e:
            logging.warning('ClaudeJudge: rule execution failed: {}'.format(e))
            return False

    def _judge_direct(self, features):
        """
        直调 Claude API 判断单笔交易。
        先查缓存（用特征哈希），miss 时才调 API。
        """
        fhash = self._hash_features(features)
        if fhash in self._cache:
            return self._cache[fhash]

        # 格式化特征为文本
        signal_text = self._format_single_signal(features)

        prompt = DIRECT_JUDGE_PROMPT.format(
            feature_descriptions=FEATURE_DESCRIPTIONS,
            signal_text=signal_text,
        )

        try:
            response = self._call_claude(prompt, max_tokens=256)
            result = json.loads(self._extract_json(response))
            should_block = not result.get("approve", True)
            self._cache[fhash] = should_block
            return should_block
        except Exception as e:
            logging.warning('ClaudeJudge: direct judge failed: {}'.format(e))
            return False  # 异常时放行

    def _judge_batch(self, features_list):
        """
        批量判断多笔交易（一次 API 调用）。
        """
        if not features_list:
            return []

        # 检查缓存
        results = []
        uncached = []
        uncached_indices = []

        for i, features in enumerate(features_list):
            fhash = self._hash_features(features)
            if fhash in self._cache:
                results.append((i, self._cache[fhash]))
            else:
                uncached.append(features)
                uncached_indices.append(i)

        if not uncached:
            return [r[1] for r in sorted(results, key=lambda x: x[0])]

        # 批量调 API
        signals_text = "\n---\n".join(
            "交易 #{}:\n{}".format(idx, self._format_single_signal(f))
            for idx, f in enumerate(uncached)
        )

        prompt = BATCH_JUDGE_PROMPT.format(
            count=len(uncached),
            feature_descriptions=FEATURE_DESCRIPTIONS,
            signals_text=signals_text,
        )

        try:
            response = self._call_claude(prompt, max_tokens=1024)
            batch_results = json.loads(self._extract_json(response))

            for jr in batch_results:
                idx = jr.get("id", 0)
                should_block = not jr.get("approve", True)
                if idx < len(uncached):
                    features = uncached[idx]
                    fhash = self._hash_features(features)
                    self._cache[fhash] = should_block
                    results.append((uncached_indices[idx], should_block))

        except Exception as e:
            logging.warning('ClaudeJudge: batch judge failed: {}'.format(e))
            # 异常时全部放行
            for idx in uncached_indices:
                results.append((idx, False))

        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    # ── 规则提取 (模式1) ──────────────────────────────

    def _extract_rules(self, orders_pd):
        """
        发送历史交易数据给 Claude，让它生成判断规则代码。
        """
        # 提取有结果的交易（盈利或亏损）
        df = orders_pd.copy()
        if 'result' in df.columns:
            df = df[df['result'] != 0]

        if len(df) == 0:
            logging.warning('ClaudeJudge: no settled trades for rule extraction')
            self._set_default_rules()
            return

        # 限制样本量（prompt 长度考虑）
        if len(df) > 80:
            # 分层抽样：保留全部亏损 + 部分盈利
            loss_df = df[df['result'] == -1] if 'result' in df.columns else pd.DataFrame()
            win_df = df[df['result'] == 1] if 'result' in df.columns else pd.DataFrame()
            if len(loss_df) >= 40:
                loss_df = loss_df.head(40)
            win_sample_n = min(len(win_df), 40)
            if win_sample_n > 0:
                win_df = win_df.head(win_sample_n)
            df = pd.concat([loss_df, win_df])

        # 提取特征列
        feature_cols = [c for c in df.columns if c.startswith(('buy_', 'sell_'))]
        if not feature_cols:
            logging.warning('ClaudeJudge: no feature columns found in orders_pd')
            self._set_default_rules()
            return

        # 构建训练数据文本
        # 选择最多 15 个关键特征
        key_features = [c for c in feature_cols
                        if any(p in c for p in ('deg_ang', 'price_rank', 'wave_score',
                                                 'atr_std', 'jump_'))]
        if len(key_features) > 15:
            key_features = key_features[:15]

        rows = []
        for idx, (_, row) in enumerate(df.iterrows()):
            profit_cg = row.get('profit_cg', row.get('profit', 0))
            result = row.get('result', 0)
            result_str = '盈利' if result == 1 else '亏损'
            feats = {c: row[c] for c in key_features if c in row and not pd.isna(row[c])}
            feats_str = ', '.join('{}={:.4f}'.format(k, v) for k, v in feats.items())
            rows.append('交易{} [{}] 盈亏={:.3f} | {}'.format(idx + 1, result_str, profit_cg, feats_str))

        training_data = '\n'.join(rows)

        prompt = RULE_EXTRACTION_PROMPT.format(
            feature_descriptions=FEATURE_DESCRIPTIONS,
            training_data=training_data,
        )

        try:
            response = self._call_claude(prompt, max_tokens=2048)
            code = self._extract_python_code(response)
            self._rule_code = code
            self._compile_rule(code)
        except Exception as e:
            logging.warning('ClaudeJudge: rule extraction failed: {}'.format(e))
            self._set_default_rules()

    def _compile_rule(self, code):
        """编译 Claude 生成的规则代码为可执行函数"""
        namespace = {}
        try:
            exec(code, namespace)
            if 'judge_trade' in namespace:
                self._rule_func = namespace['judge_trade']
                logging.info('ClaudeJudge: rule compiled successfully')
            else:
                logging.warning('ClaudeJudge: judge_trade not found in generated code')
                self._set_default_rules()
        except Exception as e:
            logging.warning('ClaudeJudge: rule compile failed: {}'.format(e))
            self._set_default_rules()

    def _set_default_rules(self):
        """
        默认规则: 拦截极端追高 + 趋势向下 + 高波动的组合。
        这是保守的 fallback，确保没有 Claude 也能用。
        """
        default_code = '''
def judge_trade(features):
    """默认规则 — 拦截明显的风险信号"""
    score = 0

    # 追高检测
    price_rank = features.get('buy_price_rank90', features.get('buy_price_rank60', 0.5))
    if price_rank is None:
        price_rank = 0.5

    # 趋势检测
    trend_short = features.get('buy_deg_ang21', features.get('buy_deg_ang42', 0))
    trend_long = features.get('buy_deg_ang60', features.get('buy_deg_ang252', 0))
    if trend_short is None:
        trend_short = 0
    if trend_long is None:
        trend_long = 0

    # 波动率
    atr = features.get('buy_atr_std', 0.5)
    if atr is None:
        atr = 0.5

    # 规则: 追高 + 短期趋势转弱
    if price_rank > 0.85 and trend_short < 3:
        score += 3

    # 规则: 长期下跌趋势
    if trend_long < -5:
        score += 2

    # 规则: 高波动 + 追高
    if atr > 0.8 and price_rank > 0.9:
        score += 3

    # 规则: 缺口向下且距离近
    jump_down = features.get('buy_jump_down_power', features.get('buy_diff_down_days', 999))
    if jump_down is not None and isinstance(jump_down, (int, float)):
        diff_days = features.get('buy_diff_down_days', 999)
        if diff_days is not None and diff_days < 10 and jump_down < -2:
            score += 3

    return score >= 5  # 5分及以上拦截
'''
        self._rule_code = default_code
        self._compile_rule(default_code)

    # ── 缓存相关 ──────────────────────────────────────

    def _warm_cache(self, orders_pd):
        """预热缓存: 批量判断历史交易"""
        if orders_pd is None or len(orders_pd) == 0:
            return

        feature_cols = [c for c in orders_pd.columns
                        if c.startswith(('buy_', 'sell_'))]
        if not feature_cols:
            return

        features_list = []
        for _, row in orders_pd.iterrows():
            feats = {c: row[c] for c in feature_cols
                     if c in row and not pd.isna(row[c])}
            if feats:
                features_list.append(feats)

        if features_list:
            self._judge_batch(features_list)

    def _hash_features(self, features):
        """对特征字典做确定性哈希（用于缓存 key）"""
        # 只保留数值特征，四舍五入到 3 位小数
        stable = {}
        for k, v in sorted(features.items()):
            if isinstance(v, (int, float, np.integer, np.floating)):
                stable[k] = round(float(v), 3)
        raw = json.dumps(stable, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def clear_cache(self):
        """清除判断缓存"""
        self._cache.clear()

    # ── 格式化辅助 ─────────────────────────────────────

    def _format_single_signal(self, features):
        """格式化单笔交易信号为可读文本"""
        # 分组展示
        lines = []
        trend_keys = [k for k in sorted(features) if 'deg_ang' in k]
        price_keys = [k for k in sorted(features) if 'price_rank' in k]
        wave_keys = [k for k in sorted(features) if 'wave_score' in k or 'atr' in k]
        jump_keys = [k for k in sorted(features) if 'jump' in k]
        other_keys = [k for k in sorted(features)
                      if k not in trend_keys + price_keys + wave_keys + jump_keys]

        if trend_keys:
            lines.append('趋势角度: ' + ', '.join(
                '{}={:.2f}°'.format(k, features[k]) for k in trend_keys if not pd.isna(features[k])))
        if price_keys:
            lines.append('价格位置: ' + ', '.join(
                '{}={:.3f}'.format(k, features[k]) for k in price_keys if not pd.isna(features[k])))
        if wave_keys:
            lines.append('波动率: ' + ', '.join(
                '{}={:.3f}'.format(k, features[k]) for k in wave_keys if not pd.isna(features[k])))
        if jump_keys:
            lines.append('跳空: ' + ', '.join(
                '{}={:.3f}'.format(k, features[k]) for k in jump_keys if not pd.isna(features[k])))
        if other_keys:
            lines.append('其他: ' + ', '.join(
                '{}={:.3f}'.format(k, features[k]) for k in other_keys[:5] if not pd.isna(features[k])))

        return '\n'.join(lines) if lines else str(features)

    # ── Claude API 调用 ────────────────────────────────

    def _call_claude(self, prompt, max_tokens=None):
        """
        调用 Claude API。
        使用 requests 直接调 HTTP API（避免依赖 anthropic SDK）。
        """
        import requests

        self.config.validate()

        headers = {
            'x-api-key': self.config.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }

        payload = {
            'model': self.config.model,
            'max_tokens': max_tokens or self.config.max_tokens,
            'temperature': self.config.temperature,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
        }

        try:
            resp = requests.post(
                '{}/v1/messages'.format(self.config.base_url.rstrip('/')),
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # 提取响应文本
            content = data.get('content', [])
            if isinstance(content, list) and len(content) > 0:
                return content[0].get('text', '')
            return str(content)
        except Exception as e:
            logging.error('ClaudeJudge: API call failed: {}'.format(e))
            raise

    def _extract_json(self, text):
        """从 Claude 响应中提取 JSON（处理 markdown code block）"""
        text = text.strip()
        # 尝试提取 ```json ... ``` 块
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 尝试提取 [...] 或 {...}
        m = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text

    def _extract_python_code(self, text):
        """从 Claude 响应中提取 Python 代码"""
        text = text.strip()
        m = re.search(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # 没有 code block → 尝试提取整个响应
        return text

    # ── 状态查询 ───────────────────────────────────────

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def rule_code(self):
        """查看当前使用的规则代码"""
        return self._rule_code

    @property
    def cache_size(self):
        return len(self._cache)

    def __repr__(self):
        status = 'fitted' if self._is_fitted else 'not fitted'
        return 'ClaudeJudge(mode={}, {}, cache={})'.format(self.mode, status, self.cache_size)


# ──────────────────────────────────────────────────────────
# 便捷工厂函数
# ──────────────────────────────────────────────────────────

class ClaudeSellJudge(ClaudeJudge):
    """卖出裁判 — 用于判断卖出信号"""
    def is_buy_ump(self):
        return False
