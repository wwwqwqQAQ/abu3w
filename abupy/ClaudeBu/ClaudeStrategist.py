# -*- encoding:utf-8 -*-
"""
    ClaudeStrategist — 回测后策略分析

    把回测结果发给 Claude，让它做深度分析报告:
    - 盈亏模式识别
    - 参数优化建议
    - 下一轮改进方向

    用法:
        from ClaudeBu import ClaudeStrategist
        analyst = ClaudeStrategist()
        report = analyst.analyze(orders_pd, action_pd, capital, benchmark)
        print(report['summary'])
"""
from __future__ import absolute_import, print_function, division

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

try:
    from .ClaudeJudge import ClaudeJudge, FEATURE_DESCRIPTIONS
except ImportError:
    from ClaudeJudge import ClaudeJudge, FEATURE_DESCRIPTIONS

__author__ = 'ClaudeBu'
__all__ = ['ClaudeStrategist']


class ClaudeStrategist(object):
    """
    回测后策略分析器。

    Usage:
        strategist = ClaudeStrategist()
        report = strategist.analyze(orders_pd, action_pd, capital, benchmark)
        # report = {
        #     'summary': '...',
        #     'loss_patterns': '...',
        #     'win_patterns': '...',
        #     'parameter_suggestions': '...',
        #     'next_steps': '...',
        # }
    """

    def __init__(self, config=None):
        """
        :param config: ClaudeConfig 实例，默认用环境变量
        """
        self.config = config or ClaudeConfig().for_strategy()

    def analyze(self, orders_pd, action_pd=None, capital=None, benchmark=None,
                buy_factors=None, sell_factors=None):
        """
        对回测结果做全面分析。

        :param orders_pd: 订单 DataFrame（abu 回测输出的 orders_pd）
        :param action_pd: 动作 DataFrame（可选）
        :param capital: AbuCapital 实例（可选，用于资金曲线）
        :param benchmark: AbuBenchmark 实例（可选，用于基准对比）
        :param buy_factors: 使用的买入因子配置（可选）
        :param sell_factors: 使用的卖出因子配置（可选）
        :return: dict, keys: summary, loss_patterns, win_patterns,
                 parameter_suggestions, next_steps
        """
        if orders_pd is None or len(orders_pd) == 0:
            return {'summary': '无交易数据', 'loss_patterns': '', 'win_patterns': '',
                    'parameter_suggestions': '', 'next_steps': ''}

        # 计算关键指标
        metrics = self._compute_metrics(orders_pd, capital, benchmark)

        # 构建 prompt
        prompt = self._build_analysis_prompt(
            orders_pd, metrics, buy_factors, sell_factors
        )

        # 调 Claude
        try:
            response = self._call_claude(prompt)
            result = json.loads(self._extract_json(response))
            # 确保所有 key 都存在
            for key in ('summary', 'loss_patterns', 'win_patterns',
                        'parameter_suggestions', 'next_steps'):
                if key not in result:
                    result[key] = ''
            return result
        except Exception as e:
            logging.error('ClaudeStrategist: analysis failed: {}'.format(e))
            return {
                'summary': '分析失败: {}'.format(e),
                'loss_patterns': '', 'win_patterns': '',
                'parameter_suggestions': '', 'next_steps': '',
            }

    def quick_scan(self, orders_pd, top_n=10):
        """
        快速扫描 — 只分析最差/最好的交易。

        :return: dict with 'worst_trades_analysis' and 'best_trades_analysis'
        """
        if orders_pd is None or len(orders_pd) == 0:
            return {}

        # 按盈亏排序
        if 'profit_cg' in orders_pd.columns:
            sorted_orders = orders_pd.sort_values('profit_cg')
        elif 'profit' in orders_pd.columns:
            sorted_orders = orders_pd.sort_values('profit')
        else:
            return {}

        worst = sorted_orders.head(top_n)
        best = sorted_orders.tail(top_n)

        # 构建紧凑 prompt
        worst_text = self._format_orders_compact(worst, '最差')
        best_text = self._format_orders_compact(best, '最好')

        prompt = textwrap.dedent('''\
        快速分析以下交易:

        ## 亏损最大的 {n} 笔
        {worst}

        ## 盈利最大的 {n} 笔
        {best}

        用一句话总结亏损交易的根本原因，一句话总结盈利交易的成功因素。
        输出 JSON: {{"loss_root_cause": "...", "win_key_factor": "..."}}
        ''').format(n=top_n, worst=worst_text, best=best_text)

        try:
            response = self._call_claude(prompt)
            return json.loads(self._extract_json(response))
        except Exception as e:
            return {'loss_root_cause': str(e), 'win_key_factor': ''}

    def suggest_parameters(self, orders_pd, buy_factors, sell_factors):
        """
        参数优化建议 — 让 Claude 建议更好的因子参数。

        :return: dict with 'buy_suggestions' and 'sell_suggestions'
        """
        metrics = self._compute_metrics(orders_pd)

        prompt = textwrap.dedent('''\
        当前策略表现:
        - 胜率: {win_rate:.1%}
        - 总收益: {total_return:.1%}
        - 夏普比率: {sharpe_ratio}
        - 最大回撤: {max_drawdown:.1%}

        当前买入因子: {buy_factors}
        当前卖出因子: {sell_factors}

        基于以上表现，建议参数调整方向。
        输出 JSON:
        {{
          "buy_suggestions": "买入因子参数调整建议",
          "sell_suggestions": "卖出因子参数调整建议"
        }}
        ''').format(
            win_rate=metrics.get('win_rate', 0),
            total_return=metrics.get('total_return', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 'N/A'),
            max_drawdown=metrics.get('max_drawdown', 0),
            buy_factors=buy_factors or '未提供',
            sell_factors=sell_factors or '未提供',
        )

        try:
            response = self._call_claude(prompt)
            return json.loads(self._extract_json(response))
        except Exception as e:
            return {'buy_suggestions': str(e), 'sell_suggestions': ''}

    # ── 内部方法 ────────────────────────────────────────

    def _compute_metrics(self, orders_pd, capital=None, benchmark=None):
        """计算回测核心指标"""
        metrics = {}

        # 基础统计
        total = len(orders_pd)
        metrics['total_trades'] = total

        if 'result' in orders_pd.columns:
            settled = orders_pd[orders_pd['result'] != 0]
            if len(settled) > 0:
                wins = len(settled[settled['result'] == 1])
                metrics['win_rate'] = wins / len(settled)
                metrics['settled_trades'] = len(settled)
            else:
                metrics['win_rate'] = 0
                metrics['settled_trades'] = 0

            metrics['avg_profit'] = settled['profit'].mean() if 'profit' in settled.columns else 0
            metrics['total_profit'] = settled['profit'].sum() if 'profit' in settled.columns else 0
        else:
            metrics['win_rate'] = 0
            metrics['settled_trades'] = 0
            metrics['avg_profit'] = 0
            metrics['total_profit'] = 0

        # 收益率（如果有 capital 信息）
        if capital is not None:
            try:
                cap_pd = capital.capital_pd
                if cap_pd is not None and 'capital_blance' in cap_pd.columns:
                    metrics['read_cash'] = capital.read_cash
                    final_value = cap_pd['capital_blance'].iloc[-1]
                    metrics['total_return'] = (final_value - capital.read_cash) / capital.read_cash
                    # 简易最大回撤
                    cummax = cap_pd['capital_blance'].cummax()
                    drawdown = (cap_pd['capital_blance'] - cummax) / cummax
                    metrics['max_drawdown'] = drawdown.min()
                    # 简易夏普
                    returns = cap_pd['capital_blance'].pct_change().dropna()
                    if len(returns) > 0 and returns.std() > 0:
                        metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
                    else:
                        metrics['sharpe_ratio'] = 'N/A'
                else:
                    metrics['read_cash'] = 'N/A'
                    metrics['total_return'] = 'N/A'
                    metrics['max_drawdown'] = 'N/A'
                    metrics['sharpe_ratio'] = 'N/A'
            except Exception:
                metrics['read_cash'] = 'N/A'
                metrics['total_return'] = 'N/A'
                metrics['max_drawdown'] = 'N/A'
                metrics['sharpe_ratio'] = 'N/A'
        else:
            metrics['read_cash'] = 'N/A'
            metrics['total_return'] = 'N/A'
            metrics['max_drawdown'] = 'N/A'
            metrics['sharpe_ratio'] = 'N/A'

        return metrics

    def _build_analysis_prompt(self, orders_pd, metrics, buy_factors, sell_factors):
        """构建策略分析 prompt"""
        # 格式化交易明细
        orders_text = self._format_orders_summary(orders_pd)

        prompt = textwrap.dedent('''\
        你是一个量化交易策略分析师。下面是一个交易策略的回测结果。

        {feature_descriptions}

        ## 策略概况
        - 总交易笔数: {total_trades}
        - 已结算交易: {settled_trades}
        - 胜率: {win_rate}
        - 总盈亏: {total_profit}
        - 收益率: {total_return}
        - 夏普比率: {sharpe_ratio}
        - 最大回撤: {max_drawdown}
        - 买入因子: {buy_factors}
        - 卖出因子: {sell_factors}

        ## 交易明细
        {orders_text}

        ## 任务

        请分析：
        1. **总体评价**: 这个策略表现如何？优缺点是什么？(100字以内)
        2. **亏损共性**: 亏损交易有什么共同特征？是追高、抄底、还是波动率问题？(150字以内)
        3. **盈利共性**: 盈利交易有什么共同特征？(150字以内)
        4. **参数建议**: 买入/卖出因子参数如何优化？(150字以内)
        5. **下一步**: 建议尝试什么改进方向？(100字以内)

        输出 JSON 格式（要有实质内容，不要空话）:
        {{"summary": "...", "loss_patterns": "...", "win_patterns": "...",
          "parameter_suggestions": "...", "next_steps": "..."}}
        ''').format(
            feature_descriptions=FEATURE_DESCRIPTIONS,
            total_trades=metrics.get('total_trades', 0),
            settled_trades=metrics.get('settled_trades', 0),
            win_rate='{:.1%}'.format(metrics.get('win_rate', 0)) if isinstance(metrics.get('win_rate'), float) else 'N/A',
            total_profit='{:.2f}'.format(metrics.get('total_profit', 0)),
            total_return='{:.1%}'.format(metrics.get('total_return', 0)) if isinstance(metrics.get('total_return'), float) else 'N/A',
            sharpe_ratio=metrics.get('sharpe_ratio', 'N/A'),
            max_drawdown='{:.1%}'.format(metrics.get('max_drawdown', 0)) if isinstance(metrics.get('max_drawdown'), float) else 'N/A',
            buy_factors=buy_factors or '默认突破因子',
            sell_factors=sell_factors or '默认ATR止损因子',
            orders_text=orders_text,
        )
        return prompt

    def _format_orders_summary(self, orders_pd, max_rows=50):
        """格式化交易汇总表"""
        df = orders_pd.copy()

        # 选关键列
        cols = []
        for c in ['symbol', 'buy_date', 'buy_price', 'sell_price', 'profit',
                   'profit_cg', 'result', 'buy_factor', 'sell_type']:
            if c in df.columns:
                cols.append(c)
        # 添加特征列
        feat_cols = [c for c in df.columns
                     if any(p in c for p in ('deg_ang42', 'price_rank90', 'atr_std'))]
        cols.extend(feat_cols[:3])

        df = df[cols] if cols else df

        if len(df) > max_rows:
            # 分层抽样
            if 'result' in df.columns:
                loss = df[df['result'] == -1].head(max_rows // 2)
                win = df[df['result'] == 1].head(max_rows // 2)
                df = pd.concat([loss, win])
            else:
                df = df.head(max_rows)

        return df.to_string(max_rows=max_rows)

    def _format_orders_compact(self, orders_pd, label):
        """紧凑格式化几笔交易"""
        if orders_pd is None or len(orders_pd) == 0:
            return '无'

        lines = ['{} {} 笔:'.format(label, len(orders_pd))]
        for idx, (_, row) in enumerate(orders_pd.iterrows()):
            profit = row.get('profit_cg', row.get('profit', 0))
            symbol = row.get('symbol', row.get('buy_symbol', '?'))
            result = '盈' if row.get('result', 0) == 1 else '亏'
            lines.append('  {}. {} {} {:.3f}'.format(idx + 1, symbol, result, profit))
        return '\n'.join(lines)

    # ── Claude API 调用 ────────────────────────────────

    def _call_claude(self, prompt):
        """调用 Claude API (复用 ClaudeJudge 的实现)"""
        import requests

        self.config.validate()
        if not hasattr(self, '_judge_for_api'):
            self._judge_for_api = ClaudeJudge(mode="direct", config=self.config)

        return self._judge_for_api._call_claude(prompt, max_tokens=self.config.max_tokens)

    def _extract_json(self, text):
        """从响应提取 JSON"""
        text = text.strip()
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text
