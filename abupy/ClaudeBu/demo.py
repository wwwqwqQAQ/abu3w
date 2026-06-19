#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
    ClaudeBu 完整演示
    展示 ClaudeJudge 如何替换 abu 的 UMP 裁判系统

    运行: python3 demo_claude_bu.py

    前置: export ANTHROPIC_API_KEY="sk-ant-..."  (可选，没设也能用默认规则)
"""
from __future__ import print_function, absolute_import, division

import sys
import os
# 确保 abu 根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings('ignore')

# ── 1. 初始化 abu ────────────────────────────────
import abupy
abupy.env.enable_example_env_ipython()
print('✓ abu 沙盒环境就绪')

# ── 2. 导入因子 ───────────────────────────────────
from abupy import (AbuFactorBuyBreak, AbuFactorSellBreak,
                   AbuFactorAtrNStop, AbuFactorPreAtrNStop,
                   AbuFactorCloseAtrNStop,
                   AbuBenchmark, AbuCapital,
                   ABuPickTimeExecute, AbuMetricsBase)

# ── 3. 导入 ClaudeBu ──────────────────────────────
from abupy.ClaudeBu import ClaudeJudge, ClaudeStrategist
print('✓ ClaudeBu 已加载')

# ── 4. 配置因子 ───────────────────────────────────
buy_factors = [
    {'xd': 60, 'class': AbuFactorBuyBreak},
    {'xd': 42, 'class': AbuFactorBuyBreak},
]

sell_factors = [
    {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
    {'pre_atr_n': 1.0, 'class': AbuFactorPreAtrNStop},
    {'close_atr_n': 1.5, 'class': AbuFactorCloseAtrNStop},
]

print('策略配置:')
print('  买入: 60日突破 + 42日突破')
print('  卖出: ATR动态止损/止盈 + 暴跌止损 + 保护止盈')

# ── 5. 方式A: 不用裁判，直接跑 ──────────────────────
print('\n' + '='*60)
print('  方式A: 无裁判（原始回测）')
print('='*60)

benchmark = AbuBenchmark()
capital = AbuCapital(1000000, benchmark)

orders_pd_a, action_pd_a, _ = ABuPickTimeExecute.do_symbols_with_same_factors(
    ['usTSLA'], benchmark, buy_factors, sell_factors, capital, show=False
)

if orders_pd_a is not None and len(orders_pd_a) > 0:
    metrics_a = AbuMetricsBase(orders_pd_a, action_pd_a, capital, benchmark)
    metrics_a.fit_metrics_order()
    settled = orders_pd_a[orders_pd_a['result'] != 0] if 'result' in orders_pd_a.columns else orders_pd_a
    wins = len(settled[settled['result'] == 1]) if 'result' in settled.columns else 0
    total = len(settled)
    win_rate = wins / total if total > 0 else 0
    total_profit = settled['profit'].sum() if 'profit' in settled.columns else 0
    print('  交易笔数: {} | 胜率: {:.1%} | 总盈亏: {:.2f}'.format(total, win_rate, total_profit))
else:
    print('  无交易信号生成')
    orders_pd_a = None

# ── 6. 方式B: 启用 ClaudeJudge（默认规则）─────────────
print('\n' + '='*60)
print('  方式B: ClaudeJudge 规则模式（默认规则）')
print('='*60)

judge = ClaudeJudge(mode='rule')
print('  当前规则（前5行）:')
for line in judge.rule_code.strip().split('\n')[:5]:
    print('    ' + line)

# 注意: 完整的裁判集成需要 abu 的 UMP 框架参与
# 这里演示核心判断逻辑
print('\n  演示裁判对交易信号的判断:')

# 模拟几个交易信号的特征
test_signals = [
    {
        'name': '追高信号',
        'features': {'buy_price_rank90': 0.92, 'buy_deg_ang21': 2.0,
                     'buy_deg_ang60': 15.0, 'buy_atr_std': 0.6},
    },
    {
        'name': '下跌趋势信号',
        'features': {'buy_price_rank90': 0.3, 'buy_deg_ang21': -8.0,
                     'buy_deg_ang60': -12.0, 'buy_atr_std': 0.7},
    },
    {
        'name': '健康回调信号',
        'features': {'buy_price_rank90': 0.45, 'buy_deg_ang21': 5.0,
                     'buy_deg_ang60': 12.0, 'buy_atr_std': 0.35},
    },
    {
        'name': '高波动追高',
        'features': {'buy_price_rank90': 0.95, 'buy_deg_ang21': -1.0,
                     'buy_deg_ang60': -5.0, 'buy_atr_std': 1.2},
    },
]

for sig in test_signals:
    result = judge.predict(**sig['features'])
    verdict = '🔴 拦截' if result == -1 else '🟢 放行'
    print('  {} → {}'.format(sig['name'], verdict))

# ── 7. 方式C: Claude 策略分析（需要 API Key）───────────
print('\n' + '='*60)
print('  方式C: ClaudeStrategist 策略分析')
print('='*60)

import os
if os.environ.get('ANTHROPIC_API_KEY'):
    strategist = ClaudeStrategist()
    if orders_pd_a is not None:
        print('  正在调 Claude 分析回测结果...')
        report = strategist.quick_scan(orders_pd_a, top_n=5)
        print('  亏损原因: {}'.format(report.get('loss_root_cause', 'N/A')))
        print('  盈利因素: {}'.format(report.get('win_key_factor', 'N/A')))
    else:
        print('  无回测数据，跳过')
else:
    print('  未设置 ANTHROPIC_API_KEY，跳过 Claude API 调用')
    print('  设置方法: export ANTHROPIC_API_KEY="sk-ant-..."')

# ── 8. 集成到 UMP 的方法 ────────────────────────────
print('\n' + '='*60)
print('  如何集成到 abu 回测中')
print('='*60)
print('''
from abupy import AbuUmpManager
from ClaudeBu import ClaudeJudge

# 1. 创建裁判
judge = ClaudeJudge(mode="rule")

# 2. 如果有历史回测数据，训练它
# judge.fit(historical_orders_pd)  # → 调 Claude 提取专属规则

# 3. 注册到 abu
AbuUmpManager.append_user_ump(judge)
AbuUmpManager.g_enable_user_ump = True

# 4. 之后所有回测自动走 ClaudeJudge 过滤交易信号
# abu.run_loop_back(read_cash=1000000, ...)
''')

print('═══ ClaudeBu 演示完成 ═══')
