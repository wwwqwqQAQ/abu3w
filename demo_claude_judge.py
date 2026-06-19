#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
    完整演示: ClaudeJudge 替换 abu UMP 裁判系统
    对比: 不带裁判 vs 带 ClaudeJudge 默认规则
"""
from __future__ import print_function, absolute_import, division
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import abupy
from abupy import (AbuFactorBuyBreak, AbuFactorAtrNStop,
                   AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop,
                   AbuBenchmark, AbuCapital, ABuPickTimeExecute,
                   AbuMetricsBase)
from abupy.UmpBu.ABuUmpManager import AbuUmpManager, clear_user_ump, append_user_ump

from abupy.ClaudeBu import ClaudeJudge, ClaudeSellJudge, ClaudeStrategist

print('=' * 60)
print('  ClaudeJudge 完整演示')
print('  abu v{} | ClaudeBu v0.1.0'.format(abupy.__version__))
print('=' * 60)

# ── 1. 初始化沙盒环境 ──────────────────────────────
abupy.env.enable_example_env_ipython()
print('\n1. 沙盒环境就绪 (离线数据模式)')

# ── 2. 策略配置 ────────────────────────────────────
symbols = ['usTSLA', 'usAAPL', 'usGOOG', 'usBIDU']
buy_factors = [
    {'xd': 60, 'class': AbuFactorBuyBreak},
    {'xd': 42, 'class': AbuFactorBuyBreak},
]
sell_factors = [
    {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
    {'pre_atr_n': 1.0, 'class': AbuFactorPreAtrNStop},
    {'close_atr_n': 1.5, 'class': AbuFactorCloseAtrNStop},
]

print('2. 策略:')
print('   标的: {}'.format(', '.join(symbols)))
print('   买入: 60日突破 + 42日突破')
print('   卖出: ATR止损止盈 + 暴跌止损 + 保护止盈')

# ── 3. 回测: 不带裁判 (baseline) ────────────────────
print('\n' + '=' * 60)
print('  回测 A: 无裁判 (原始 abu)')
print('=' * 60)

benchmark = AbuBenchmark()
capital = AbuCapital(1000000, benchmark)
orders_pd_a, action_pd_a, _ = ABuPickTimeExecute.do_symbols_with_same_factors(
    symbols, benchmark, buy_factors, sell_factors, capital, show=False
)

if orders_pd_a is not None and len(orders_pd_a) > 0:
    settled = orders_pd_a[orders_pd_a['result'] != 0] if 'result' in orders_pd_a.columns else orders_pd_a
    wins = len(settled[settled['result'] == 1]) if 'result' in settled.columns else 0
    total = len(settled)
    win_rate_a = wins / total if total > 0 else 0
    profit_a = settled['profit'].sum() if 'profit' in settled.columns else 0
    print('  交易笔数: {} | 胜率: {:.1%} | 总盈亏: ${:,.2f}'.format(total, win_rate_a, profit_a))

    # 显示前5笔
    cols = [c for c in ['symbol', 'buy_date', 'profit', 'result'] if c in orders_pd_a.columns]
    if cols:
        print('\n  前5笔交易:')
        print(orders_pd_a[cols].head(5).to_string())
else:
    print('  无交易信号!')
    orders_pd_a = None
    win_rate_a = 0
    profit_a = 0

# ── 4. 注册 ClaudeJudge ─────────────────────────────
print('\n' + '=' * 60)
print('  回测 B: 启用 ClaudeJudge (默认规则)')
print('=' * 60)

judge = ClaudeJudge(mode='rule')
print('  裁判规则: {} 行 Python 代码'.format(len(judge.rule_code.strip().split('\n'))))

# 注册到 UMP
clear_user_ump()
append_user_ump(judge, check=False)
AbuUmpManager.g_enable_user_ump = True
print('  已注册到 AbuUmpManager')

# ── 5. 回测: 带裁判 ────────────────────────────────
benchmark2 = AbuBenchmark()
capital2 = AbuCapital(1000000, benchmark2)
orders_pd_b, action_pd_b, _ = ABuPickTimeExecute.do_symbols_with_same_factors(
    symbols, benchmark2, buy_factors, sell_factors, capital2, show=False
)

if orders_pd_b is not None and len(orders_pd_b) > 0:
    settled_b = orders_pd_b[orders_pd_b['result'] != 0] if 'result' in orders_pd_b.columns else orders_pd_b
    wins_b = len(settled_b[settled_b['result'] == 1]) if 'result' in settled_b.columns else 0
    total_b = len(settled_b)
    win_rate_b = wins_b / total_b if total_b > 0 else 0
    profit_b = settled_b['profit'].sum() if 'profit' in settled_b.columns else 0
    print('  交易笔数: {} | 胜率: {:.1%} | 总盈亏: ${:,.2f}'.format(total_b, win_rate_b, profit_b))
else:
    print('  无交易信号! (可能被裁判拦截了所有信号)')
    orders_pd_b = None
    total_b, win_rate_b, profit_b = 0, 0, 0

# ── 6. 对比 ─────────────────────────────────────────
print('\n' + '=' * 60)
print('  对比')
print('=' * 60)

if orders_pd_a is not None:
    reduction = total - total_b if 'total' in dir() else 0
    print('  交易笔数: {} → {} (减少 {})'.format(
        len(orders_pd_a), len(orders_pd_b) if orders_pd_b is not None else 0,
        len(orders_pd_a) - (len(orders_pd_b) if orders_pd_b is not None else 0)))
    print('  胜率:      {:.1%} → {:.1%}'.format(win_rate_a, win_rate_b))
    print('  总盈亏:    ${:,.2f} → ${:,.2f}'.format(profit_a, profit_b))

    # 被拦截的交易特征
    if orders_pd_a is not None and orders_pd_b is not None:
        # 找被拦截的（在A中有但B中没有）
        print('\n  ClaudeJudge 拦截统计:')
        blocked_count = 0
        for _, row_a in orders_pd_a.iterrows():
            features = {}
            for c in orders_pd_a.columns:
                if any(p in str(c) for p in ('deg_ang', 'price_rank', 'atr_std', 'wave_score', 'jump')):
                    if c in row_a.index and not pd.isna(row_a[c]):
                        features[str(c)] = row_a[c]
            if features:
                blocked = judge._apply_rules(features)
                if blocked:
                    blocked_count += 1
        print('  理论上应拦截: {} 笔 (默认规则)'.format(blocked_count))
else:
    print('  无基准数据可对比')

# ── 7. 策略分析 (如果有 API Key) ──────────────────────
print('\n' + '=' * 60)
print('  策略分析 (ClaudeStrategist)')
print('=' * 60)

import os
if os.environ.get('ANTHROPIC_API_KEY') and orders_pd_a is not None:
    strategist = ClaudeStrategist()
    print('  正在调 Claude 分析...')
    report = strategist.quick_scan(orders_pd_a, top_n=5)
    print('  亏损原因: {}'.format(report.get('loss_root_cause', 'N/A')))
    print('  盈利因素: {}'.format(report.get('win_key_factor', 'N/A')))
else:
    print('  跳过 (需要 ANTHROPIC_API_KEY 环境变量)')
    print('  设置: export ANTHROPIC_API_KEY="sk-ant-..."')

# ── 8. ClaudeJudge 规则训练演示 ──────────────────────
print('\n' + '=' * 60)
print('  规则训练 (judge.fit)')
print('=' * 60)

if orders_pd_a is not None and len(orders_pd_a) > 5:
    print('  输入: {} 笔历史交易'.format(len(orders_pd_a)))
    print('  调用: judge.fit(orders_pd)')

    if os.environ.get('ANTHROPIC_API_KEY'):
        judge2 = ClaudeJudge(mode='rule')
        print('  正在调 Claude 分析历史交易...')
        judge2.fit(orders_pd_a)
        print('  ✓ 新规则已生成 ({} 行)'.format(len(judge2.rule_code.strip().split('\n'))))
        print('\n  新规则代码:')
        for line in judge2.rule_code.strip().split('\n')[:20]:
            print('   | ' + line)
    else:
        print('  跳过 API 调用 (需要 ANTHROPIC_API_KEY)')
        print('  本地演示: 默认规则已足够过滤明显风险信号')
else:
    print('  无足够历史数据')

# ── 9. 清理 ─────────────────────────────────────────
clear_user_ump()
AbuUmpManager.g_enable_user_ump = False

print('\n' + '=' * 60)
print('  演示完成')
print('=' * 60)

# 使用方式速查
print('''
┌─────────────────────────────────────────────────────┐
│  快速集成 (3行代码):                                │
│                                                     │
│  from abupy.ClaudeBu import ClaudeJudge             │
│  judge = ClaudeJudge(mode="rule")                   │
│  AbuUmpManager.append_user_ump(judge)               │
│  AbuUmpManager.g_enable_user_ump = True             │
│                                                     │
│  → 所有后续回测自动过滤风险信号                     │
└─────────────────────────────────────────────────────┘
''')
