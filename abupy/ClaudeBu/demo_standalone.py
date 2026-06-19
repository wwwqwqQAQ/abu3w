#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
    ClaudeBu 独立演示（不依赖 abu 完整安装）
    展示 ClaudeJudge 裁判系统全部功能
"""
from __future__ import print_function

import sys
import os

# 把 ClaudeBu 目录加入 sys.path，这样就可以直接 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from ClaudeConfig import ClaudeConfig
from ClaudeJudge import ClaudeJudge, ClaudeSellJudge, _EEdgeType
from ClaudeStrategist import ClaudeStrategist

print('═══ ClaudeBu 独立演示 ═══\n')

# ── 1. 配置 ────────────────────────────────────────
config = ClaudeConfig(api_key=os.environ.get('ANTHROPIC_API_KEY', 'demo-key'))
print('1. 配置: {}'.format(config))

# ── 2. 裁判接口 ─────────────────────────────────────
print('\n2. ClaudeJudge — UMP 兼容接口:')
judge = ClaudeJudge(mode='rule')
print('   class_unique_id(): {}'.format(judge.class_unique_id()))
print('   is_buy_ump(): {}'.format(judge.is_buy_ump()))
print('   is_fitted: {}'.format(judge.is_fitted))

# ── 3. 默认规则 ─────────────────────────────────────
print('\n3. 默认拦截规则（无需 API Key，立即可用）:')
for line in judge.rule_code.strip().split('\n'):
    print('   | ' + line)

# ── 4. 模拟交易信号判断 ──────────────────────────────
print('\n4. 模拟 abu 交易信号判断:')

signals = [
    ('追高 + 趋势弱 + 高波动 → 应拦截', dict(
        buy_price_rank90=0.95, buy_deg_ang21=1.0,
        buy_deg_ang60=-6.0, buy_atr_std=1.0,
        buy_jump_down_power=-3.0, buy_diff_down_days=3,
    )),
    ('下跌趋势抄底 → 应拦截', dict(
        buy_price_rank90=0.15, buy_deg_ang21=-10.0,
        buy_deg_ang60=-15.0, buy_atr_std=0.5,
    )),
    ('追高 + 向下跳空 → 应拦截', dict(
        buy_price_rank90=0.90, buy_deg_ang21=3.0,
        buy_deg_ang60=8.0, buy_atr_std=0.4,
        buy_jump_down_power=-5.0, buy_diff_down_days=5,
    )),
    ('趋势正常回调 → 应放行', dict(
        buy_price_rank90=0.45, buy_deg_ang21=6.0,
        buy_deg_ang60=12.0, buy_atr_std=0.35,
    )),
    ('健康上涨 → 应放行', dict(
        buy_price_rank90=0.60, buy_deg_ang21=15.0,
        buy_deg_ang60=10.0, buy_atr_std=0.25,
    )),
]

for name, features in signals:
    ret = judge.predict(**features)
    verdict = '🔴 拦截' if ret == _EEdgeType.E_EEdge_TOP_LOSS else '🟢 放行'
    print('   {} → {}'.format(name, verdict))

# ── 5. 卖出裁判 ──────────────────────────────────────
print('\n5. ClaudeSellJudge（卖出信号裁判）:')
sell_judge = ClaudeSellJudge(mode='rule')
print('   is_buy_ump(): {}'.format(sell_judge.is_buy_ump()))

# ── 6. API 模式验证 ──────────────────────────────────
print('\n6. Direct 模式（需要 API Key）:')
direct_judge = ClaudeJudge(mode='direct', config=config)
direct_judge._cache['hash_example'] = False
print('   缓存大小: {}, 清空后: '.format(direct_judge.cache_size), end='')
direct_judge.clear_cache()
print('{}'.format(direct_judge.cache_size))

# ── 7. 集成指南 ──────────────────────────────────────
print('\n7. 集成到 abu 回测:')
print('''
┌──────────────────────────────────────────────────────────┐
│  from abupy import AbuUmpManager                        │
│  from abupy.ClaudeBu import ClaudeJudge                 │
│                                                         │
│  # 方式A: 默认规则（无需 API Key）                       │
│  judge = ClaudeJudge(mode="rule")                       │
│                                                         │
│  # 方式B: Claude 专属规则（需要 API Key）                │
│  judge = ClaudeJudge(mode="rule")                       │
│  judge.fit(historical_orders_pd)                        │
│                                                         │
│  # 注册 → 所有后续回测自动过滤                          │
│  AbuUmpManager.append_user_ump(judge)                   │
│  AbuUmpManager.g_enable_user_ump = True                 │
└──────────────────────────────────────────────────────────┘
''')

key_status = '✓ 已配置' if os.environ.get('ANTHROPIC_API_KEY') else '✗ 未配置（默认规则仍可用）'
print('8. API Key 状态: {}'.format(key_status))

print('\n═══ 演示完成 ═══')
