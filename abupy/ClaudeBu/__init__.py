# -*- encoding:utf-8 -*-
"""
    ClaudeBu — Claude-powered trading judge & strategist for abu quant.

    替换 abu 的 UMP 裁判系统，用 Claude API 做交易信号判断和策略分析。

    两句话上手:
        from abupy.ClaudeBu import ClaudeJudge, ClaudeStrategist

        # 裁判 — 替换 UMP
        judge = ClaudeJudge(mode="rule")
        judge.fit(orders_pd)
        AbuUmpManager.append_user_ump(judge)
        AbuUmpManager.g_enable_user_ump = True

        # 策略师 — 回测后分析
        strategist = ClaudeStrategist()
        report = strategist.analyze(orders_pd, action_pd, capital, benchmark)
"""
from __future__ import absolute_import, print_function, division

from .ClaudeConfig import ClaudeConfig
from .ClaudeJudge import ClaudeJudge, ClaudeSellJudge
from .ClaudeStrategist import ClaudeStrategist

__all__ = [
    'ClaudeConfig',
    'ClaudeJudge',
    'ClaudeSellJudge',
    'ClaudeStrategist',
]

__version__ = '0.1.0'
__author__ = 'ClaudeBu for abu quant'
