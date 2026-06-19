# -*- encoding:utf-8 -*-
"""
    Claude API 配置模块
    从环境变量读取 API Key，管理模型选择和调用参数
"""
from __future__ import absolute_import, print_function, division

import os
import logging

__author__ = 'ClaudeBu'
__all__ = ['ClaudeConfig']


class ClaudeConfig(object):
    """
    Claude API 配置

    优先级: 构造参数 > 环境变量 > 默认值

    Usage:
        config = ClaudeConfig(model="claude-sonnet-4-6")
        # 或使用默认值（从 ANTHROPIC_API_KEY 环境变量读取 key）
        config = ClaudeConfig()
    """

    # 可用的模型及其适用场景
    MODEL_FAST = "claude-haiku-4-5-20251001"     # 快速便宜，做批量判单
    MODEL_STANDARD = "claude-sonnet-4-6"           # 默认，平衡速度与质量
    MODEL_SMART = "claude-opus-4-8"                # 复杂策略分析

    def __init__(self,
                 api_key=None,
                 model=None,
                 max_tokens=4096,
                 temperature=0.3,
                 base_url=None):
        """
        :param api_key: Anthropic API Key，默认从 ANTHROPIC_API_KEY 环境变量读取
        :param model: 模型 ID，默认 claude-sonnet-4-6
        :param max_tokens: 最大输出 token 数
        :param temperature: 温度参数，量化场景需要低温度保证一致性
        :param base_url: API 地址，默认 https://api.anthropic.com
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.model = model or self.MODEL_STANDARD
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url or 'https://api.anthropic.com'

    @property
    def is_configured(self):
        """检查是否已配置 API Key"""
        return bool(self.api_key)

    def validate(self):
        """验证配置，未配置 API Key 时给出明确提示"""
        if not self.is_configured:
            raise RuntimeError(
                'Claude API Key 未配置。请设置环境变量:\n'
                '  export ANTHROPIC_API_KEY="sk-ant-..."\n'
                '或通过 ClaudeConfig(api_key="sk-ant-...") 传入'
            )
        return True

    def for_strategy(self):
        """策略分析用 - 用最强模型"""
        self.model = self.MODEL_SMART
        self.max_tokens = 8192
        self.temperature = 0.2
        return self

    def for_judge_fast(self):
        """快速判单用 - 用最便宜的模型"""
        self.model = self.MODEL_FAST
        self.max_tokens = 1024
        self.temperature = 0.1
        return self

    def __repr__(self):
        return 'ClaudeConfig(model={}, temp={}, key_ok={})'.format(
            self.model, self.temperature, self.is_configured)
