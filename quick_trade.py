#!/usr/bin/env python3
"""
三涨三跌策略 — 跌3天买，涨3天卖
用法: python3 quick_trade.py
"""
import subprocess, json, sys, os
from datetime import datetime

# ── 1. 拉数据（新浪） ─────────────────────────────
def fetch_kline(code, days=500):
    market = 'sh' if code.startswith(('6', '9')) else 'sz'
    url = (f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
           f'CN_MarketData.getKLineData?symbol={market}{code}&scale=240&ma=no&datalen={days}')
    r = subprocess.run(['curl', '-s', '--noproxy', '*', url],
                       capture_output=True, text=True, timeout=15)
    if not r.stdout or r.stdout.startswith('null'):
        return []
    data = json.loads(r.stdout)
    return [(k['day'], float(k['open']), float(k['close']), float(k['high']), float(k['low']))
            for k in data]

# ── 2. 策略 ───────────────────────────────────────
def backtest(code, klines, down_days=3, up_days=3, cash=100000):
    """
    策略: 连续跌 down_days 天 → 第二天开盘买入
         连续涨 up_days 天 → 第二天开盘卖出
    """
    trades = []
    holding = False
    buy_price = 0
    buy_date = ''
    cash_start = cash

    for i in range(max(down_days, up_days) + 1, len(klines) - 1):
        date, open_p, close_p, high_p, low_p = klines[i]

        if not holding:
            # 检查连续下跌
            down_count = 0
            for j in range(1, down_days + 1):
                if klines[i - j][2] < klines[i - j - 1][2]:  # 收盘价比前一天低
                    down_count += 1
            if down_count == down_days:
                # 买入: 第二天开盘
                buy_price = klines[i + 1][1]
                buy_date = klines[i + 1][0]
                holding = True

        else:
            # 检查连续上涨 → 卖出
            up_count = 0
            for j in range(1, up_days + 1):
                if klines[i - j][2] > klines[i - j - 1][2]:
                    up_count += 1
            if up_count == up_days:
                sell_price = klines[i + 1][1]
                sell_date = klines[i + 1][0]
                profit = (sell_price - buy_price) / buy_price
                shares = int(cash / buy_price / 100) * 100  # A股100股整数倍
                if shares >= 100:
                    cash += shares * (sell_price - buy_price)
                trades.append({
                    'buy_date': buy_date, 'buy_price': buy_price,
                    'sell_date': sell_date, 'sell_price': sell_price,
                    'profit_pct': profit * 100,
                })
                holding = False

    # 如果还持有，按最后一天收盘价强制卖出
    if holding:
        last_price = klines[-1][2]
        profit = (last_price - buy_price) / buy_price
        shares = int(cash_start / buy_price / 100) * 100
        if shares >= 100:
            cash += shares * (last_price - buy_price)
        trades.append({
            'buy_date': buy_date, 'buy_price': buy_price,
            'sell_date': klines[-1][0], 'sell_price': last_price,
            'profit_pct': profit * 100,
        })

    return trades, cash - cash_start


# ── 3. ClaudeJudge 过滤 ───────────────────────────
# 把 ClaudeBu 路径加入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'abupy'))
from ClaudeBu.ClaudeJudge import _EEdgeType

def judge_trade(trade, klines):
    """用 ClaudeJudge 默认规则判断该不该做这笔交易"""
    # 从K线提取简单特征
    buy_date = trade['buy_date']
    buy_price = trade['buy_price']

    # 找到买入日前一天的位置
    buy_idx = None
    for i, k in enumerate(klines):
        if k[0] == buy_date:
            buy_idx = i
            break
    if buy_idx is None or buy_idx < 60:
        return False  # 数据不够，放行

    # 算趋势角度（简化版: 最近21天和60天的价格变化率）
    def calc_angle(prices, n):
        if len(prices) < n:
            return 0
        recent = prices[-n:]
        x = list(range(n))
        y = recent
        n_ = len(x)
        slope = (n_ * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / \
                (n_ * sum(xi * xi for xi in x) - sum(x) ** 2 + 0.0001)
        import math
        return math.degrees(math.atan(slope))

    prices = [k[2] for k in klines[:buy_idx + 1]]  # 收盘价序列

    features = {
        'buy_deg_ang21': calc_angle(prices, 21),
        'buy_deg_ang60': calc_angle(prices, 60),
        'buy_price_rank90': sum(1 for p in prices[-90:] if p < buy_price) / max(len(prices[-90:]), 1) if len(prices) >= 90 else 0.5,
        'buy_atr_std': (max(prices[-21:]) - min(prices[-21:])) / (sum(prices[-21:]) / 21) if len(prices) >= 21 else 0.5,
    }

    # 默认规则判断
    score = 0
    price_rank = features['buy_price_rank90']
    trend_short = features['buy_deg_ang21']
    trend_long = features['buy_deg_ang60']
    atr = features['buy_atr_std']

    if price_rank > 0.85 and trend_short < 3:
        score += 3
    if trend_long < -5:
        score += 2
    if atr > 0.8 and price_rank > 0.9:
        score += 3
    return score >= 5  # True = 拦截


# ── 4. 主流程 ─────────────────────────────────────
print('=' * 55)
print('  三跌三卖策略 — A股回测')
print('  规则: 连跌3天→买入, 连涨3天→卖出')
print('=' * 55)

stocks = [
    ('600519', '贵州茅台'),
    ('000001', '平安银行'),
    ('600036', '招商银行'),
    ('300750', '宁德时代'),
    ('000858', '五粮液'),
]

all_trades = []
for code, name in stocks:
    print(f'\n{name} ({code})', end=' ', flush=True)
    klines = fetch_kline(code, days=500)
    if not klines:
        print('数据拉取失败')
        continue

    trades, total_profit = backtest(code, klines, down_days=3, up_days=3)

    wins = sum(1 for t in trades if t['profit_pct'] > 0)
    total = len(trades)
    avg_profit = sum(t['profit_pct'] for t in trades) / total if total > 0 else 0
    print(f'| {total}笔 | 胜率 {wins/total*100:.0f}%' if total > 0 else '| 0笔',
          f'| 盈亏 {total_profit:+,.0f}元')

    # ClaudeJudge 过滤
    if total > 0:
        blocked = sum(1 for t in trades if judge_trade(t, klines))
        kept = [t for t in trades if not judge_trade(t, klines)]
        kept_wins = sum(1 for t in kept if t['profit_pct'] > 0)
        kept_total = len(kept)
        kept_avg = sum(t['profit_pct'] for t in kept) / kept_total if kept_total > 0 else 0

        print(f'  ClaudeJudge: 拦截{blocked}笔 → 剩余{kept_total}笔',
              f'| 胜率 {kept_wins/kept_total*100:.0f}%' if kept_total > 0 else '',
              f'| 均收益 {kept_avg:+.1f}%' if kept_total > 0 else '')

        for t in trades:
            t['code'] = code
            t['name'] = name
            all_trades.append(t)

# ── 5. 汇总 ──────────────────────────────────────
print('\n' + '=' * 55)
print('  汇总')
print('=' * 55)
if all_trades:
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['profit_pct'] > 0)
    blocked_total = sum(1 for t in all_trades if judge_trade(t, []))
    print(f'全部交易: {total_trades}笔 | 胜率: {total_wins/total_trades*100:.1f}%')
    print(f'ClaudeJudge 建议拦截: {blocked_total}笔')

    # 被拦截的 vs 放行的
    blocked_trades = [t for t in all_trades if judge_trade(t, [])]
    passed_trades = [t for t in all_trades if not judge_trade(t, [])]
    if blocked_trades:
        b_avg = sum(t['profit_pct'] for t in blocked_trades) / len(blocked_trades)
        print(f'被拦截的平均收益: {b_avg:+.1f}%')
    if passed_trades:
        p_wins = sum(1 for t in passed_trades if t['profit_pct'] > 0)
        p_avg = sum(t['profit_pct'] for t in passed_trades) / len(passed_trades)
        print(f'放行的平均收益: {p_avg:+.1f}% | 胜率: {p_wins/len(passed_trades)*100:.0f}%')

print('\n提示: 这只是最简单的演示策略，不是投资建议。')
