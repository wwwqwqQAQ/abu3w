#!/usr/bin/env python3
"""
实时监控 + 买卖预测
用新浪实时行情 + RSI(25,80) 策略，预测每只股票该买还是该卖

用法: python3 live_monitor.py
"""
import subprocess, json, time, os, math
from datetime import datetime

# ═══════════════════════════════════════
# 配置: 你要监控的股票
# ═══════════════════════════════════════
WATCHLIST = [
    ('sh600519', '茅台'),
    ('sh600036', '招行'),
    ('sz000001', '平安银行'),
    ('sz300750', '宁德时代'),
    ('sh601318', '中国平安'),
    ('sz000858', '五粮液'),
    ('sh603259', '药明康德'),
    ('sz002415', '海康威视'),
]

RSI_OVERSOLD = 25   # 低于这个 → 买入信号
RSI_OVERBOUGHT = 80  # 高于这个 → 卖出信号
RSI_PERIOD = 14
INTERVAL = 10  # 刷新间隔(秒)

# ═══════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════

def fetch_history(code, days=300):
    """拉历史数据算RSI"""
    raw = code.replace('sh','').replace('sz','')
    market = 'sh' if code.startswith('sh') else 'sz'
    url = (f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
           f'CN_MarketData.getKLineData?symbol={market}{raw}&scale=240&ma=no&datalen={days}')
    r = subprocess.run(['curl','-s','--noproxy','*',url],
                       capture_output=True, text=True, timeout=15)
    if not r.stdout or r.stdout.startswith('null'): return []
    data = json.loads(r.stdout)
    return [float(k['close']) for k in data]

def fetch_realtime(codes):
    """拉实时行情"""
    ids = ','.join(codes)
    r = subprocess.run(['curl','-s','--noproxy','*',
        '-H','Referer: https://finance.sina.com.cn',
        '-H','User-Agent: Mozilla/5.0',
        f'http://hq.sinajs.cn/list={ids}'],
        capture_output=True, timeout=10)
    # 新浪返回GBK编码
    text = r.stdout.decode('gbk', errors='replace')
    results = {}
    for line in text.strip().split('\n'):
        if not line or '=' not in line: continue
        code = line.split('=')[0].split('_')[-1]
        data = line.split('"')[1] if '"' in line else ''
        if data:
            parts = data.split(',')
            if len(parts) >= 5:
                try:
                    results[code] = {
                        'name': parts[0],
                        'open': float(parts[1]),
                        'yest_close': float(parts[2]),
                        'price': float(parts[3]),
                        'high': float(parts[4]),
                        'low': float(parts[5]),
                    }
                except (ValueError, IndexError):
                    pass
    return results

def calc_rsi(closes, n=14):
    """计算RSI"""
    if len(closes) < n+1: return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(d if d>0 else 0)
        losses.append(-d if d<0 else 0)
    avg_gain = sum(gains[-n:])/n
    avg_loss = sum(losses[-n:])/n
    if avg_loss == 0: return 100
    return 100 - 100/(1 + avg_gain/avg_loss)

# ═══════════════════════════════════════
# 主循环
# ═══════════════════════════════════════

print('='*60)
print('  📡 实时监控 — RSI反转策略')
print(f'  买入信号: RSI < {RSI_OVERSOLD}  |  卖出信号: RSI > {RSI_OVERBOUGHT}')
print(f'  刷新间隔: {INTERVAL}秒')
print('='*60)

# 加载历史数据算初始RSI
print('\n⏳ 加载历史数据...')
history = {}
for code, name in WATCHLIST:
    closes = fetch_history(code, 300)
    if closes:
        # 用最近实时价更新最后一根K线
        rsi = calc_rsi(closes, RSI_PERIOD)
        history[code] = {'name': name, 'closes': closes, 'rsi': rsi}
        print(f'  {name} ({code}) — {len(closes)}天数据, 初始RSI={rsi:.0f}')
    else:
        print(f'  {name} ({code}) — 数据拉取失败')

print('\n🟢 开始实时监控... (Ctrl+C 退出)\n')

try:
    while True:
        # 获取实时行情
        rt = fetch_realtime([c for c,_ in WATCHLIST])

        now = datetime.now().strftime('%H:%M:%S')
        print(f'\n━━━ {now} ━━━')
        print(f'  {"股票":<10} {"现价":>8} {"涨跌":>8} {"RSI":>6} {"信号":<16} {"建议"}')
        print(f'  {"-"*55}')

        for code, name in WATCHLIST:
            if code not in rt or code not in history:
                continue

            info = rt[code]
            price = info['price']
            change = (price - info['yest_close']) / info['yest_close'] * 100

            # 更新历史数据最后一根K线 → 重算RSI
            closes = history[code]['closes'].copy()
            closes[-1] = price
            rsi = calc_rsi(closes, RSI_PERIOD)
            prev_rsi = history[code]['rsi']
            history[code]['rsi'] = rsi

            # 判断信号
            signal = ''
            suggestion = ''

            if rsi < RSI_OVERSOLD:
                signal = '🔵 超卖'
                color = '\033[34m'
                if prev_rsi >= RSI_OVERSOLD:  # 刚跌破
                    suggestion = '⬆️ 准备买入!'
                    color_bright = '\033[1;34m'
                else:
                    suggestion = '等待反弹'
            elif rsi > RSI_OVERBOUGHT:
                signal = '🔴 超买'
                color = '\033[31m'
                if prev_rsi <= RSI_OVERBOUGHT:  # 刚突破
                    suggestion = '⬇️ 准备卖出!'
                else:
                    suggestion = '持有观察'
            elif rsi > 70:
                signal = '🟡 偏高'
                suggestion = '注意回调'
            elif rsi < 35:
                signal = '🟢 偏低'
                suggestion = '关注机会'
            else:
                signal = '⚪ 中性'
                suggestion = '观望'

            # 趋势判断
            trend = ''
            if len(closes) >= 60:
                ma60 = sum(closes[-60:])/60
                trend = '↗' if price > ma60 else '↘'

            rsi_bar = '█' * int(rsi/5) + '░' * (20-int(rsi/5))

            print(f'  {name:<10} {price:>8.2f} {change:>+7.2f}% {rsi:>5.0f} {trend} {signal:<16} {suggestion}')

        print(f'\n  RSI仪表板:  0{rsi_bar}100')
        print(f'             超卖区(<{RSI_OVERSOLD})  ← →  超买区(>{RSI_OVERBOUGHT})')

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print('\n\n👋 监控结束')
