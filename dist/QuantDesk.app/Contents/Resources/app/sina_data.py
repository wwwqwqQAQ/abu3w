#!/usr/bin/env python3
"""
新浪 A 股数据获取工具
用法:
    python3 sina_data.py 600519          # 拉贵州茅台日线
    python3 sina_data.py 000001 300750   # 拉多只
    python3 sina_data.py 600519 --years 3  # 拉3年数据
"""
import subprocess, json, sys, os
from datetime import datetime

SINA_URL = (
    'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
    'CN_MarketData.getKLineData?symbol={market}{code}&scale=240&ma=no&datalen={days}'
)


def fetch_kline(code, days=1000):
    """拉取单只股票历史日线"""
    market = 'sh' if code.startswith(('6', '9')) else 'sz'

    r = subprocess.run(
        ['curl', '-s', '--noproxy', '*',
         SINA_URL.format(market=market, code=code, days=days)],
        capture_output=True, text=True, timeout=15)

    if not r.stdout or r.stdout.startswith('null'):
        return None

    data = json.loads(r.stdout)
    rows = []
    for k in data:
        rows.append({
            'date': k['day'],
            'open': float(k['open']),
            'high': float(k['high']),
            'low': float(k['low']),
            'close': float(k['close']),
            'volume': int(float(k['volume'])),
        })
    return rows


def save_csv(code, rows):
    """保存为 CSV"""
    filename = f'{code}.csv'
    with open(filename, 'w') as f:
        f.write('date,open,high,low,close,volume\n')
        for r in rows:
            f.write(f"{r['date']},{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']}\n")
    return filename


def print_summary(code, rows):
    """打印摘要"""
    first, last = rows[0], rows[-1]
    change = (last['close'] - first['close']) / first['close'] * 100
    print(f'\n{code}')
    print(f'  {len(rows)} 个交易日 | {first["date"]} → {last["date"]}')
    print(f'  收盘: {first["close"]:.2f} → {last["close"]:.2f} ({change:+.1f}%)')


if __name__ == '__main__':
    codes = []
    years = 2

    # 解析参数
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--years':
            years = int(args[i + 1])
            i += 2
        else:
            codes.append(args[i])
            i += 1

    if not codes:
        codes = ['600519', '000001', '300750']  # 默认: 茅台、平安银行、宁德时代

    days = years * 250  # 每年约250个交易日

    for code in codes:
        print(f'拉取 {code}...', end=' ', flush=True)
        rows = fetch_kline(code, days=days)
        if rows:
            fname = save_csv(code, rows)
            print_summary(code, rows)
            print(f'  → 已保存 {fname}')
        else:
            print('失败 (代码可能不对)')
