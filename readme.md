# 阿布量化 · 3w 量化软件与论文

> 基于开源 **阿布量化（AbuQuant）** 框架，在保留其核心量化能力之上，作者二次开发并封装了一套可实际使用的量化桌面工具链（数据获取 → 量化策略 → 实盘监控 → AI 研判 → 桌面可视化），并随附论文。

## 项目构成

| 目录 / 文件 | 说明 |
|---|---|
| `abupy` | 阿布量化系统源代码（原始框架） |
| `abupy_lecture` / `abupy_ui` | 使用教程 / 非编程界面操作（原始框架） |
| `ipython` / `python` | 《量化交易之路》示例代码（原始框架） |
| `server.py` + `static/` | **作者开发** — FastAPI 应用后端与前端 |
| `macos/` + `package.json` | **作者开发** — QuantDesk macOS 桌面应用（bun 构建，WKWebView 原生窗口） |
| `sina_data.py` | **作者开发** — 新浪 A 股数据获取工具 |
| `quick_trade.py` | **作者开发** — 「三涨三跌」策略（连跌 3 天买、连涨 3 天卖） |
| `live_monitor.py` | **作者开发** — 实盘监控 |
| `demo_claude_judge.py` | **作者开发** — Claude AI 研判 |
| `predict/` · `radar/` | **作者开发** — 预测与分析模块 |
| `thesis/` | 论文 |

## 作者贡献

本仓库在阿布量化框架之上的**个人贡献说明**，详见 **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)**。

## 快速开始

以下为作者开发的自定义工具示例：

```bash
# 拉取贵州茅台日线并保存为 CSV
python3 sina_data.py 600519

# 「三涨三跌」策略回测
python3 quick_trade.py
```

其中 **QuantDesk macOS 桌面应用** 的构建与使用，见 [README_APP.md](README_APP.md)。

## 归属与致谢

- 基础框架（`abupy`、`abupy_lecture`、`abupy_ui`、`ipython`、`python`、原始 `thesis` 目录、`readme` 等）版权归 **阿布量化原作者（阿布 / maxmon）** 所有。
- 作者 `wwwqwqQAQ`（3w）在框架之上完成桌面化封装与自定义量化工具的二次开发，详见 [CONTRIBUTIONS.md](CONTRIBUTIONS.md)。
- 官方站点：<https://www.abuquant.com>

## 许可

沿用原框架的 **GPL-3.0** 许可证（见 [LICENSE](LICENSE)）。
