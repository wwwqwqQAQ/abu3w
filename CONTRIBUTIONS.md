# 阿布量化（abu3w）项目 · 个人贡献说明

> 作者：wwwqwqQAQ（3w）　·　仓库：github.com/wwwqwqQAQ/abu3w　·　提交：Publish 3w quant software and thesis

## 1. 归属说明

本仓库发布“3w 量化软件与论文”，核心框架基于开源【阿布量化（AbuQuant）】。

- **基础框架部分**（`abupy`、`abupy_lecture`、`abupy_ui`、`ipython`、`python`、`thesis` 目录、`readme.md` / `readme-en.md`、`LICENSE` 等）版权归阿布量化原作者（阿布 / maxmon）所有。
- **以下所列** 为作者 `wwwqwqQAQ` 在框架之上自行开发、延伸的贡献部分。

## 2. QuantDesk macOS 桌面应用

在阿布量化框架之上，封装为原生 macOS 桌面应用 **QuantDesk**，把量化数据、后端服务与前端界面整合进一个可双击运行的 App。

- `bun` 脚本构建，输出 `QuantDesk.app`（`package.json`）。
- 内嵌 FastAPI 后端（`server.py`，监听 8888 端口）。
- 提供 `static/` 前端资源，App 启动后拉取 `/api/stocks` 渲染界面。
- 用 `WKWebView` 在原生 macOS 窗口内展示。
- 日志写入 `~/Library/Application Support/QuantDesk/logs/server.log`。
- 集成 Claude AI 分析（读取 Anthropic API key，见 `demo_claude_judge.py`）。

## 3. 自定义量化工具

- `sina_data.py` —— 新浪 A 股数据获取工具：拉取历史日线、存 CSV、打印摘要，支持单只或多只股票。
- `quick_trade.py` —— “三涨三跌”策略：连跌 3 天买入、连涨 3 天卖出，含回测逻辑。
- `live_monitor.py` —— 实盘监控。
- `demo_claude_judge.py` —— Claude AI 研判 / 分析。
- `server.py` —— FastAPI 应用后端（约 107KB）。
- `radar/`、`predict/` —— 预测与分析模块。
- `gen/`、`scripts/`、`assets/`、`static/` —— 生成、脚本与前端资源。
- `000001.csv`、`600519.csv` —— 示例行情数据（上证指数 / 贵州茅台）。

## 4. 论文

随软件一并发布（提交信息注明 “and thesis”）。

## 5. 小结

作者在阿布量化开源框架基础上，完成了桌面化封装（QuantDesk）、自定义数据与策略工具（`sina_data`、`quick_trade`、`live_monitor`、Claude AI 研判）、以及配套论文，形成一套“数据获取 → 量化策略 → 实盘监控 → AI 分析 → 桌面可视化”的完整工具链。
