# 第七章 选股与Alpha生成：AlphaBu与PickStockBu

## 7.1 Alpha生成架构

AlphaBu模块是阿布量化系统的策略编排核心，它实现了将选股（Stock Picking，选择交易哪些标的）与择时（Market Timing，决定何时交易）分离的架构。这一架构设计体现了量化投资中的一个核心认知：选股Alpha和择时Alpha是两个相对独立的收益来源，应该由不同的子系统负责。

AlphaBu采用主-工作（Master-Worker）并行架构。Master负责任务的分配和结果的聚合，Worker负责具体标的的选股筛选和择时回测。通过多进程并行化，系统可以在合理的时间内完成全市场数千只标的的批量回测。

## 7.2 选股因子设计

**源码片段：回归角度选股策略**

```python
# -*- encoding:utf-8 -*-
"""
    选股示例因子：价格拟合角度选股因子
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from ..UtilBu import ABuRegUtil
from .ABuPickStockBase import AbuPickStockBase, reversed_result

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuPickRegressAngMinMax(AbuPickStockBase):
    """拟合角度选股因子示例类"""
    def _init_self(self, **kwargs):
        """通过kwargs设置拟合角度边际条件，配置因子参数"""

        # 暂时与base保持一致不使用kwargs.pop('a', default)方式
        # fit_pick中 ang > threshold_ang_min, 默认负无穷，即默认所有都符合
        self.threshold_ang_min = -np.inf
        if 'threshold_ang_min' in kwargs:
            # 设置最小角度阀值
            self.threshold_ang_min = kwargs['threshold_ang_min']

        # fit_pick中 ang < threshold_ang_max, 默认正无穷，即默认所有都符合
        self.threshold_ang_max = np.inf
        if 'threshold_ang_max' in kwargs:
            # 设置最大角度阀值
            self.threshold_ang_max = kwargs['threshold_ang_max']

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        """开始根据自定义拟合角度边际参数进行选股"""

        # 计算走势角度
        ang = ABuRegUtil.calc_regress_deg(kl_pd.close, show=False)
        # 根据参数进行角度条件判断
        if self.threshold_ang_min < ang < self.threshold_ang_max:
            return True
        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        raise NotImplementedError('AbuPickRegressAng fit_first_choice unsupported now!')
```


### 7.2.1 抽象接口

ABuPickBase定义了选股因子和择时工作类的抽象接口。选股因子必须实现fit()方法和init_stock_pickers()方法。fit()方法接收候选标的池和基准K线，通过一系列条件筛选出符合条件的标的。

### 7.2.2 两级筛选机制

ABuPickStockWorker实现了两级筛选的选股架构：

第一级——初选（_first_batch_fit）：初选因子（first_stock_pickers）以批量方式运行，每个初选因子对全部候选标的调用fit_first_choice方法。这是一个"漏斗"模型——每个初选因子都作为一个粗筛网，逐步缩小候选池。初选主要使用计算成本低、数据需求少的条件（如价格区间、市值范围、行业分类等）。

第二级——精细选股（_batch_fit）：对第一阶段保留下来的每个标的，逐一迭代常规选股因子（stock_pickers）。与初选的"全或无"批量过滤不同，精细选股采用的是"一票否决"制——任何一个因子调用fit_pick返回False都将该标的从候选池中移除。精选用到的因子计算成本更高、分析更深入（如技术形态识别、趋势质量评估、多因子评分等）。

这种两级筛选设计在计算效率和筛选精度之间取得了良好的平衡。初选通过低成本条件快速淘汰大部分不合格标的，精选用高成本分析对剩余少量标的进行深度筛选。

### 7.2.3 核心选股策略

ABuPickRegressAngMinMax：基于回归角度的选股策略。对每只标的的历史收盘价序列进行多项式回归，计算拟合线的斜率角度。选择角度最大的标的（最强上升趋势）或角度最小的标的（最弱下降趋势/最可能反弹）。该策略利用了TLineBu系统的趋势识别能力，将趋势强度作为选股的核心标准。

ABuPickSimilarNTop：基于相似性的N强选股策略。使用SimilarBu模块的相关性分析，找出与目标标的（或基准指数）最相似的N只标的。相似性基于价格变化率（p_change）的滚动时间加权相关性，而非原始价格的相关性——这确保捕捉的是"共同波动模式"而非"价格水平相似"。

ABuPickStockPriceMinMax：基于价格区间的选股策略。选择价格处于历史百分位区间（如20%-80%）内的标的，避免选择处于极端高价（可能顶部）或极端低价（可能有基本面问题）的标的。使用scipy.stats.percentileofscore计算当前价格在历史分布中的位置。

### 7.2.4 选股并行化

ABuPickStockMaster是选股并行化的调度器。其算法流程为：

1. 如果未提供choice_symbols（预选标的列表），则从对应市场获取全部标的。

2. 使用split_k_market函数将标的列表均匀分割为n_process_pick_stock个子列表。

3. 使用Parallel/delayed（joblib接口）在每个进程上启动do_pick_stock_work工作函数，传入子列表和选股因子。

4. 所有进程完成后，使用itertools.chain.from_iterable合并各进程的结果。

5. 支持训练/测试集分割模式：通过g_enable_last_split_test和g_enable_train_test_split全局标志控制。在训练/测试模式下，选股在训练集上执行，回测在测试集上执行，模拟了真实的样本外（Out-of-Sample）测试环境。

do_pick_stock_thread_work函数在进程内部进一步使用ThreadPoolExecutor进行线程级分解。由于Python的GIL（Global Interpreter Lock）限制，线程并不能实现真正的CPU并行（线程适合I/O密集型任务），但对于选股这种涉及大量K线数据读取的任务，线程并行仍然能带来性能提升。

## 7.3 择时执行引擎

### 7.3.1 任务循环架构

ABuPickTimeWorker是择时执行的核心。它维护着买入因子、卖出因子和未平仓订单的内部状态，通过日任务、周任务和月任务三个层次的循环来驱动交易信号。

fit()方法首先在K线数据上标注周任务和月任务的触发点：自然周五标记为周任务日，自然月末标记为月任务日。然后通过kl_pd.apply(_task_loop, axis=1)逐行（逐交易日）迭代整个回测周期。axis=1意味着apply沿DataFrame的行方向（时间方向）迭代，每个交易日调用一次_task_loop函数。

### 7.3.2 任务分层

_task_loop在每个交易日执行三个层次的任务：

日任务（每个交易日执行）：
- 检查所有未平仓订单（遍历卖出因子，对每个因子调用read_fit_day检查其管辖范围内的订单是否应离场）
- 检查所有未锁定买入因子（遍历买入因子，对每个因子调用read_fit_day检查是否应该入场）
- 买入因子的锁状态由其专属选股因子控制：如果专属选股因子判定当前不适合交易（fit_pick返回False），则锁定该买入因子

周任务（周五执行）：
- 执行专属卖出因子的周操作（例如基于周线数据的止损调整）
- 执行全局卖出因子的fit_week操作
- 执行买入因子专属选股因子的周筛选（重新评估标的的适宜性）
- 执行买入因子的fit_week操作（例如重新计算星期效应策略的工作日胜率）

月任务（月末执行）：
- 与周任务相同的四类月操作（fit_month），但使用月度数据
- 对于自适应策略（如ABuFactorBuyDM），月任务中重新计算自适应参数（慢线周期和快线周期）
- 对于大盘过滤器（如ABuSDBreak），月任务中重新评估大盘震荡程度

### 7.3.3 专属卖出因子

_task_attached_sell方法实现了买入因子与卖出因子的绑定机制。在周任务或月任务执行时，系统会找出每个买入因子产生的所有未平仓订单，按买入因子分组后分别传递给该买入因子的专属卖出因子进行离场检查。

这种机制允许不同的买入因子使用不同的出场策略。例如，突破买入（追逐趋势）可以配置追踪止盈（让利润奔跑），而回调买入（期待反弹）可以配置时间止损（限定持有期）。这种差异性出场策略是"策略多样性"的核心——每个子策略根据自己的交易逻辑选择最合适的离场方式。

### 7.3.4 因子任务分类

为提高运行效率，filter_long_task_factors函数在初始化时通过hasattr反射检查对所有因子进行分类，预先将具有fit_week方法的因子归入周任务序列，将具有fit_month方法的因子归入月任务序列。这样在逐日迭代（这是回测中最频繁的操作，可能执行数千次）中，不需要每次都进行hasattr检查，直接使用预先分类的结果。

### 7.3.5 择时并行化

ABuPickTimeMaster是择时并行化的调度器。与选股并行化不同，择时并行化的关键是每个子进程使用相同的买入/卖出因子集合（即相同的策略逻辑），但处理不同的标的子集：

1. 如果K线数据尚未获取，通过kl_pd_manager.batch_get_pick_time_kl_pd批量并行预取所有候选标的的K线数据。

2. 将标的序列按n_process_pick_time进行分割。

3. 每个进程运行do_symbols_with_same_factors，使用完全相同的买入/卖出因子集合处理其分配的标的子集。

4. 所有进程完成后，合并orders_pd和action_pd（按日期和操作类型排序），然后通过ABuTradeExecute.apply_action_to_capital将所有交易操作应用到资金账户进行模拟。

### 7.3.6 补位机制

do_symbols_with_same_factors函数实现了补位（Backfill）机制。在执行过程中，某些标的可能因为数据错误、停牌或策略未能产生任何交易信号而失败。补位机制维护了一个后备标的池（back_target_symbols），当当前标的失败时，从补位池中弹出一个替代标的，使用相同的因子集合进行分析。这确保了在指定数量的进程中始终有足够的工作量，最大化了并行效率。

## 7.4 多标的异质策略

do_symbols_with_diff_factors函数支持每个标的使用不同的策略组合。函数接收一个factor_dict字典（键为标的符号，值为(买入因子列表, 卖出因子列表)的元组），以及一个func_factors回调函数，根据每个标的的符号动态查找其对应的策略配置。

这种异质策略设计对于以下场景特别有价值：
- 不同行业适用不同的策略参数（周期性行业用长周期策略，科技行业用短周期策略）
- 不同市值规模的标的适用不同的策略逻辑（大盘股用趋势策略，小盘股用反转策略）
- A/B测试——同一批标的使用不同的策略变体进行对比分析
