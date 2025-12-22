# 多因子量化回测框架

本项目是一个基于 Python 开发的专业多因子量化回测与分析平台。框架采用解耦设计，支持自定义因子开发、多维风险审计（VaR/ES）以及多基准动态对标。

## 核心功能特性 (v2.5)

* **多因子打分体系**: 支持多因子加权打分排序（Top-N 策略），自动处理因子标准化与横截面对比。
* **多维风险审计**: 集成  置信度的 **VaR (风险价值)** 与 **ES (预期缺口)** 计算，动态追踪策略的下行风险暴露。
* **专业双轴可视化**:
* **左轴 (Left Axis)**: 展示策略与基准（S&P 500, MXWD 等）的累计净值曲线。
* **右轴 (Right Axis)**: 以紫色填充区域展示**累计超额收益 (Alpha)**，直观呈现超额收益的来源与回撤。


* **交互性能优化**: 利用 `st.session_state` 实现回测结果持久化，彻底解决了在分析因子相关性时标签页自动跳转的 Bug。
* **费用与损耗分析**: 详细拆解 **Commission (佣金)** 与 **Slippage (滑点)**，计算交易成本对总收益的损耗（Return Drag）。

---

## 项目结构

```text
my_llm_backtester/
├── llm_quant_lib/           # 核心量化算法库
│   ├── data_handler.py      # 数据加载与基准对齐
│   ├── factor_definitions.py # 【因子定义层】
│   ├── factor_engine.py     # 【因子计算与注册】
│   ├── strategy.py          # 多因子 Top-N 策略逻辑
│   ├── portfolio.py         # 账户持仓与交易执行
│   ├── backtest_engine.py   # 回测循环驱动
│   └── performance.py       # 性能分析与风险指标 (VaR/ES)
├── data/
│   └── processed/           # 价格 CSV 与基准收益率数据 (SPXT, BCOM, etc.)
├── app.py                   # Streamlit 双轴看板主程序
├── requirements.txt         # 自动化依赖清单
└── config.yaml              # 全局路径与参数配置

```

---

## 指南：如何添加新因子

框架支持插件化扩展，只需三步即可完成新因子的集成：

### 1. 定义计算逻辑

在 `llm_quant_lib/factor_definitions.py` 中创建一个继承自 `BaseAlpha` 的类。

```python
class FactorRSIDev(BaseAlpha):
    """RSI 偏离因子"""
    def predict(self):
        # 示例逻辑：计算 14 日 RSI 并提取其变动
        rsi = self.calculate_rsi(window=14) 
        return rsi.pct_change()

```

### 2. 注册因子

在 `llm_quant_lib/factor_engine.py` 的 `FACTOR_REGISTRY` 字典中添加映射。

```python
FACTOR_REGISTRY = {
    'momentum': (FactorMomentum, {'window': 168}),
    'rsi_dev': (FactorRSIDev, {}), # 注册新因子
}

```

### 3. 前端启用

在 `app.py` 的 `available_factors` 列表中加入名称，运行后即可在侧边栏勾选并分配权重。

---

## 核心数学指标说明

* **VaR ()**:



代表策略单日亏损有  的概率不会超过该值。
* **Expected Shortfall (ES)**:



衡量当亏损突破 VaR 阈值后的平均损失深度，更有效地度量尾部极端风险。

---

## 快速启动

1. **安装依赖**:
```bash
pip install -r requirements.txt

```


2. **启动看板**:
```bash
streamlit run app.py

```


