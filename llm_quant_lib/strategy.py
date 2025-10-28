# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# 【修改 1】导入 OpenAI SDK
from openai import OpenAI
# import deepseek # 不再需要导入 deepseek
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseStrategy(ABC):
    """策略基类 (接口定义)"""
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        """
        初始化策略。

        Args:
            universe_df (pd.DataFrame): 包含 'sec_code' 和 'universe' 列的 DataFrame。
            **kwargs: 其他策略特定参数。
        """
        if 'sec_code' not in universe_df.columns or 'universe' not in universe_df.columns:
            raise ValueError("universe_df 必须包含 'sec_code' 和 'universe' 列。")
        self.universe_df = universe_df
        self.params = kwargs

    @abstractmethod
    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """
        根据当前市场情况和投资组合状态，计算目标权重。
        这是策略的核心决策逻辑。

        Args:
            current_date (pd.Timestamp): 当前决策日期 (因子数据的日期)。
            factor_snapshot (pd.DataFrame): 当天所有相关 ETF 的因子截面数据 (索引: sec_code, 列: 因子名称)。
                                         因子值已经过预处理 (例如填充 NaN)。
            portfolio_state (dict): 当前投资组合状态，包含:
                                  { 'cash': float, 'current_positions': {sec_code: shares} }

        Returns:
            dict: 目标权重字典 {sec_code: target_weight}。权重总和应约为 1.0 (如果要做多)。
                  返回空字典 {} 表示清空所有仓位。
        """
        pass

class LLMStrategy(BaseStrategy):
    """
    基于大型语言模型 (DeepSeek, 通过 OpenAI 兼容接口调用) 决策的策略。
    它接收因子快照，调用 AI，并解析其返回的目标持仓。
    """
    def __init__(self, universe_df: pd.DataFrame, universe_to_trade: str = 'equity_us', top_n: int = 5, api_key: Optional[str] = None):
        """
        初始化 LLM 策略。

        Args:
            universe_df (pd.DataFrame): 资产池定义。
            universe_to_trade (str): 指定 AI 关注哪个资产池 (例如 'equity_us')。 'All' 代表使用所有资产。
            top_n (int): 要求 AI 选择的 ETF 数量上限。
            api_key (str, optional): DeepSeek API 密钥。如果为 None，则尝试从环境变量 DEEPSEEK_API_KEY 读取。
        """
        super().__init__(universe_df)
        self.universe_to_trade = universe_to_trade
        self.top_n = top_n
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未提供。请设置 DEEPSEEK_API_KEY 环境变量或在初始化时传入 api_key。")

        # 【修改 2】使用 OpenAI SDK 初始化客户端，并指定 base_url
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com" # 指定 DeepSeek 的 API 地址
            )
            # 你可以尝试调用一次简单的 API 来测试连接（可选）
            # self.client.models.list() # 检查是否能获取模型列表
        except Exception as e:
            # 更新错误信息以反映使用的是 OpenAI SDK
            raise RuntimeError(f"使用 OpenAI SDK 初始化 DeepSeek 连接失败，请检查 API Key 和 base_url: {e}")

        self.trade_log: List[Dict] = [] # 用于记录 AI 的每次决策和理由

    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """
        调用 LLM API 获取目标权重。
        """
        print(f"\n[LLMStrategy] 正在 {current_date.date()} 为资产池 '{self.universe_to_trade}' 请求 AI 决策...")

        # 1. 确定本次决策的资产列表
        if self.universe_to_trade.lower() == 'all':
            assets_in_scope = factor_snapshot.index.tolist() # 使用因子快照中所有有效的代码
        else:
            assets_in_scope = self.universe_df[self.universe_df['universe'] == self.universe_to_trade]['sec_code'].tolist()

        # 过滤因子快照，只保留当前范围内的资产
        current_factor_data = factor_snapshot[factor_snapshot.index.isin(assets_in_scope)]

        # 如果该资产池当前无有效因子数据，则清仓
        if current_factor_data.empty:
            print(f"[LLMStrategy] 警告: 在 {current_date.date()} 未找到资产池 '{self.universe_to_trade}' 的有效因子数据。清空仓位。")
            self.trade_log.append({
                "date": current_date, "justification": "无有效因子数据，清仓。",
                "ai_recommendation": [], "valid_selection": [], "target_weights": {}
            })
            return {} # 返回空字典表示清仓

        # 2. 构建发送给 AI 的 Prompt
        prompt_messages = self._create_prompt(current_date, current_factor_data, portfolio_state)

        # 3. 调用 API (使用 OpenAI SDK 的 client)
        target_weights: Dict[str, float] = {}
        ai_recommendation: List[str] = []
        valid_selection: List[str] = []
        justification = "AI 决策失败或跳过。"
        raw_output = "" # 初始化 raw_output

        try:
            # ---【API 调用点】---
            print("[LLMStrategy] 正在调用 DeepSeek API (via OpenAI SDK)...")
            response = self.client.chat.completions.create(
                model="deepseek-reasoner", # 仍然使用 DeepSeek 的模型名称
                messages=prompt_messages,
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=1024 # 限制最大输出长度
            )
            # 获取返回内容的方式与 OpenAI SDK 一致
            raw_output = response.choices[0].message.content
            # print(f"[LLMStrategy] AI 原始返回: {raw_output}") # 调试时可以取消注释
            decision = json.loads(raw_output)

            # 4. 解析 AI 决策
            justification = decision.get("justification", "AI 未提供理由。")
            ai_recommendation = decision.get("long_list", [])

            # 清理和验证 AI 返回的代码
            valid_selection = [code for code in ai_recommendation if code in current_factor_data.index]

            print(f"[LLMStrategy] AI 理由: {justification}")
            # print(f"[LLMStrategy] AI 推荐 (原始): {ai_recommendation}")
            print(f"[LLMStrategy] AI 推荐 (有效): {valid_selection}")

            # 5. 将 AI 选择转换为目标权重 (等权)
            if valid_selection:
                # 只取前 top_n 个有效的
                target_securities = valid_selection[:self.top_n]
                if target_securities:
                    equal_weight = 1.0 / len(target_securities)
                    target_weights = {sec: equal_weight for sec in target_securities}
                    print(f"[LLMStrategy] 最终目标权重 ({len(target_weights)}个): { {k: f'{v:.2%}' for k, v in target_weights.items()} }")
                else:
                    print("[LLMStrategy] AI 推荐的有效资产为空，清空仓位。")
                    target_weights = {} # 清仓
            else:
                print("[LLMStrategy] AI 未推荐任何有效资产，清空仓位。")
                target_weights = {} # 清仓

        except json.JSONDecodeError:
            print(f"[LLMStrategy] AI 返回的不是有效的 JSON 格式: {raw_output}。本期跳过交易。")
            justification = f"AI返回格式错误: {raw_output}"
            target_weights = {} # 跳过则清仓
        except Exception as e:
            # 捕获 OpenAI SDK 可能抛出的其他错误，例如认证失败、连接错误等
            print(f"[LLMStrategy] 调用 API 或处理时发生错误: {e}。本期跳过交易。")
            import traceback
            traceback.print_exc() # 打印详细错误信息
            justification = f"API调用或处理错误: {e}"
            target_weights = {} # 跳过则清仓

        # 记录到日志 (无论成功失败都记录)
        self.trade_log.append({
            "date": current_date,
            "justification": justification,
            "ai_recommendation": ai_recommendation,
            "valid_selection": valid_selection,
            "target_weights": target_weights
        })

        return target_weights

    def _create_prompt(self, current_date: pd.Timestamp, factor_data: pd.DataFrame, portfolio_state: dict) -> List[Dict]:
        """
        构建发送给 DeepSeek API 的 Prompt (保持不变)。
        """
        # 将因子数据转换为 CSV 字符串
        factor_csv = factor_data.to_csv(float_format='%.3f')
        num_assets = len(factor_data)

        # 系统提示：定义 AI 的角色、目标和规则
        system_prompt = f"""
        你是一位顶尖的量化投资组合经理，专精于美国 ETF 市场。
        你的目标是：通过分析提供的技术因子数据，选出未来一个月最有可能实现最大利润 (Maximize PnL) 的 TOP {self.top_n} 支 ETF (最多不超过 {num_assets} 支)。
        你将收到当前日期、当前持仓（仅供参考），以及资产池 '{self.universe_to_trade}' 中所有可用 ETF ({num_assets} 支) 的技术因子快照（EMA, MACD, RSI）。

        因子解释:
        - ema_12, ema_26: 短期和长期指数移动平均线。价格在均线之上通常看涨，金叉 (短期上穿长期) 是买入信号。
        - macd_line_12_26: MACD 快线 (EMA12 - EMA26)。正值且上升表示强劲上升趋势。
        - macd_signal_9: MACD 信号线 (MACD快线的9期EMA)。快线在慢线上方且距离拉大是看涨信号。
        - macd_hist_12_26_9: MACD 柱状图 (快线 - 慢线)。柱状图在零轴上方且放大是强烈的看涨信号。
        - rsi_14: 相对强弱指数 (0-100)。低于 30 通常表示超卖（可能反弹），高于 70 通常表示超买（可能回调）。50 是中性区域。

        你的任务与规则:
        1.  **核心任务**: 分析下面 CSV 表格中的因子数据，识别出短期上涨潜力最大的 ETF。关注点：
            * **趋势**: 价格是否在 EMA 之上？EMA 是否形成金叉？MACD 线是否在零轴之上？
            * **动量**: MACD 柱状图是否为正且在放大？RSI 是否在中性或强势区域 (例如 > 50)？
            * **反转**: RSI 是否处于超卖区域 (例如 < 30 或 < 40) 暗示潜在反弹？
        2.  **策略**: 只做多 (Long-Only)。
        3.  **选股**: 从所有提供的 ETF 中，选出你认为未来一个月表现会最好的 **最多 {self.top_n} 支**。如果好的机会不足 {self.top_n} 个，可以选择更少的数量，甚至不选。
        4.  **输出格式**: 你的回应**必须**是一个有效的 JSON 对象。
        5.  **JSON 结构**:
            * `"justification"` (string): 简要说明（1-3句话）你选择这些 ETF 的主要原因，**必须**基于提供的因子数据。清晰地解释你的选股逻辑。
            * `"long_list"` (list of strings): 一个包含你选择的 ETF 代码 (sec_code) 的列表。例如： `["SPY.P", "QQQ.O", "XLK.P"]`。列表中的代码必须来自下面提供的 CSV 数据。列表可以为空 `[]` 如果你认为没有好的买入机会。
        6.  **目标**: 最大化 PnL。请基于因子数据做出最有利可图的选择。
        """

        # 用户提示：提供具体的數據
        user_prompt = f"""
        ### 分析背景
        - 今天日期: {current_date.strftime('%Y-%m-%d')}
        - 你的目标: 从以下 {num_assets} 支 ETF 中，选出未来一个月 PnL 潜力最大的 **最多 {self.top_n} 支**。
        - 当前持仓 (仅供参考): {list(portfolio_state.get('current_positions', {}).keys())}

        ### ETF 技术因子快照 (资产池: '{self.universe_to_trade}', CSV 格式)
        ```csv
        {factor_csv}
        ```

        ### 你的决策
        请根据上述因子数据，选出最多 {self.top_n} 支 ETF，并以要求的 JSON 格式返回你的选择和理由。
        """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def get_trade_log(self) -> pd.DataFrame:
        """返回 AI 决策日志的 DataFrame"""
        if not self.trade_log:
            return pd.DataFrame(columns=['date', 'justification', 'ai_recommendation', 'valid_selection', 'target_weights'])
        return pd.DataFrame(self.trade_log)

# --- (可选) 添加 StaticWeightStrategy 类 ---
# 如果你想保留原来的静态策略作为对比，可以在这里添加
# class StaticWeightStrategy(BaseStrategy):
#     # ... (省略，参考之前的回复)

