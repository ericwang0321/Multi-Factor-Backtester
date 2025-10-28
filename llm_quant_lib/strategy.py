# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from openai import OpenAI # 使用 OpenAI SDK
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
    它接收因子快照，调用 AI，并解析其返回的目标持仓及其权重。
    """
    def __init__(self, universe_df: pd.DataFrame, universe_to_trade: str = 'equity_us', top_n: int = 5, api_key: Optional[str] = None):
        """
        初始化 LLM 策略。

        Args:
            universe_df (pd.DataFrame): 资产池定义。
            universe_to_trade (str): 指定 AI 关注哪个资产池 (例如 'equity_us')。 'All' 代表使用所有资产。
            top_n (int): AI 最多可以选择的 ETF 数量上限 (Max Top N)。
            api_key (str, optional): DeepSeek API 密钥。如果为 None，则尝试从环境变量 DEEPSEEK_API_KEY 读取。
        """
        super().__init__(universe_df)
        self.universe_to_trade = universe_to_trade
        self.max_top_n = top_n # 将传入的 top_n 视为最大数量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未提供。请设置 DEEPSEEK_API_KEY 环境变量或在初始化时传入 api_key。")

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com" # 指定 DeepSeek 的 API 地址
            )
        except Exception as e:
            raise RuntimeError(f"使用 OpenAI SDK 初始化 DeepSeek 连接失败，请检查 API Key 和 base_url: {e}")

        self.trade_log: List[Dict] = [] # 用于记录 AI 的每次决策和理由

    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """
        调用 LLM API 获取目标 ETF 及其权重。
        """
        print(f"\n[LLMStrategy] 正在 {current_date.date()} 为资产池 '{self.universe_to_trade}' 请求 AI 决策 (含权重)...")

        # 1. 确定本次决策的资产列表
        if self.universe_to_trade.lower() == 'all':
            assets_in_scope = factor_snapshot.index.tolist()
        else:
            assets_in_scope = self.universe_df[self.universe_df['universe'] == self.universe_to_trade]['sec_code'].tolist()

        current_factor_data = factor_snapshot[factor_snapshot.index.isin(assets_in_scope)]

        if current_factor_data.empty:
            print(f"[LLMStrategy] 警告: 在 {current_date.date()} 未找到资产池 '{self.universe_to_trade}' 的有效因子数据。清空仓位。")
            self.trade_log.append({
                "date": current_date, "justification": "无有效因子数据，清仓。",
                "ai_recommendation": {}, "final_weights": {}
            })
            return {}

        # 2. 构建发送给 AI 的 Prompt (提示 AI 返回权重)
        prompt_messages = self._create_prompt(current_date, current_factor_data, portfolio_state)

        # 3. 调用 API
        target_weights: Dict[str, float] = {}
        ai_recommendation: Dict[str, float] = {}
        justification = "AI 决策失败或跳过。"
        raw_output = ""

        try:
            print("[LLMStrategy] 正在调用 DeepSeek API (via OpenAI SDK)...")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=prompt_messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1536
            )
            raw_output = response.choices[0].message.content
            # print(f"[LLMStrategy] AI 原始返回: >>>{raw_output}<<<") # 调试时取消注释
            decision = json.loads(raw_output)

            # 4. 解析 AI 决策
            justification = decision.get("justification", "AI 未提供理由。")
            ai_recommendation = decision.get("target_weights", {})

            # 清理和验证 AI 返回的权重
            valid_weights: Dict[str, float] = {}
            if isinstance(ai_recommendation, dict):
                for code, weight in ai_recommendation.items():
                    if code in current_factor_data.index and isinstance(weight, (int, float)) and weight > 1e-6:
                         valid_weights[code] = float(weight)
            else:
                 print(f"[LLMStrategy] 警告: AI 返回的 target_weights 不是字典格式: {ai_recommendation}")

            print(f"[LLMStrategy] AI 理由: {justification}")
            print(f"[LLMStrategy] AI 推荐权重 (有效): { {k: f'{v:.2%}' for k, v in valid_weights.items()} }")

            # 5. 权重处理与归一化
            if valid_weights:
                sorted_weights = dict(sorted(valid_weights.items(), key=lambda item: item[1], reverse=True))
                target_weights_unnormalized = dict(list(sorted_weights.items())[:self.max_top_n])
                total_weight = sum(target_weights_unnormalized.values())

                if total_weight > 1e-6:
                    target_weights = {sec: w / total_weight for sec, w in target_weights_unnormalized.items()}
                    print(f"[LLMStrategy] 最终目标权重 ({len(target_weights)}个, 已归一化): { {k: f'{v:.2%}' for k, v in target_weights.items()} }")
                else:
                    print("[LLMStrategy] AI 推荐的有效权重总和过小，清空仓位。")
                    target_weights = {}
            else:
                print("[LLMStrategy] AI 未推荐任何有效权重，清空仓位。")
                target_weights = {}

        except json.JSONDecodeError:
            print(f"[LLMStrategy] AI 返回的不是有效的 JSON 格式: >>>{raw_output}<<<。本期跳过交易。")
            justification = f"AI返回格式错误: {raw_output}"
            target_weights = {}
        except Exception as e:
            print(f"[LLMStrategy] 调用 API 或处理时发生错误: {e}。本期跳过交易。")
            import traceback
            traceback.print_exc()
            justification = f"API调用或处理错误: {e}"
            target_weights = {}

        # 记录到日志
        self.trade_log.append({
            "date": current_date,
            "justification": justification,
            "ai_recommendation": ai_recommendation,
            "final_weights": target_weights
        })

        return target_weights

    def _create_prompt(self, current_date: pd.Timestamp, factor_data: pd.DataFrame, portfolio_state: dict) -> List[Dict]:
        """
        构建发送给 DeepSeek API 的 Prompt (要求返回权重字典)。
        """
        factor_csv = factor_data.to_csv(float_format='%.3f')
        num_assets = len(factor_data)

        # --- 【修正点】将 JSON 示例字符串单独定义 ---
        example_json_str = '{ "SPY.P": 0.4, "QQQ.O": 0.35, "XLK.P": 0.25 }'

        # 系统提示：更新规则，要求返回权重字典
        system_prompt = f"""
        你是一位顶尖的量化投资组合经理，专精于美国 ETF 市场。
        你的目标是：通过分析提供的技术因子数据，构建一个**最优的**做多投资组合，以在未来一个月实现最大化利润 (Maximize PnL)。
        你将收到当前日期、当前持仓（仅供参考），以及资产池 '{self.universe_to_trade}' 中所有可用 ETF ({num_assets} 支) 的技术因子快照（EMA, MACD, RSI）。

        因子解释:
        - ema_12, ema_26: 短期和长期指数移动平均线。价格在均线之上通常看涨，金叉 (短期上穿长期) 是买入信号。
        - macd_line_12_26: MACD 快线 (EMA12 - EMA26)。正值且上升表示强劲上升趋势。
        - macd_signal_9: MACD 信号线 (MACD快线的9期EMA)。快线在慢线上方且距离拉大是看涨信号。
        - macd_hist_12_26_9: MACD 柱状图 (快线 - 慢线)。柱状图在零轴上方且放大是强烈的看涨信号。
        - rsi_14: 相对强弱指数 (0-100)。低于 30 通常表示超卖（可能反弹），高于 70 通常表示超买（可能回调）。50 是中性区域。

        你的任务与规则:
        1.  **核心任务**: 分析下面 CSV 表格中的因子数据，识别出短期上涨潜力最大的 ETF **并决定它们的投资权重**。关注点：
            * **趋势强度**: 结合 EMA 和 MACD 判断趋势的强度和方向。
            * **动量信号**: MACD 柱状图和 RSI 是否支持短期动量？
            * **反转信号**: RSI 是否指示超卖？结合其他指标判断反转可能性。
            * **综合判断**: 基于所有因子，评估每支 ETF 的风险收益潜力。
        2.  **策略**: 只做多 (Long-Only)。
        3.  **选股数量**: 你可以决定持有 **1 到 {self.max_top_n} 支** ETF。如果好的机会很少，可以选择少于 {self.max_top_n} 支，甚至可以选择持有 0 支 (空仓)。
        4.  **权重分配**: 为你选中的每一支 ETF 分配一个**具体的投资权重 (0 到 1 之间的小数)**。权重总和**必须大约等于 1.0** (允许小的浮点误差)。权重的大小应反映你对该 ETF 未来表现的信心以及风险的考量。
        5.  **输出格式**: 你的回应**必须**是一个有效的 JSON 对象。
        6.  **JSON 结构**:
            * `"justification"` (string): 简要说明（1-3句话）你构建这个投资组合（选股和权重分配）的主要原因，**必须**基于提供的因子数据。
            # --- 【修正点】使用变量嵌入示例字符串 ---
            * `"target_weights"` (dict): 一个字典，其中键 (key) 是你选择的 ETF 代码 (sec_code)，值 (value) 是对应的目标权重 (0到1之间的小数)。例如： `{example_json_str}`。如果决定空仓，返回空字典 `{{}}`。
        7.  **目标**: 最大化 PnL。请基于因子数据做出最有利可图的投资组合决策（选股 + 权重）。
        """

        # 用户提示：提供具体的數據
        user_prompt = f"""
        ### 分析背景
        - 今天日期: {current_date.strftime('%Y-%m-%d')}
        - 你的目标: 从以下 {num_assets} 支 ETF 中，构建最优投资组合 (最多 {self.max_top_n} 支，权重和为 1.0)，以最大化未来一个月 PnL。
        - 当前持仓 (仅供参考): {list(portfolio_state.get('current_positions', {}).keys())}

        ### ETF 技术因子快照 (资产池: '{self.universe_to_trade}', CSV 格式)
        ```csv
        {factor_csv}
        ```

        ### 你的决策
        请根据上述因子数据，构建你的目标投资组合，并以要求的 JSON 格式返回 `"justification"` 和 `"target_weights"`。
        """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def get_trade_log(self) -> pd.DataFrame:
        """返回 AI 决策日志的 DataFrame"""
        if not self.trade_log:
            return pd.DataFrame(columns=['date', 'justification', 'ai_recommendation', 'final_weights'])
        return pd.DataFrame(self.trade_log)