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

    def get_required_factors(self) -> List[str]:
        """
        (可选) 返回一个列表，包含此策略运行所需的因子名称。
        如果未实现，引擎将使用默认因子或配置。
        """
        return []


class LLMStrategy(BaseStrategy):
    """
    【升级版】基于大型语言模型 (DeepSeek) 决策的策略。
    它能动态接收来自 config.yaml 的因子列表和解释。
    """
    
    # --- 【修改点 1】: 更新 __init__ 签名以接收 **kwargs ---
    def __init__(self, universe_df: pd.DataFrame, **kwargs):
        """
        初始化 LLM 策略。
        现在从 kwargs 动态接收配置。

        Args:
            universe_df (pd.DataFrame): 资产池定义。
            **kwargs: 必须包含:
                - 'universe_to_trade' (str): AI 关注的资产池。
                - 'top_n' (int): AI 最多可选的 ETF 数量。
                - 'factors_to_use' (List[str]): 策略应使用的因子名称列表。
                - 'factor_explanations' (Dict[str, str]): 因子解释字典。
                - 'api_key' (str, optional): API 密钥。
        """
        super().__init__(universe_df, **kwargs)
        
        # 从 kwargs 中解析参数
        self.universe_to_trade = kwargs.get('universe_to_trade', 'equity_us')
        self.max_top_n = kwargs.get('top_n', 5)
        
        # --- 【修改点 2】: 存储来自 config.yaml 的因子配置 ---
        self.factors_to_use = kwargs.get('factors_to_use')
        if not self.factors_to_use:
             raise ValueError("LLMStrategy: 'factors_to_use' (因子列表) 未在配置中提供。")
             
        self.factor_explanations = kwargs.get('factor_explanations')
        if not self.factor_explanations:
             raise ValueError("LLMStrategy: 'factor_explanations' (因子解释) 未在配置中提供。")
        # --------------------------------------------------

        self.api_key = kwargs.get('api_key') or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未提供。请设置 DEEPSEEK_API_KEY 环境变量或在初始化时传入 api_key。")

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        except Exception as e:
            raise RuntimeError(f"使用 OpenAI SDK 初始化 DeepSeek 连接失败: {e}")

        self.trade_log: List[Dict] = []

    # --- 【修改点 3】: 实现 get_required_factors ---
    def get_required_factors(self) -> List[str]:
        """
        告诉 BacktestEngine 需要从 FactorEngine 请求哪些因子。
        """
        return self.factors_to_use
    # ----------------------------------------------

    def get_target_weights(self, current_date: pd.Timestamp, factor_snapshot: pd.DataFrame, portfolio_state: dict) -> Dict[str, float]:
        """
        调用 LLM API 获取目标 ETF 及其权重。
        (此函数内部逻辑保持不变)
        """
        print(f"\n[LLMStrategy] 正在 {current_date.date()} 为资产池 '{self.universe_to_trade}' 请求 AI 决策 (含权重)...")

        # 1. 确定本次决策的资产列表
        if self.universe_to_trade.lower() == 'all':
            assets_in_scope = factor_snapshot.index.tolist()
        else:
            assets_in_scope = self.universe_df[self.universe_df['universe'] == self.universe_to_trade]['sec_code'].tolist()

        # --- 【修改点 4】: 确保只向 AI 发送请求的因子 ---
        # (重要) factor_snapshot 可能包含比 self.factors_to_use 更多的因子 (如果引擎缓存了)
        # 我们只应选择 config.yaml 中指定的因子发送给 AI
        factors_to_send = [col for col in self.factors_to_use if col in factor_snapshot.columns]
        current_factor_data = factor_snapshot.loc[factor_snapshot.index.isin(assets_in_scope), factors_to_send]
        # -------------------------------------------------

        if current_factor_data.empty or current_factor_data.columns.empty:
            print(f"[LLMStrategy] 警告: 在 {current_date.date()} 未找到资产池 '{self.universe_to_trade}' 的有效因子数据。清空仓位。")
            self.trade_log.append({
                "date": current_date, "justification": "无有效因子数据，清仓。",
                "ai_recommendation": {}, "final_weights": {}
            })
            return {}

        # 2. 构建发送给 AI 的 Prompt (现在是动态的)
        prompt_messages = self._create_prompt(current_date, current_factor_data, portfolio_state)

        # 3. 调用 API (保持不变)
        target_weights: Dict[str, float] = {}
        ai_recommendation: Dict[str, float] = {}
        justification = "AI 决策失败或跳过。"
        raw_output = ""

        try:
            print(f"[LLMStrategy] 正在调用 DeepSeek API (via OpenAI SDK)...")
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

            # 4. 解析 AI 决策 (保持不变)
            justification = decision.get("justification", "AI 未提供理由。")
            ai_recommendation = decision.get("target_weights", {})

            valid_weights: Dict[str, float] = {}
            if isinstance(ai_recommendation, dict):
                for code, weight in ai_recommendation.items():
                    if code in current_factor_data.index and isinstance(weight, (int, float)) and weight > 1e-6:
                         valid_weights[code] = float(weight)
            else:
                 print(f"[LLMStrategy] 警告: AI 返回的 target_weights 不是字典格式: {ai_recommendation}")

            print(f"[LLMStrategy] AI 理由: {justification}")
            print(f"[LLMStrategy] AI 推荐权重 (有效): { {k: f'{v:.2%}' for k, v in valid_weights.items()} }")

            # 5. 权重处理与归一化 (保持不变)
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

        # 记录到日志 (保持不变)
        self.trade_log.append({
            "date": current_date,
            "justification": justification,
            "ai_recommendation": ai_recommendation,
            "final_weights": target_weights
        })

        return target_weights

    # --- 【修改点 5】: 动态构建 Prompt ---
    def _create_prompt(self, current_date: pd.Timestamp, factor_data: pd.DataFrame, portfolio_state: dict) -> List[Dict]:
        """
        【升级版】构建发送给 DeepSeek API 的 Prompt。
        因子解释现在从 self.factor_explanations 动态生成。
        """
        # factor_data 现在只包含您想要的因子
        factor_csv = factor_data.to_csv(float_format='%.3f')
        num_assets = len(factor_data)
        
        # --- 动态生成因子解释 ---
        explanation_lines = []
        for factor_name in factor_data.columns: # 遍历 AI 实际收到的因子
             explanation = self.factor_explanations.get(factor_name, "N/A - 未提供解释。")
             explanation_lines.append(f"- {factor_name}: {explanation}")
        
        if not explanation_lines:
             factor_explanation_str = "未提供任何因子解释。"
        else:
             factor_explanation_str = "\n".join(explanation_lines)
        # --- 结束 ---

        example_json_str = '{ "SPY.P": 0.4, "QQQ.O": 0.35, "XLK.P": 0.25 }'

        system_prompt = f"""
        你是一位顶尖的量化投资组合经理，专精于 ETF 市场。
        你的目标是：通过分析提供的量化因子数据，构建一个**最优的**做多投资组合，以在未来一个月实现最大化利润 (Maximize PnL)。
        你将收到当前日期、当前持仓（仅供参考），以及资产池 '{self.universe_to_trade}' 中所有可用 ETF ({num_assets} 支) 的因子快照。

        因子解释:
        {factor_explanation_str}

        你的任务与规则:
        1.  **核心任务**: 分析下面 CSV 表格中的因子数据，识别出短期上涨潜力最大的 ETF **并决定它们的投资权重**。
            * **因子值**: 因子值通常越高越好（代表信号越强），除非解释中另有说明（例如，反转因子可能值越低越好，但我们已在计算时处理为值越高越好）。
            * **综合判断**: 基于所有因子，评估每支 ETF 的风险收益潜力。
        2.  **策略**: 只做多 (Long-Only)。
        3.  **选股数量**: 你可以决定持有 **1 到 {self.max_top_n} 支** ETF。如果好的机会很少，可以选择少于 {self.max_top_n} 支，甚至可以选择持有 0 支 (空仓)。
        4.  **权重分配**: 为你选中的每一支 ETF 分配一个**具体的投资权重 (0 到 1 之间的小数)**。权重总和**必须大约等于 1.0**。权重的大小应反映你对该 ETF 未来表现的信心。
        5.  **输出格式**: 你的回应**必须**是一个有效的 JSON 对象。
        6.  **JSON 结构**:
            * `"justification"` (string): 简要说明（1-3句话）你构建这个投资组合（选股和权重分配）的主要原因，**必须**基于提供的因子数据。
            * `"target_weights"` (dict): 一个字典，其中键 (key) 是你选择的 ETF 代码 (sec_code)，值 (value) 是对应的目标权重。例如： `{example_json_str}`。如果决定空仓，返回空字典 `{{}}`。
        7.  **目标**: 最大化 PnL。请基于因子数据做出最有利可图的投资组合决策。
        """

        user_prompt = f"""
        ### 分析背景
        - 今天日期: {current_date.strftime('%Y-%m-%d')}
        - 你的目标: 从以下 {num_assets} 支 ETF 中，构建最优投资组合 (最多 {self.max_top_n} 支，权重和为 1.0)，以最大化未来一个月 PnL。
        - 当前持仓 (仅供参考): {list(portfolio_state.get('current_positions', {}).keys())}

        ### ETF 量化因子快照 (资产池: '{self.universe_to_trade}', CSV 格式)
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