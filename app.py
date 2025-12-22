import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # 【新增】用于绘制归因图表
import yaml
import os
from datetime import datetime

# Core framework imports
from llm_quant_lib.data_handler import DataHandler
from llm_quant_lib.strategy import FactorTopNStrategy
from llm_quant_lib.backtest_engine import BacktestEngine
from llm_quant_lib.performance import calculate_extended_metrics

# --- Page Setup ---
st.set_page_config(page_title="Multi-Factor Backtest App", layout="wide")
st.title("Quantitative Strategy Explorer")

# --- Resource Caching ---
@st.cache_resource
def load_framework(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dh = DataHandler(
        csv_path=config['paths']['price_data_csv'],
        start_date="2016-01-01",
        end_date="2025-12-31"
    )
    dh.load_data()
    return dh, config

# --- Sidebar: Interactive Parameters ---
with st.sidebar:
    st.header("Strategy parameters")
    dh, config = load_framework('config.yaml')
    
    # 1. Factor Selection
    available_factors = [
        'trend_score', 'momentum', 'volatility', 'turnover_mean', 
        'alpha001', 'breakout_quality_score', 'rsi', 'stochastic_k',
        'amount_mean', 'amihud_illiquidity'
    ]
    selected_factors = st.multiselect(
        "Select Factors", 
        available_factors, 
        default=['trend_score', 'momentum']
    )
    
    # 2. Dynamic Weight Inputs
    factor_weights = {}
    if selected_factors:
        st.write("Set Factor Weights:")
        for factor in selected_factors:
            weight = st.number_input(
                f"Weight: {factor}", 
                min_value=0.0, max_value=1.0, 
                value=1.0/len(selected_factors), 
                step=0.05
            )
            factor_weights[factor] = weight

    # 【新增】3. Transaction Cost Settings (交互式成本设置)
    st.divider()
    st.header("Transaction Costs")
    comm_rate = st.number_input("Commission Rate", 0.0, 0.01, 0.001, format="%.4f")
    slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, format="%.4f")

    # 4. Execution Parameters
    rebalance_days = st.slider("Rebalance Frequency (Trading Days)", 1, 60, 20)
    top_n = st.number_input("Top N Selection", 1, 15, 5)
    
    # 5. Dates
    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Start", datetime(2018, 1, 1))
    end_date = col_end.date_input("End", datetime(2024, 7, 31))
    
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Logic Execution ---
if run_btn:
    if not selected_factors:
        st.error("Error: Please select at least one factor.")
    else:
        bt_config = {
            'INITIAL_CAPITAL': config['backtest'].get('initial_capital', 1000000),
            'COMMISSION_RATE': comm_rate, # 【修改】使用侧边栏输入
            'SLIPPAGE': slip_rate,       # 【修改】使用侧边栏输入
            'REBALANCE_DAYS': rebalance_days
        }

        with st.spinner('Running multi-factor simulation...'):
            u_df = dh.load_universe_data(config['paths']['universe_definition'])
            
            strategy = FactorTopNStrategy(
                universe_df=u_df,
                factor_weights=factor_weights,
                top_n=top_n,
                ascending=False
            )
            
            engine = BacktestEngine(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                config=bt_config,
                strategy=strategy,
                data_handler=dh
            )
            
            engine.factor_engine.current_weights = factor_weights 
            equity_df, final_portfolio = engine.run()
            
            metrics = calculate_extended_metrics(
                portfolio_equity=equity_df['total_value'],
                benchmark_equity=equity_df['total_value'],
                portfolio_instance=final_portfolio
            )

        # --- Visualization ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Return", f"{metrics.get('总回报率', 0):.2%}")
        m_col2.metric("Annual Return", f"{metrics.get('年化回报率', 0):.2%}")
        m_col3.metric("Sharpe", f"{metrics.get('夏普比率', 0):.2f}")
        m_col4.metric("Max Drawdown", f"{metrics.get('最大回撤', 0):.2%}")

        # 【新增】交易成本归因看板 (Cost Attribution Summary)
        st.divider()
        st.subheader("Transaction Cost Attribution")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cost", f"¥{metrics.get('总交易成本', 0):,.0f}")
        c2.metric("Commission", f"¥{metrics.get('累计佣金支出', 0):,.0f}")
        c3.metric("Slippage", f"¥{metrics.get('累计滑点支出', 0):,.0f}")
        drag = metrics.get('交易成本对收益损耗', 0)
        c4.metric("Return Drag", f"-{drag:.2%}", delta_color="inverse")

        # 【新增】归因可视化图表
        col_pie, col_bar = st.columns(2)
        with col_pie:
            cost_df = pd.DataFrame({
                'Component': ['Commission', 'Slippage'],
                'Amount': [metrics.get('累计佣金支出', 0), metrics.get('累计滑点支出', 0)]
            })
            fig_pie = px.pie(cost_df, values='Amount', names='Component', title="Cost Breakdown", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            comparison_df = pd.DataFrame({
                'Type': ['Actual Return', 'Theoretical (No Cost)'],
                'Value': [metrics.get('总回报率', 0) * 100, metrics.get('理论无成本总回报', 0) * 100]
            })
            fig_bar = px.bar(comparison_df, x='Type', y='Value', text_auto='.2f', 
                             title="Actual vs Theoretical Return (%)", color='Type')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df['total_value'], 
            name='Equity Curve',
            line=dict(color='#00d1b2', width=2)
        ))
        fig.update_layout(hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Data Detail Tabs
        t1, t2, t3 = st.tabs(["Performance Metrics", "Decision Log", "Holdings"])
        with t1:
            st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).astype(str))
        with t2:
            st.dataframe(strategy.get_trade_log(), use_container_width=True)
        with t3:
            st.dataframe(final_portfolio.get_holdings_history(), use_container_width=True)