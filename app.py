import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import os
from datetime import datetime

# Import core modules
from llm_quant_lib.data_handler import DataHandler
from llm_quant_lib.strategy import FactorTopNStrategy
from llm_quant_lib.backtest_engine import BacktestEngine
from llm_quant_lib.performance import calculate_extended_metrics

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Factor Backtest Dashboard", layout="wide")
st.title("Interactive Multi-Factor Dashboard")

# --- Data Loading (Cached) ---
@st.cache_resource
def get_cached_data_handler(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    dh = DataHandler(
        csv_path=config['paths']['price_data_csv'],
        start_date="2016-01-01",
        end_date="2025-12-31"
    )
    dh.load_data()
    return dh, config

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Strategy Settings")
    dh, default_cfg = get_cached_data_handler('config.yaml')
    
    # NEW: Multi-select box for factors
    available_factors = [
        'trend_score', 'momentum', 'volatility', 'turnover_mean', 
        'alpha001', 'breakout_quality_score', 'rsi', 'stochastic_k',
        'amount_mean', 'amihud_illiquidity'
    ]
    selected_factors = st.multiselect(
        "Select Factors to Combine", 
        available_factors, 
        default=['trend_score']
    )
    
    rebalance_days = st.slider("Rebalancing Days", 1, 60, 20)
    top_n = st.number_input("Top N Assets", 1, 20, 5)
    
    start_dt = st.date_input("Start Date", datetime(2018, 1, 1))
    end_dt = st.date_input("End Date", datetime(2024, 7, 31))
    
    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Main Execution ---
if run_button:
    if not selected_factors:
        st.error("Please select at least one factor.")
    else:
        backtest_config = {
            'INITIAL_CAPITAL': default_cfg['backtest'].get('initial_capital', 1000000),
            'COMMISSION_RATE': default_cfg['backtest'].get('commission_rate', 0.001),
            'SLIPPAGE': default_cfg['backtest'].get('slippage', 0.0005),
            'REBALANCE_DAYS': rebalance_days
        }
        
        with st.spinner('Calculating normalized composite scores...'):
            universe_path = default_cfg['paths']['universe_definition']
            u_df = dh.load_universe_data(universe_path)
            
            # Pass the list of factors to the updated Strategy
            strat = FactorTopNStrategy(
                universe_df=u_df,
                factor_names=selected_factors,
                top_n=top_n,
                ascending=False
            )
            
            engine = BacktestEngine(
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d'),
                config=backtest_config,
                strategy=strat,
                data_handler=dh
            )
            
            equity_df, final_portfolio = engine.run()
            
            metrics = calculate_extended_metrics(
                portfolio_equity=equity_df['total_value'],
                benchmark_equity=equity_df['total_value'],
                portfolio_instance=final_portfolio
            )

        # --- Dashboard Visualization ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics.get('总回报率', 0):.2%}")
        col2.metric("Annual Return", f"{metrics.get('年化回报率', 0):.2%}")
        col3.metric("Sharpe Ratio", f"{metrics.get('夏普比率', 0):.2f}")
        col4.metric("Max Drawdown", f"{metrics.get('最大回撤', 0):.2%}")

        st.subheader("Equity Curve")
        # Displaying the combined factors in the chart legend
        factor_label = ", ".join(selected_factors) if len(selected_factors) <= 3 else "Multiple Factors"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df['total_value'], 
            name=f'Strategy ({factor_label})',
            line=dict(color='#00CC96', width=2.5)
        ))
        fig.update_layout(hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Performance", "Signals & Logs", "Portfolio History"])
        with tab1:
            st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).astype(str))
        with tab2:
            st.dataframe(strat.get_trade_log(), use_container_width=True)
        with tab3:
            st.dataframe(final_portfolio.get_holdings_history(), use_container_width=True)
else:
    st.info("Select your factors and rebalancing frequency to begin.")