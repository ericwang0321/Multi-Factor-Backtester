import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
import os
from datetime import datetime

# Import core modules from your library
from llm_quant_lib.data_handler import DataHandler
from llm_quant_lib.strategy import FactorTopNStrategy
from llm_quant_lib.backtest_engine import BacktestEngine
from llm_quant_lib.performance import calculate_extended_metrics

# --- Page Configuration ---
st.set_page_config(page_title="Quantitative Backtest Dashboard", layout="wide")
st.title("Factor Backtesting Dashboard")

# --- Data Loading (Cached) ---
@st.cache_resource
def get_cached_data_handler(config_path):
    """
    Loads and caches the DataHandler to prevent reloading large CSVs.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    price_path = config['paths']['price_data_csv']
    
    dh = DataHandler(
        csv_path=price_path,
        start_date="2016-01-01",
        end_date="2025-12-31"
    )
    dh.load_data()
    return dh, config

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Backtest Configuration")
    
    # Load defaults from config.yaml
    dh, default_cfg = get_cached_data_handler('config.yaml')
    
    # Factor Selection
    # List of factors registered in your FactorEngine
    factor_list = [
        'trend_score', 'momentum', 'volatility', 'turnover_mean', 
        'alpha001', 'breakout_quality_score', 'rsi', 'stochastic_k'
    ]
    selected_factor = st.selectbox("Select Factor", factor_list, index=0)
    
    # Parameters
    rebalance_days = st.slider("Rebalancing Frequency (Days)", 1, 60, 20)
    top_n = st.number_input("Number of Assets (Top N)", 1, 20, 5)
    
    # Date Range
    start_dt = st.date_input("Start Date", datetime(2018, 1, 1))
    end_dt = st.date_input("End Date", datetime(2024, 7, 31))
    
    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Main Execution ---
if run_button:
    # Prepare engine configuration
    backtest_config = {
        'INITIAL_CAPITAL': default_cfg['backtest'].get('initial_capital', 1000000),
        'COMMISSION_RATE': default_cfg['backtest'].get('commission_rate', 0.001),
        'SLIPPAGE': default_cfg['backtest'].get('slippage', 0.0005),
        'REBALANCE_DAYS': rebalance_days
    }
    
    with st.spinner('Calculating factors and running backtest...'):
        # Initialize Strategy
        universe_path = default_cfg['paths']['universe_definition']
        u_df = dh.load_universe_data(universe_path)
        
        strat = FactorTopNStrategy(
            universe_df=u_df,
            factor_name=selected_factor,
            top_n=top_n,
            ascending=False
        )
        
        # Initialize and Run Engine
        engine = BacktestEngine(
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_dt.strftime('%Y-%m-%d'),
            config=backtest_config,
            strategy=strat,
            data_handler=dh
        )
        
        equity_df, final_portfolio = engine.run()
        
        # Calculate Performance Metrics
        metrics = calculate_extended_metrics(
            portfolio_equity=equity_df['total_value'],
            benchmark_equity=equity_df['total_value'], # Placeholder for benchmark
            portfolio_instance=final_portfolio
        )

    # --- Results Display ---
    # Metric Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics.get('总回报率', 0):.2%}")
    col2.metric("Annual Return", f"{metrics.get('年化回报率', 0):.2%}")
    col3.metric("Sharpe Ratio", f"{metrics.get('夏普比率', 0):.2f}")
    col4.metric("Max Drawdown", f"{metrics.get('最大回撤', 0):.2%}")

    # Interactive Chart
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index, 
        y=equity_df['total_value'], 
        name='Strategy Value',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        hovermode="x unified", 
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Portfolio Value"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Data Tabs
    tab1, tab2, tab3 = st.tabs(["Metrics Detail", "Trade Log", "Holdings History"])
    
    with tab1:
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).astype(str))
        
    with tab2:
        st.dataframe(strat.get_trade_log(), use_container_width=True)
        
    with tab3:
        st.dataframe(final_portfolio.get_holdings_history(), use_container_width=True)

else:
    st.info("Configure the parameters in the sidebar and click 'Run Backtest' to see results.")