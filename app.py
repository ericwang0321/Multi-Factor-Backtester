import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import os
import io
from datetime import datetime

# Existing imports
from llm_quant_lib.data.query_helper import DataQueryHelper
from llm_quant_lib.data_handler import DataHandler
from llm_quant_lib.strategy import FactorTopNStrategy
from llm_quant_lib.backtest_engine import BacktestEngine
from llm_quant_lib.performance import calculate_extended_metrics

# New Analysis Imports
from llm_quant_lib.analysis.task_runner import FactorTaskRunner

# --- Page Setup ---
st.set_page_config(page_title="Multi-Factor Backtest App", layout="wide")

# --- Resource Caching ---
@st.cache_resource
def load_framework(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 保持原有逻辑：加载价格数据
    dh = DataHandler(csv_path=config['paths']['price_data_csv'], start_date="2016-01-01", end_date="2025-12-31")
    dh.load_data()
    return dh, config

@st.cache_resource
def get_query_helper():
    """New helper for Parquet data visualization"""
    return DataQueryHelper(storage_path='data/processed/all_price_data.parquet')

@st.cache_resource
def get_analysis_runner(_dh):
    """Initialize the factor EDA orchestrator"""
    return FactorTaskRunner(_dh)

# --- Module: Data Explorer ---
def render_data_explorer():
    st.header("Data Warehouse Explorer")
    helper = get_query_helper()
    
    # 1. Market Coverage Metrics
    summary = helper.get_market_summary()
    st.subheader("Market Coverage")
    if not summary.empty:
        cols = st.columns(len(summary))
        for i, row in summary.iterrows():
            cols[i].metric(label=row['category_id'].upper(), value=f"{row['count']} Tickers")

    st.divider()

    # 2. Dual-Level Asset Selector
    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.subheader("Asset Selector")
        all_assets = helper.get_all_symbols()
        
        # 第一步：选择资产组
        groups = sorted(all_assets['category_id'].unique())
        selected_group = st.selectbox("Select Group", ["All Groups"] + list(groups))
        
        # 第二步：根据组别过滤标的
        if selected_group != "All Groups":
            filtered_list = all_assets[all_assets['category_id'] == selected_group]['sec_code'].unique()
        else:
            filtered_list = all_assets['sec_code'].unique()
            
        selected_symbol = st.selectbox("Select Security", sorted(filtered_list))

    with col_r:
        if selected_symbol:
            df = helper.get_history(selected_symbol)
            fig = px.line(df, x='datetime', y='close', title=f"{selected_symbol} Historical Price")
            fig.update_layout(template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            tab1, tab2 = st.tabs(["Volume Analysis", "Data Table"])
            with tab1:
                st.plotly_chart(px.bar(df, x='datetime', y='volume'), use_container_width=True)
            with tab2:
                st.dataframe(df.sort_values('datetime', ascending=False), use_container_width=True)

# --- New Module: Analysis Explorer ---
def render_analysis_explorer(dh):
    st.header("Factor Analysis Explorer")
    st.info("Perform cross-sectional EDA to evaluate factor predictive power and monotonicity.")
    
    runner = get_analysis_runner(dh)
    
    # 1. Selection and Config
    col_a, col_b = st.columns(2)
    with col_a:
        # Get factors directly from your Registry
        factor_list = sorted(list(runner.factor_engine.FACTOR_REGISTRY.keys()))
        selected_factor = st.selectbox("Select Factor for EDA", factor_list)
    with col_b:
        horizon = st.number_input("Forward Return Horizon (Days)", 1, 20, 1)

    if st.button("Run One-Click Analysis", type="primary"):
        with st.spinner(f"Computing metrics for {selected_factor}..."):
            # Execute Preprocessing -> IC -> Quantiles
            stats, ic_series, cum_group_ret = runner.run_analysis_pipeline(selected_factor, horizon=horizon)
            
            # Save to state for persistence
            st.session_state.analysis_ready = True
            st.session_state.ana_stats = stats
            st.session_state.ana_ic = ic_series
            st.session_state.ana_groups = cum_group_ret
            st.session_state.ana_name = selected_factor

    if st.session_state.get('analysis_ready'):
        s = st.session_state.ana_stats
        
        # 2. Key Stats
        st.divider()
        st.subheader(f"Statistical Summary: {st.session_state.ana_name}")
        st.markdown(f"Metrics calculated using **Rank IC** against **T+{horizon}** returns.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean IC", f"{s['IC Mean']:.4f}")
        c2.metric("IC Std", f"{s['IC Std']:.4f}")
        c3.metric("IR (Info Ratio)", f"{s['IR']:.4f}")
        c4.metric("IC > 0 Rate", f"{s['IC > 0 Rate']:.2%}")

        # 3. IC Time Series
        st.subheader("IC Time Series")
        fig_ic = px.area(st.session_state.ana_ic, title="Daily Rank IC", labels={'value': 'IC', 'datetime': 'Date'})
        fig_ic.update_layout(template="plotly_white")
        st.plotly_chart(fig_ic, use_container_width=True)

        # 4. Monotonicity / Group Analysis
        st.subheader("Quantile Analysis (Cumulative Group Returns)")
        st.write("Assets split into 5 groups daily based on factor value. Group 4 is the highest.")
        fig_groups = px.line(st.session_state.ana_groups, title="Group Wealth Curves", template="plotly_white")
        st.plotly_chart(fig_groups, use_container_width=True)

# --- Sidebar: Navigation & Parameters ---
with st.sidebar:
    st.header("Navigation")
    # New Module added to radio
    app_mode = st.radio("Choose Module", ["Strategy Explorer", "Data Explorer", "Analysis Explorer"])
    
    st.divider()
    
    # Shared resource across modules
    dh, config = load_framework('config.yaml')
    
    if app_mode == "Strategy Explorer":
        st.header("Parameters")
        bench_options = {
            "S&P 500 (SPXT)": "spxt_index_daily_return.csv",
            "Global Equity (MXWD)": "mxwd_index_daily_return.csv",
            "Commodity (BCOM)": "bcom_index_daily_return.csv",
            "Global Bond": "global_bond_index_daily_return.csv"
        }
        selected_bench_label = st.selectbox("Compare against Benchmark", list(bench_options.keys()))
        
        available_factors = sorted(list(dh.price_df.columns)) if hasattr(dh, 'price_df') else []
        if not available_factors:
             available_factors = ['trend_score', 'momentum', 'volatility', 'turnover_mean', 'alpha001', 'rsi', 'amount_mean']
             
        selected_factors = st.multiselect("Select Factors", available_factors, default=['trend_score', 'momentum'])
        
        factor_weights = {}
        if selected_factors:
            st.write("Set Factor Weights:")
            for f in selected_factors:
                factor_weights[f] = st.number_input(f"Weight: {f}", 0.0, 1.0, 1.0/len(selected_factors), 0.05)

        st.divider()
        st.header("Costs & Execution")
        comm_rate = st.number_input("Commission Rate", 0.0, 0.01, 0.0010, format="%.4f")
        slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, format="%.4f")
        rebalance_days = st.slider("Rebalance Frequency (Trading Days)", 1, 60, 20)
        
        col_s, col_e = st.columns(2)
        start_date = col_s.date_input("Start", datetime(2018, 1, 1))
        end_date = col_e.date_input("End", datetime(2024, 7, 31))
        
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Main App Logic ---
if app_mode == "Data Explorer":
    render_data_explorer()

elif app_mode == "Analysis Explorer":
    render_analysis_explorer(dh)

elif app_mode == "Strategy Explorer":
    st.title("Quantitative Strategy Explorer")
    
    if run_btn:
        if not selected_factors:
            st.error("Error: Please select at least one factor.")
        else:
            with st.spinner('Running simulation...'):
                bt_config = {'INITIAL_CAPITAL': 1000000, 'COMMISSION_RATE': comm_rate, 'SLIPPAGE': slip_rate, 'REBALANCE_DAYS': rebalance_days}
                u_df = dh.load_universe_data(config['paths']['universe_definition'])
                strategy = FactorTopNStrategy(universe_df=u_df, factor_weights=factor_weights, top_n=5)
                engine = BacktestEngine(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), config=bt_config, strategy=strategy, data_handler=dh)
                engine.factor_engine.current_weights = factor_weights 
                equity_df, final_portfolio = engine.run()

                # Base Benchmarking
                bench_file = bench_options[selected_bench_label]
                bench_path = os.path.join("data", "processed", bench_file)
                b_raw = pd.read_csv(bench_path)
                b_raw['report_date'] = pd.to_datetime(b_raw['report_date'])
                b_raw = b_raw.set_index('report_date').sort_index()
                b_rets = b_raw.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'), 'default']
                benchmark_equity = (1 + b_rets).cumprod() * bt_config['INITIAL_CAPITAL']

                metrics = calculate_extended_metrics(equity_df['total_value'], benchmark_equity, final_portfolio)
                
                st.session_state.bt_ready = True
                st.session_state.equity_df = equity_df
                st.session_state.metrics = metrics
                st.session_state.strategy = strategy
                st.session_state.final_portfolio = final_portfolio
                st.session_state.engine = engine
                st.session_state.selected_factors = selected_factors
                st.session_state.bench_label = selected_bench_label

    if st.session_state.get('bt_ready'):
        m = st.session_state.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha (Excess)", f"{m.get('Alpha', 0):+.2%}")
        c2.metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
        c3.metric("Info Ratio", f"{m.get('Info Ratio', 0):.2f}")
        c4.metric("Beta", f"{m.get('Beta', 0):.2f}")

        st.divider()
        st.subheader("Transaction Cost Attribution")
        ct1, ct2, ct3, ct4 = st.columns(4)
        ct1.metric("Total Cost", f"${m.get('Total Cost', 0):,.0f}")
        ct2.metric("Commission", f"${m.get('Commission', 0):,.0f}")
        ct3.metric("Slippage", f"${m.get('Slippage', 0):,.0f}")
        ct4.metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            summary_df = pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value'])
            summary_df.to_excel(writer, sheet_name='Summary')
            pd.DataFrame({'Strategy': m['strategy_curve'], 'Benchmark': m['benchmark_curve'], 'Excess': m['excess_curve']}).to_excel(writer, sheet_name='Comparison')
        st.download_button("Download Excel Report", buffer.getvalue(), f"Backtest_Report.xlsx", use_container_width=True)

        st.subheader(f"Strategy vs {st.session_state.bench_label}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['strategy_curve'].index, y=m['strategy_curve'], name='Strategy', line=dict(color='#0B3D59', width=2.5)))
        fig.add_trace(go.Scatter(x=m['benchmark_curve'].index, y=m['benchmark_curve'], name=st.session_state.bench_label, line=dict(color='#5EA9CE', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=m['excess_curve'].index, y=m['excess_curve'], name='Excess Return', yaxis='y2', fill='tozeroy', line=dict(color='#8E44AD', width=1.5), fillcolor='rgba(142, 68, 173, 0.2)'))        
        # 4. 布局配置 (这是修复双轴显示的关键！)
        fig.update_layout(
            hovermode="x unified", 
            template="plotly_white",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            ),
            # --- 必须显式定义两个 Y 轴 ---
            # 左侧 Y 轴配置 (净值)
            yaxis=dict(
                title=dict(text="Normalized Value", font=dict(color="#0B3D59")), 
                tickfont=dict(color="#0B3D59")
            ),
            # 右侧 Y 轴配置 (超额收益)
            yaxis2=dict(
                title=dict(text="Cumulative Excess Return", font=dict(color="#8E44AD")), 
                tickfont=dict(color="#8E44AD"), 
                overlaying="y",  # <--- 关键：声明覆盖在主轴上
                side="right"     # <--- 关键：放置在右侧
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        nav_options = ["Performance", "Signals", "Holdings", "Factor Correlation", "Risk Analysis"]
        active_tab = st.radio("Analysis View", nav_options, horizontal=True, key="active_nav_tab")

        if active_tab == "Performance":
            st.table(pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value']).astype(str))
        elif active_tab == "Signals":
            st.dataframe(st.session_state.strategy.get_trade_log(), use_container_width=True)
        elif active_tab == "Holdings":
            st.dataframe(st.session_state.final_portfolio.get_holdings_history(), use_container_width=True)
        elif active_tab == "Factor Correlation":
            st.subheader("Dynamic Factor Correlation Analysis")
            current_factors = st.session_state.get('selected_factors', [])
            if len(current_factors) > 1:
                a_range = st.slider("Select Analysis Period", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM-DD", key="corr_slider")
                f_list = []
                for fn in current_factors:
                    if fn in st.session_state.engine.factor_engine._factor_cache:
                        f_cache = st.session_state.engine.factor_engine._factor_cache[fn]
                        f_slice = f_cache.loc[a_range[0].strftime('%Y-%m-%d'):a_range[1].strftime('%Y-%m-%d')].stack()
                        f_slice.name = fn
                        f_list.append(f_slice)
                if f_list:
                    corr_m = pd.concat(f_list, axis=1).corr()
                    st.plotly_chart(px.imshow(corr_m, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
        elif active_tab == "Risk Analysis":
            st.subheader("Daily Risk Exposure (95% Confidence)")
            if 'rolling_var_series' in m:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=m['rolling_var_series'].index, y=m['rolling_var_series'].values * 100, fill='tozeroy', name='95% Rolling VaR', line=dict(color='rgba(255, 0, 0, 0.6)')))
                fig_r.update_layout(yaxis_title="Potential Loss (%)", template="plotly_white")
                st.plotly_chart(fig_r, use_container_width=True)
                st.markdown(f"**Metrics**: 95% Historical VaR: **{abs(m.get('VaR_95', 0)):.2%}**, 95% ES: **{abs(m.get('ES_95', 0)):.2%}**.")