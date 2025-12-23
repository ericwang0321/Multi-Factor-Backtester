import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import os
import io
from datetime import datetime
import sys
import subprocess
import time

# --- 核心库导入 (适配新架构) ---
from quant_core.data.query_helper import DataQueryHelper
# [修改] 导入新的策略类
from quant_core.strategies.rules import LinearWeightedStrategy
from quant_core.backtest_engine import BacktestEngine
from quant_core.performance import calculate_extended_metrics
from quant_core.factors.engine import FactorEngine  # 需要用它来准备数据

# EDA 分析模块导入
from quant_core.analysis.task_runner import FactorTaskRunner

# --- Page Setup ---
st.set_page_config(page_title="Multi-Factor Backtest App", layout="wide")

# --- Resource Caching ---
@st.cache_resource
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

@st.cache_resource
def get_query_helper():
    """Single source of truth for Data"""
    # 确保路径指向您的 Parquet 文件
    return DataQueryHelper(storage_path='data/processed/all_price_data.parquet')

@st.cache_resource
def get_factor_engine(_query_helper):
    """
    获取因子引擎实例 (用于在 App 中临时计算因子)
    """
    return FactorEngine(query_helper=_query_helper)

@st.cache_resource
def get_analysis_runner(_query_helper):
    return FactorTaskRunner(_query_helper)

# [新增] 专门为新策略准备数据的函数
def prepare_factor_data_for_strategy(_engine, codes, factors, start_date, end_date):
    """
    在内存中计算选定因子的历史数据，并转换为 Strategy 需要的 MultiIndex DataFrame。
    """
    # 确保数据已初始化
    if _engine.xarray_data is None:
        _engine._get_xarray_data()
        
    data_dict = {}
    
    # 进度条
    progress_text = "Computing factors in-memory..."
    my_bar = st.progress(0, text=progress_text)
    
    total = len(factors)
    for i, f_name in enumerate(factors):
        # 调用 FactorEngine 计算全量历史
        df = _engine._compute_and_cache_factor(f_name)
        
        if not df.empty:
            # 截取时间段 (为了性能，虽然 Engine 算的是全量)
            # 转为 stack 格式: index=[datetime, sec_code], value=factor_value
            df_slice = df.loc[str(start_date):str(end_date)]
            # 过滤 Universe
            valid_cols = [c for c in df_slice.columns if c in codes]
            if valid_cols:
                stacked = df_slice[valid_cols].stack()
                stacked.name = f_name
                data_dict[f_name] = stacked
        
        my_bar.progress((i + 1) / total, text=f"Computed {f_name}")
    
    my_bar.empty()
    
    if not data_dict:
        return pd.DataFrame()
        
    # 合并为大宽表 (Index: datetime, sec_code; Columns: factor1, factor2...)
    full_factor_df = pd.concat(data_dict.values(), axis=1)
    
    # 确保索引名为 datetime, sec_code 以匹配 BaseStrategy 的逻辑
    full_factor_df.index.names = ['datetime', 'sec_code']
    
    return full_factor_df

# --- Module 1: Data Explorer ---
def render_data_explorer():
    st.header("Data Warehouse Explorer (Parquet)")
    helper = get_query_helper()
    
    summary = helper.get_market_summary()
    st.subheader("Market Coverage")
    if not summary.empty:
        cols = st.columns(len(summary))
        for i, row in summary.iterrows():
            cols[i].metric(label=row['category_id'].upper(), value=f"{row['count']} Tickers")

    st.divider()
    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.subheader("Asset Selector")
        all_assets = helper.get_all_symbols()
        
        groups = sorted(all_assets['category_id'].unique())
        selected_group = st.selectbox("Select Group", ["All Groups"] + list(groups))
        
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
            t1, t2 = st.tabs(["Volume Analysis", "Data Table"])
            with t1: st.plotly_chart(px.bar(df, x='datetime', y='volume'), use_container_width=True)
            with t2: st.dataframe(df.sort_values('datetime', ascending=False), use_container_width=True)

# --- Module 2: Analysis Explorer ---
def render_analysis_explorer(helper):
    st.header("Factor Analysis Explorer")
    runner = get_analysis_runner(helper)
    
    col_a, col_b = st.columns(2)
    with col_a:
        factor_list = sorted(list(runner.factor_engine.FACTOR_REGISTRY.keys()))
        selected_factor = st.selectbox("Select Factor for EDA", factor_list)
    with col_b:
        horizon = st.number_input("Forward Return Horizon (Days)", 1, 60, 20)

    if st.button("Run One-Click Analysis", type="primary"):
        with st.spinner(f"Processing {selected_factor}..."):
            stats, ic_series, cum_group_ret = runner.run_analysis_pipeline(selected_factor, horizon=horizon)
            st.session_state.ana_ready = True
            st.session_state.ana_stats = stats
            st.session_state.ana_ic = ic_series
            st.session_state.ana_groups = cum_group_ret
            st.session_state.ana_name = selected_factor
            st.session_state.ana_horizon = horizon

    if st.session_state.get('ana_ready'):
        s, h = st.session_state.ana_stats, st.session_state.ana_horizon
        st.divider()
        st.subheader(f"Stats: {st.session_state.ana_name} (T+{h})")
        if s:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean IC", f"{s['IC Mean']:.4f}")
            c2.metric("IC Std", f"{s['IC Std']:.4f}")
            c3.metric("IR", f"{s['IR']:.4f}")
            c4.metric("IC > 0 Rate", f"{s['IC > 0 Rate']:.2%}")
            
            st.subheader("Daily Rank IC")
            st.line_chart(st.session_state.ana_ic)
            
            st.subheader("Quantile Analysis (Fixed Wealth Curves)")
            st.plotly_chart(px.line(st.session_state.ana_groups, template="plotly_white"), use_container_width=True)
        else:
            st.warning("No valid statistics generated. Check data quality.")

# --- Sidebar & Main Logic ---
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Choose Module", ["Strategy Explorer", "Data Explorer", "Analysis Explorer"])
    
    config = load_config('config.yaml')
    helper = get_query_helper()
    
    if app_mode == "Strategy Explorer":
        st.header("Parameters")
        bench_options = {
            "S&P 500 (SPY)": "SPY", 
            "Global Equity (ACWI)": "ACWI", 
            "Global Bond (AGG)": "AGG", 
            "Commodities (GSG)": "GSG"
        }
        selected_bench_label = st.selectbox("Compare against Benchmark", list(bench_options.keys()))
        
        # 获取因子列表
        runner_temp = get_analysis_runner(helper)
        available_factors = sorted(list(runner_temp.factor_engine.FACTOR_REGISTRY.keys()))
        
        selected_factors = st.multiselect("Select Factors", available_factors, default=['momentum', 'rsi'])
        
        factor_weights = {f: st.number_input(f"Weight: {f}", 0.0, 1.0, 1.0/len(selected_factors), 0.05) for f in selected_factors} if selected_factors else {}
        
        st.divider()
        st.header("Costs & Execution")
        top_k = st.slider("Top K Stocks", 1, 20, 5)
        comm_rate = st.number_input("Commission Rate", 0.0, 0.01, 0.0010, format="%.4f")
        slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, format="%.4f")
        rebalance_days = st.slider("Rebalance Frequency", 1, 60, 20)
        col_s, col_r = st.columns(2)
        start_date = col_s.date_input("Start", datetime(2018, 1, 1))
        end_date = col_r.date_input("End", datetime(2024, 7, 31))
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
        
    st.markdown("---")
    with st.expander("📡 Data Status", expanded=False):
        try:
            h_temp = get_query_helper()
            mkt_summary = h_temp.get_market_summary()
            if not mkt_summary.empty:
                latest_date = mkt_summary['end'].max()
                st.caption(f"Data up to: **{latest_date.strftime('%Y-%m-%d')}**")
            else:
                st.caption("Data: Empty")
        except Exception:
            st.caption("Status: Unknown")

        if st.button("🔄 Sync Now", use_container_width=True):
            status_box = st.empty()
            status_box.info("⏳ Connecting to IBKR...")
            try:
                result = subprocess.run([sys.executable, "run_data_sync.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    status_box.success("✅ Complete!")
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    status_box.error("❌ Failed")
                    with st.expander("Log"): st.code(result.stderr)
            except Exception as e:
                status_box.error(f"Err: {str(e)}")

# --- Sidebar End ---

if app_mode == "Data Explorer": 
    render_data_explorer()
elif app_mode == "Analysis Explorer": 
    render_analysis_explorer(helper)
elif app_mode == "Strategy Explorer":
    st.title("Quantitative Strategy Explorer")
    
    if run_btn:
        if not selected_factors:
            st.error("Error: Please select at least one factor.")
        else:
            try:
                # --- A. 准备配置 ---
                bt_config = {
                    'INITIAL_CAPITAL': 1000000, 
                    'COMMISSION_RATE': comm_rate, 
                    'SLIPPAGE': slip_rate, 
                    'REBALANCE_DAYS': rebalance_days
                }
                
                # --- B. 数据准备 ---
                u_df = helper.get_all_symbols()
                universe_codes = u_df['sec_code'].tolist()
                
                # [核心步骤] 准备因子数据 (内存计算)
                f_engine = get_factor_engine(helper)
                factor_data = prepare_factor_data_for_strategy(
                    f_engine, universe_codes, selected_factors, start_date, end_date
                )
                
                if factor_data.empty:
                    st.error("No factor data generated. Please check data source.")
                else:
                    # --- C. 初始化新策略 ---
                    strategy = LinearWeightedStrategy(
                        name="App_Linear_Strategy", 
                        weights=factor_weights, 
                        top_k=top_k
                    )
                    
                    # [关键] 注入因子数据
                    strategy.load_data(factor_data)
                    
                    # --- D. 初始化引擎 ---
                    engine = BacktestEngine(
                        start_date=start_date.strftime('%Y-%m-%d'), 
                        end_date=end_date.strftime('%Y-%m-%d'), 
                        config=bt_config, 
                        strategy=strategy, 
                        query_helper=helper
                    )
                    
                    # --- E. 运行 ---
                    with st.spinner('Running simulation...'):
                        equity_df, final_portfolio = engine.run()

                    # --- F. 基准与指标 ---
                    bench_symbol = bench_options[selected_bench_label]
                    b_rets = helper.get_benchmark_returns(bench_symbol)
                    
                    if not b_rets.empty:
                        s_ts = pd.Timestamp(start_date)
                        e_ts = pd.Timestamp(end_date)
                        b_rets = b_rets.loc[s_ts:e_ts]
                        benchmark_equity = (1 + b_rets).cumprod() * bt_config['INITIAL_CAPITAL']
                        # 对齐数据
                        benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(bt_config['INITIAL_CAPITAL'])
                    else:
                        st.warning(f"Benchmark data missing for {bench_symbol}")
                        benchmark_equity = pd.Series(bt_config['INITIAL_CAPITAL'], index=equity_df.index)
                        
                    metrics = calculate_extended_metrics(equity_df['total_value'], benchmark_equity, final_portfolio)
                    
                    # 存入 Session State
                    st.session_state.bt_ready = True
                    st.session_state.equity_df = equity_df
                    st.session_state.metrics = metrics
                    st.session_state.strategy = strategy
                    st.session_state.final_portfolio = final_portfolio
                    st.session_state.engine = engine
                    st.session_state.selected_factors = selected_factors
                    st.session_state.bench_label = selected_bench_label
            
            except Exception as e:
                st.error(f"Runtime Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # --- 结果展示 (完全恢复) ---
    if st.session_state.get('bt_ready'):
        m = st.session_state.metrics
        
        # 1. 关键指标卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha (Excess)", f"{m.get('Alpha', 0):+.2%}")
        c2.metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
        c3.metric("Info Ratio", f"{m.get('Info Ratio', 0):.2f}")
        c4.metric("Beta", f"{m.get('Beta', 0):.2f}")

        # 2. 成本与回撤
        st.divider()
        st.subheader("Transaction Cost Attribution")
        ct1, ct2, ct3, ct4 = st.columns(4)
        ct1.metric("Total Cost", f"${m.get('Total Cost', 0):,.0f}")
        ct2.metric("Commission", f"${m.get('Commission', 0):,.0f}")
        ct3.metric("Slippage", f"${m.get('Slippage', 0):,.0f}")
        ct4.metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")

        # 3. Excel 下载
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            summary_df = pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value'])
            summary_df.to_excel(writer, sheet_name='Summary')
            ts_df = pd.DataFrame({'Strategy': m['strategy_curve'], 'Benchmark': m['benchmark_curve'], 'Excess': m['excess_curve']})
            ts_df.to_excel(writer, sheet_name='Comparison')
        st.download_button("Download Excel Report", buffer.getvalue(), f"Backtest_Report.xlsx", use_container_width=True)

        # 4. 净值曲线图 (Plotly)
        st.subheader(f"Strategy vs {st.session_state.bench_label}")
        fig = go.Figure()
        # 策略曲线
        fig.add_trace(go.Scatter(x=m['strategy_curve'].index, y=m['strategy_curve'], name='Strategy', line=dict(color='#0B3D59', width=2.5)))
        # 基准曲线
        fig.add_trace(go.Scatter(x=m['benchmark_curve'].index, y=m['benchmark_curve'], name=st.session_state.bench_label, line=dict(color='#5EA9CE', width=2, dash='dot')))
        # 超额收益 (阴影区)
        fig.add_trace(go.Scatter(x=m['excess_curve'].index, y=m['excess_curve'], name='Excess Return', yaxis='y2', fill='tozeroy', line=dict(color='#8E44AD', width=1.5), fillcolor='rgba(142, 68, 173, 0.2)'))
        
        fig.update_layout(
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title=dict(text="Normalized Value", font=dict(color="#0B3D59")), tickfont=dict(color="#0B3D59")),
            yaxis2=dict(title=dict(text="Cumulative Excess Return", font=dict(color="#8E44AD")), tickfont=dict(color="#8E44AD"), overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)

        # 5. 详细分析 Tab
        st.divider()
        nav_options = ["Performance", "Signals", "Holdings", "Factor Correlation", "Risk Analysis"]
        active_tab = st.radio("Analysis View", nav_options, horizontal=True, key="active_nav_tab")

        if active_tab == "Performance":
            st.table(pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value']).astype(str))
        
        elif active_tab == "Signals":
            # 注意：如果 BaseStrategy 里没写日志记录，这里可能是空的。
            # 通常我们在 Engine 里记录 trade_log，这里尝试获取
            st.info("Trade signals log (from Engine):")
            st.dataframe(st.session_state.strategy.get_trade_log() if hasattr(st.session_state.strategy, 'get_trade_log') else pd.DataFrame(), use_container_width=True)
            
        elif active_tab == "Holdings":
            st.dataframe(st.session_state.final_portfolio.get_holdings_history(), use_container_width=True)
            
        elif active_tab == "Factor Correlation":
            st.subheader("Dynamic Factor Correlation Analysis")
            # 从 strategy 对象里直接取刚才算好的 factor_data
            if hasattr(st.session_state.strategy, 'factor_data') and st.session_state.strategy.factor_data is not None:
                fd = st.session_state.strategy.factor_data
                
                # 时间滑块
                a_range = st.slider("Select Analysis Period", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM-DD", key="corr_slider")
                
                # 切片 (index level 0 is datetime)
                try:
                    fd_slice = fd.loc[str(a_range[0]):str(a_range[1])]
                    if not fd_slice.empty:
                        # factor_data 已经是宽表了 (columns=factor names)，直接 corr()
                        corr_m = fd_slice.corr()
                        st.plotly_chart(px.imshow(corr_m, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
                    else:
                        st.warning("No data in selected range.")
                except Exception as e:
                    st.error(f"Error filtering data: {e}")
            else:
                st.info("No factor data available.")
                
        elif active_tab == "Risk Analysis":
            st.subheader("Daily Risk Exposure (95% Confidence)")
            if 'rolling_var_series' in m:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=m['rolling_var_series'].index, y=m['rolling_var_series'].values * 100, fill='tozeroy', name='95% Rolling VaR', line=dict(color='rgba(255, 0, 0, 0.6)')))
                fig_r.update_layout(yaxis_title="Potential Loss (%)", template="plotly_white")
                st.plotly_chart(fig_r, use_container_width=True)
                st.markdown(f"**Metrics**: 95% Historical VaR: **{abs(m.get('VaR_95', 0)):.2%}**, 95% ES: **{abs(m.get('ES_95', 0)):.2%}**.")
    else:
        st.info("👈 Configure parameters on the left and click 'Run Backtest' to start.")