import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import os
import io
from datetime import datetime
import sys          # <--- 如果你要加刚才的“一键更新”功能，记得把这三个也加上
import subprocess   # <--- 
import time         # <---

# 核心库导入
from quant_core.data.query_helper import DataQueryHelper
# from quant_core.data_handler import DataHandler # ❌ 已删除
from quant_core.strategy import FactorTopNStrategy
from quant_core.backtest_engine import BacktestEngine
from quant_core.performance import calculate_extended_metrics

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
def get_analysis_runner(_query_helper):
    """Initialize with QueryHelper instead of DataHandler"""
    return FactorTaskRunner(_query_helper)

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
    
    # 加载配置和 Helper
    config = load_config('config.yaml')
    helper = get_query_helper() # 获取唯一的 QueryHelper 实例
    
    if app_mode == "Strategy Explorer":
        st.header("Parameters")
        # [修改] 映射显示名称到数据库中的 ETF 代码
        bench_options = {
            "S&P 500 (SPY)": "SPY", 
            "Global Equity (ACWI)": "ACWI", 
            "Global Bond (AGG)": "AGG", 
            "Commodities (GSG)": "GSG"
        }
        selected_bench_label = st.selectbox("Compare against Benchmark", list(bench_options.keys()))
        # 从 Helper 获取全量数据以提取 Columns 列表 (如果需要) 
        # 或者直接使用注册表中的因子列表
        # 这里为了简单，我们用硬编码或从引擎获取
        runner_temp = get_analysis_runner(helper)
        available_factors = sorted(list(runner_temp.factor_engine.FACTOR_REGISTRY.keys()))
        
        selected_factors = st.multiselect("Select Factors", available_factors, default=['momentum', 'rsi'])
        
        factor_weights = {f: st.number_input(f"Weight: {f}", 0.0, 1.0, 1.0/len(selected_factors), 0.05) for f in selected_factors} if selected_factors else {}
        st.divider()
        st.header("Costs & Execution")
        comm_rate = st.number_input("Commission Rate", 0.0, 0.01, 0.0010, format="%.4f")
        slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, format="%.4f")
        rebalance_days = st.slider("Rebalance Frequency", 1, 60, 20)
        col_s, col_e = st.columns(2)
        start_date = col_s.date_input("Start", datetime(2018, 1, 1))
        end_date = col_e.date_input("End", datetime(2024, 7, 31))
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
        
    # --- [新增] 侧边栏底部：隐蔽的数据同步功能 ---
    st.markdown("---")
    with st.expander("📡 Data Status", expanded=False):
        # 1. 显示当前数据日期
        try:
            # 获取 helper (如果上面没定义 helper，这里重新获取一下)
            h_temp = get_query_helper()
            mkt_summary = h_temp.get_market_summary()
            if not mkt_summary.empty:
                # 获取所有资产中最新的日期
                latest_date = mkt_summary['end'].max()
                st.caption(f"Data up to: **{latest_date.strftime('%Y-%m-%d')}**")
            else:
                st.caption("Data: Empty")
        except Exception:
            st.caption("Status: Unknown")

        # 2. 刷新按钮
        if st.button("🔄 Sync Now", use_container_width=True):
            status_box = st.empty()
            status_box.info("⏳ Connecting to IBKR...")
            
            try:
                # 调用子进程运行 run_data_sync.py
                result = subprocess.run(
                    [sys.executable, "run_data_sync.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    status_box.success("✅ Complete!")
                    # 关键：清除 Streamlit 缓存，否则它还会读取旧的 Parquet 文件
                    st.cache_resource.clear()
                    time.sleep(1)
                    st.rerun() # 刷新页面
                else:
                    status_box.error("❌ Failed")
                    with st.expander("Log"):
                        st.code(result.stderr)
            except Exception as e:
                status_box.error(f"Err: {str(e)}")

# --- Sidebar End ---
if app_mode == "Data Explorer": 
    render_data_explorer()
elif app_mode == "Analysis Explorer": 
    render_analysis_explorer(helper)
elif app_mode == "Strategy Explorer":
    st.title("Quantitative Strategy Explorer")
    
    # 1. 运行按钮逻辑
    if run_btn:
        if not selected_factors:
            st.error("Error: Please select at least one factor.")
        else:
            with st.spinner('Running simulation...'):
                try:
                    # --- A. 准备回测参数 ---
                    bt_config = {
                        'INITIAL_CAPITAL': 1000000, 
                        'COMMISSION_RATE': comm_rate, 
                        'SLIPPAGE': slip_rate, 
                        'REBALANCE_DAYS': rebalance_days
                    }
                    
                    # --- B. 初始化数据与策略 ---
                    # 使用 QueryHelper 获取 Universe (所有 distinct symbols)
                    u_df = helper.get_all_symbols()
                    
                    # 初始化策略
                    strategy = FactorTopNStrategy(universe_df=u_df, factor_weights=factor_weights, top_n=5)
                    
                    # 初始化引擎 (传入 helper)
                    engine = BacktestEngine(
                        start_date=start_date.strftime('%Y-%m-%d'), 
                        end_date=end_date.strftime('%Y-%m-%d'), 
                        config=bt_config, 
                        strategy=strategy, 
                        query_helper=helper # 关键修改：传入 query_helper
                    )
                    # 注入权重
                    engine.factor_engine.current_weights = factor_weights 
                    
                    # --- C. 执行回测 ---
                    equity_df, final_portfolio = engine.run()

                    # --- D. 处理基准数据 (Benchmark) ---
                    # [修改] 使用 helper 直接从数据库获取收益率，不再读取 CSV
                    bench_symbol = bench_options[selected_bench_label]
                    b_rets = helper.get_benchmark_returns(bench_symbol)
                    
                    if not b_rets.empty:
                        # 截取回测时间段
                        # 注意：series.loc 切片包含端点，确保索引是 datetime 类型
                        s_ts = pd.Timestamp(start_date)
                        e_ts = pd.Timestamp(end_date)
                        b_rets = b_rets.loc[s_ts:e_ts]
                        
                        # 计算净值曲线 (从初始资金开始复利)
                        benchmark_equity = (1 + b_rets).cumprod() * bt_config['INITIAL_CAPITAL']
                        
                        # [关键] 对齐索引：防止基准交易日与策略不一致（如美股休市与港股休市不同）
                        # 使用 reindex 将基准强制对齐到策略的日期轴，缺失值前向填充
                        benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill')
                        
                        # 如果起始日没有数据，填充为初始资金
                        benchmark_equity = benchmark_equity.fillna(bt_config['INITIAL_CAPITAL'])
                    else:
                        st.warning(f"⚠️ Benchmark data not found for {bench_symbol}. Using flat line.")
                        benchmark_equity = pd.Series(bt_config['INITIAL_CAPITAL'], index=equity_df.index)
                        
                    # --- E. 计算最终指标 ---
                    metrics = calculate_extended_metrics(equity_df['total_value'], benchmark_equity, final_portfolio)
                    
                    # --- F. 存入 Session State ---
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

    # 2. 结果渲染
    if st.session_state.get('bt_ready'):
        m = st.session_state.metrics
        
        # 指标卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha (Excess)", f"{m.get('Alpha', 0):+.2%}")
        c2.metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
        c3.metric("Info Ratio", f"{m.get('Info Ratio', 0):.2f}")
        c4.metric("Beta", f"{m.get('Beta', 0):.2f}")

        # 成本
        st.divider()
        st.subheader("Transaction Cost Attribution")
        ct1, ct2, ct3, ct4 = st.columns(4)
        ct1.metric("Total Cost", f"${m.get('Total Cost', 0):,.0f}")
        ct2.metric("Commission", f"${m.get('Commission', 0):,.0f}")
        ct3.metric("Slippage", f"${m.get('Slippage', 0):,.0f}")
        ct4.metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")

        # 下载
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            summary_df = pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value'])
            summary_df.to_excel(writer, sheet_name='Summary')
            ts_df = pd.DataFrame({'Strategy': m['strategy_curve'], 'Benchmark': m['benchmark_curve'], 'Excess': m['excess_curve']})
            ts_df.to_excel(writer, sheet_name='Comparison')
        st.download_button("Download Excel Report", buffer.getvalue(), f"Backtest_Report.xlsx", use_container_width=True)

        # 双轴图表
        st.subheader(f"Strategy vs {st.session_state.bench_label}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m['strategy_curve'].index, y=m['strategy_curve'], name='Strategy', line=dict(color='#0B3D59', width=2.5)))
        fig.add_trace(go.Scatter(x=m['benchmark_curve'].index, y=m['benchmark_curve'], name=st.session_state.bench_label, line=dict(color='#5EA9CE', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=m['excess_curve'].index, y=m['excess_curve'], name='Excess Return', yaxis='y2', fill='tozeroy', line=dict(color='#8E44AD', width=1.5), fillcolor='rgba(142, 68, 173, 0.2)'))
        
        fig.update_layout(
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title=dict(text="Normalized Value", font=dict(color="#0B3D59")), tickfont=dict(color="#0B3D59")),
            yaxis2=dict(title=dict(text="Cumulative Excess Return", font=dict(color="#8E44AD")), tickfont=dict(color="#8E44AD"), overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)

        # 底部 Tab
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
            else:
                st.info("Select at least 2 factors to see correlation matrix.")
        elif active_tab == "Risk Analysis":
            st.subheader("Daily Risk Exposure (95% Confidence)")
            if 'rolling_var_series' in m:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=m['rolling_var_series'].index, y=m['rolling_var_series'].values * 100, fill='tozeroy', name='95% Rolling VaR', line=dict(color='rgba(255, 0, 0, 0.6)')))
                fig_r.update_layout(yaxis_title="Potential Loss (%)", template="plotly_white")
                st.plotly_chart(fig_r, use_container_width=True)
                st.markdown(f"**Metrics**: 95% Historical VaR: **{abs(m.get('VaR_95', 0)):.2%}**, 95% ES: **{abs(m.get('ES_95', 0)):.2%}**.")
    else:
        st.info("Configure the parameters and click 'Run Backtest' to see results.")