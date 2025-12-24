import streamlit as st
import pandas as pd
import yaml
import io
import os
import json
from datetime import datetime
import sys
import subprocess
import time

# --- 核心库导入 ---
from quant_core.data.query_helper import DataQueryHelper
from quant_core.strategies.rules import LinearWeightedStrategy
from quant_core.backtest_engine import BacktestEngine
from quant_core.performance import calculate_extended_metrics
from quant_core.factors.engine import FactorEngine
from quant_core.analysis.task_runner import FactorTaskRunner
# [新增] 引入实盘需要的工厂
from quant_core.strategies import create_strategy_instance

# [核心修复] 引入项目的配置加载器 (支持 base+backtest 合并)
from config import load_config as _project_load_config

# --- 引入可视化模块 ---
from quant_core.visualization.market import MarketCharts
from quant_core.visualization.factor import FactorCharts
from quant_core.visualization.performance import PerformanceCharts
from quant_core.visualization.trading import TradingCharts     
from quant_core.visualization.reporting import ReportGenerator 

# --- Page Setup ---
st.set_page_config(page_title="Quant Cockpit & Backtester", layout="wide", page_icon="📈")

# --- Constants for Live Dashboard ---
LIVE_DATA_DIR = 'data/live'
STATE_FILE = os.path.join(LIVE_DATA_DIR, 'dashboard_state.json')
COMMAND_FILE = os.path.join(LIVE_DATA_DIR, 'command.json')
LOG_FILE = 'logs/live_strategy.log'

# --- Resource Caching ---

# [核心修复] 替换旧的 load_config，改用 Config 模块加载
@st.cache_data
def get_cached_config(mode='backtest'):
    """
    使用项目统一的配置加载逻辑 (base + override)，并进行缓存
    """
    return _project_load_config(mode=mode)

@st.cache_resource
def get_query_helper():
    return DataQueryHelper(storage_path='data/processed/all_price_data.parquet')

@st.cache_resource
def get_factor_engine(_query_helper):
    return FactorEngine(query_helper=_query_helper)

@st.cache_resource
def get_analysis_runner(_query_helper):
    return FactorTaskRunner(_query_helper)

# --- Helper Functions ---
def prepare_factor_data_for_strategy(_engine, codes, factors, start_date, end_date):
    """内存计算因子数据"""
    if _engine.xarray_data is None:
        _engine._get_xarray_data()
    data_dict = {}
    progress_text = "Computing factors in-memory..."
    my_bar = st.progress(0, text=progress_text)
    total = len(factors)
    for i, f_name in enumerate(factors):
        df = _engine._compute_and_cache_factor(f_name)
        if not df.empty:
            df_slice = df.loc[str(start_date):str(end_date)]
            valid_cols = [c for c in df_slice.columns if c in codes]
            if valid_cols:
                stacked = df_slice[valid_cols].stack()
                stacked.name = f_name
                data_dict[f_name] = stacked
        my_bar.progress((i + 1) / total, text=f"Computed {f_name}")
    my_bar.empty()
    if not data_dict: return pd.DataFrame()
    full_factor_df = pd.concat(data_dict.values(), axis=1)
    full_factor_df.index.names = ['datetime', 'sec_code']
    return full_factor_df

# --- Module 1: Live Dashboard ---
def render_live_dashboard():
    st.title("🔴 Live Trading Cockpit")
    
    # 1. 状态读取逻辑
    if not os.path.exists(STATE_FILE):
        st.warning("Waiting for live strategy to start... (dashboard_state.json not found)")
        st.info("💡 Hint: Run 'python run_live_strategy.py' in a separate terminal.")
        return

    try:
        # 使用 retry 机制防止读取时的竞争冲突
        state = {}
        for _ in range(3):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                break
            except json.JSONDecodeError:
                time.sleep(0.1)
                continue
    except Exception as e:
        st.error(f"Error reading state file: {e}")
        return

    # 2. 顶部 HUD (Heads-Up Display)
    acct = state.get('account', {})
    status = state.get('status', 'Unknown')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 状态指示灯
        color = "green" if status in ["Connected", "Monitoring"] else "red"
        st.markdown(f"### Status: :{color}[{status}]")
        st.caption(f"Last Update: {state.get('updated_at', '--')}")

    with col2:
        equity = acct.get('total_equity', 0)
        st.metric("Net Liquidation", f"${equity:,.2f}")

    with col3:
        pnl = acct.get('unrealized_pnl', 0)
        st.metric("Unrealized PnL", f"${pnl:,.2f}", delta=f"{pnl:,.2f}")

    with col4:
        # 控制台
        st.markdown("**Emergency Control**")
        c_stop, c_flat, c_cancel = st.columns(3)
        if c_stop.button("🛑 STOP", type="primary", use_container_width=True):
            with open(COMMAND_FILE, 'w') as f:
                json.dump({"action": "STOP"}, f)
            st.toast("🚨 STOP Command Sent!", icon="🛑")
            
        if c_flat.button("📉 FLAT", type="secondary", use_container_width=True):
            with open(COMMAND_FILE, 'w') as f:
                json.dump({"action": "FLAT_ALL"}, f)
            st.toast("📉 FLAT ALL Command Sent!", icon="📉")

        # [NEW] Cancel Button (English Version)
        if c_cancel.button("🚫 CANCEL", use_container_width=True, help="Cancel all open orders"):
            with open(COMMAND_FILE, 'w') as f:
                json.dump({"action": "CANCEL_ALL"}, f)
            st.toast("Command Sent: Cancel All Orders")
            
    st.divider()

    # 3. 持仓监控
    st.subheader("Current Positions")
    positions = acct.get('positions', {})
    avg_costs = acct.get('avg_costs', {})
    
    if positions:
        pos_data = []
        for sym, qty in positions.items():
            if qty != 0:
                cost = avg_costs.get(sym, 0)
                # 简单估算市值，更精准的需要 TWS 推送价格
                pos_data.append({
                    "Symbol": sym,
                    "Quantity": qty,
                    "Avg Cost": f"${cost:.2f}",
                })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No active positions (Flat).")

    # 4. 实时日志流
    st.divider()
    st.subheader("Live Logs (Tail 50)")
    
    log_content = ""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            # 只显示最后 50 行
            log_content = "".join(lines[-50:])
    else:
        log_content = "Log file not found."

    st.text_area("System Logs", log_content, height=300, disabled=True)


# --- Module 2: Data Explorer ---
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
            st.plotly_chart(MarketCharts.plot_price_history(df, selected_symbol), use_container_width=True)
            
            t1, t2 = st.tabs(["Volume Analysis", "Data Table"])
            with t1: 
                st.plotly_chart(MarketCharts.plot_volume(df), use_container_width=True)
            with t2: 
                st.dataframe(df.sort_values('datetime', ascending=False), use_container_width=True)

# --- Module 3: Analysis Explorer ---
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
            st.plotly_chart(FactorCharts.plot_ic_series(st.session_state.ana_ic), use_container_width=True)
            
            st.subheader("Quantile Analysis (Fixed Wealth Curves)")
            st.plotly_chart(FactorCharts.plot_quantile_layers(st.session_state.ana_groups), use_container_width=True)
        else:
            st.warning("No valid statistics generated. Check data quality.")

# --- Sidebar & Main Logic ---
with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio("Choose Module", [
        "🔴 Live Dashboard", 
        "Strategy Explorer", 
        "Data Explorer", 
        "Analysis Explorer"
    ])
    
    # [核心修复] 使用新的配置加载逻辑 (mode='backtest' 提供给策略浏览器默认值)
    config = get_cached_config(mode='backtest')
    helper = get_query_helper()
    
    # Live 模式自动刷新
    if app_mode == "🔴 Live Dashboard":
        st.divider()
        st.markdown("### ⏱️ Dashboard Control")
        auto_refresh = st.toggle("Auto-Refresh (3s)", value=False)
        if st.button("Manual Refresh"):
            st.rerun()

    if app_mode == "Strategy Explorer":
        st.header("Parameters")
        bench_options = {"S&P 500 (SPY)": "SPY", "Global Equity (ACWI)": "ACWI", "Global Bond (AGG)": "AGG", "Commodities (GSG)": "GSG"}
        selected_bench_label = st.selectbox("Compare against Benchmark", list(bench_options.keys()))
        
        runner_temp = get_analysis_runner(helper)
        available_factors = sorted(list(runner_temp.factor_engine.FACTOR_REGISTRY.keys()))
        selected_factors = st.multiselect("Select Factors", available_factors, default=['momentum', 'rsi'])
        factor_weights = {f: st.number_input(f"Weight: {f}", 0.0, 1.0, 1.0/len(selected_factors), 0.05) for f in selected_factors} if selected_factors else {}
        
        st.divider()
        st.header("🛡️ Risk Management")
        enable_stop_loss = st.checkbox("Enable Stop Loss", value=False)
        stop_loss_pct = (st.slider("Stop Loss %", 1, 30, 10) / 100.0) if enable_stop_loss else None
        
        enable_pos_limit = st.checkbox("Enable Position Limit", value=True)
        max_pos_weight = (st.slider("Max Weight %", 10, 100, 30) / 100.0) if enable_pos_limit else None
        
        enable_circuit_breaker = st.checkbox("Enable Circuit Breaker", value=False)
        max_drawdown_pct = (st.slider("Max Drawdown %", 5, 50, 20) / 100.0) if enable_circuit_breaker else None
        
        st.divider()
        st.header("Costs & Execution")
        top_k = st.slider("Top K Stocks", 1, 20, 5)
        
        # 使用 config 中的默认值 (如果 config 中没有，则用默认值)
        default_comm = config.get('COMMISSION_RATE', 0.0010)
        default_slip = config.get('SLIPPAGE', 0.0005)
        
        comm_rate = st.number_input("Commission Rate", 0.0, 0.01, default_comm, format="%.4f")
        slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, default_slip, format="%.4f")
        rebalance_days = st.slider("Rebalance Frequency", 1, 60, 20)
        col_s, col_r = st.columns(2)
        start_date = col_s.date_input("Start", datetime(2018, 1, 1))
        end_date = col_r.date_input("End", datetime(2024, 7, 31))
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
        
    if app_mode == "Data Explorer":
        st.markdown("---")
        with st.expander("📡 Data Status"):
            if st.button("🔄 Sync Now", use_container_width=True):
                subprocess.run([sys.executable, "run_data_sync.py"])
                st.cache_resource.clear()
                st.rerun()

# --- Main Routing ---
if app_mode == "🔴 Live Dashboard":
    render_live_dashboard()
    # 自动刷新逻辑
    if auto_refresh:
        time.sleep(3)
        st.rerun()

elif app_mode == "Data Explorer": 
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
                bt_config = {'INITIAL_CAPITAL': 1000000, 'COMMISSION_RATE': comm_rate, 'SLIPPAGE': slip_rate, 'REBALANCE_DAYS': rebalance_days}
                u_df = helper.get_all_symbols()
                universe_codes = u_df['sec_code'].tolist()
                
                f_engine = get_factor_engine(helper)
                factor_data = prepare_factor_data_for_strategy(f_engine, universe_codes, selected_factors, start_date, end_date)
                
                if factor_data.empty:
                    st.error("No factor data generated.")
                else:
                    # 使用工厂模式或原有逻辑
                    strategy = LinearWeightedStrategy(name="App_Strat", weights=factor_weights, top_k=top_k, 
                                                      stop_loss_pct=stop_loss_pct, max_pos_weight=max_pos_weight, max_drawdown_pct=max_drawdown_pct)
                    strategy.load_data(factor_data)
                    
                    engine = BacktestEngine(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'), 
                                            config=bt_config, strategy=strategy, query_helper=helper)
                    
                    with st.spinner('Running simulation...'):
                        equity_df, final_portfolio = engine.run()

                    bench_symbol = bench_options[selected_bench_label]
                    b_rets = helper.get_benchmark_returns(bench_symbol)
                    if not b_rets.empty:
                        b_rets = b_rets.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
                        benchmark_equity = (1 + b_rets).cumprod() * bt_config['INITIAL_CAPITAL']
                        benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(bt_config['INITIAL_CAPITAL'])
                    else:
                        benchmark_equity = pd.Series(bt_config['INITIAL_CAPITAL'], index=equity_df.index)
                        
                    metrics = calculate_extended_metrics(equity_df['total_value'], benchmark_equity, final_portfolio)
                    
                    st.session_state.bt_ready = True
                    st.session_state.metrics = metrics
                    st.session_state.equity_df = equity_df
                    st.session_state.strategy = strategy
                    st.session_state.final_portfolio = final_portfolio
                    st.session_state.engine = engine
                    st.session_state.bench_label = selected_bench_label
            
            except Exception as e:
                st.error(f"Runtime Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.get('bt_ready'):
        m = st.session_state.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alpha", f"{m.get('Alpha', 0):+.2%}")
        c2.metric("Sharpe", f"{m.get('Sharpe Ratio', 0):.2f}")
        c3.metric("Info Ratio", f"{m.get('Info Ratio', 0):.2f}")
        c4.metric("Beta", f"{m.get('Beta', 0):.2f}")

        st.divider()
        st.subheader("Transaction Cost Attribution")
        ct1, ct2, ct3, ct4 = st.columns(4)
        ct1.metric("Total Cost", f"${m.get('Total Cost', 0):,.0f}")
        ct2.metric("Commission", f"${m.get('Commission', 0):,.0f}")
        ct3.metric("Slippage", f"${m.get('Slippage', 0):,.0f}")
        ct4.metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")

        # --- 报告生成模块 ---
        st.divider()
        st.subheader("📊 Report & Export")
        
        reporter = ReportGenerator(strategy_name="Linear_Weighted_Strategy")
        
        col_down1, col_down2 = st.columns(2)
        
        trade_log = st.session_state.final_portfolio.get_trade_log()
        holdings = st.session_state.final_portfolio.get_holdings_history()
        
        excel_data = reporter.generate_excel_report(m, st.session_state.equity_df, holdings, trade_log)
        col_down1.download_button("📥 Download Excel Report", excel_data, f"Backtest_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        html_report = reporter.generate_html_report(
            metrics=m,
            equity_curve=m['strategy_curve'],
            benchmark_curve=m['benchmark_curve'],
            drawdown_series=m.get('drawdown_series'), 
            trade_log=trade_log
        )
        col_down2.download_button("🌏 Download HTML Report", html_report, f"Report_{datetime.now().strftime('%Y%m%d')}.html", "text/html", use_container_width=True)
        
        st.divider()
        st.subheader(f"Strategy vs {st.session_state.bench_label}")
        st.plotly_chart(PerformanceCharts.plot_equity_curve(m['strategy_curve'], m['benchmark_curve'], st.session_state.bench_label), use_container_width=True)

        st.divider()
        active_tab = st.radio("Analysis View", ["Performance", "Signals", "Trade Inspection", "Holdings", "Factor Correlation", "Risk Analysis"], horizontal=True)

        if active_tab == "Performance":
            st.table(pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value']).astype(str))
        
        elif active_tab == "Signals":
            if not trade_log.empty:
                st.dataframe(trade_log, use_container_width=True)
            else:
                st.info("No trades executed.")
            
        elif active_tab == "Trade Inspection":
            st.subheader("🔍 Individual Trade Inspection")
            if hasattr(st.session_state.strategy, 'factor_data') and st.session_state.strategy.factor_data is not None:
                available_assets = st.session_state.strategy.factor_data.index.get_level_values('sec_code').unique()
                inspected_symbol = st.selectbox("Select Asset to Inspect", sorted(available_assets))
                
                if inspected_symbol:
                    insp_price = helper.get_history(inspected_symbol).loc[str(start_date):str(end_date)]
                    
                    if not trade_log.empty and 'sec_code' in trade_log.columns:
                         insp_signals = trade_log[trade_log['sec_code'] == inspected_symbol]
                    else:
                         insp_signals = pd.DataFrame()
                    
                    if not insp_price.empty:
                        fig_trade = TradingCharts.plot_strategy_view(
                            df_price=insp_price,
                            df_signals=insp_signals
                        )
                        st.plotly_chart(fig_trade, use_container_width=True)
                    else:
                        st.warning("No price data for this symbol.")
            else:
                st.info("Run backtest to inspect trades.")

        elif active_tab == "Holdings":
            st.dataframe(holdings, use_container_width=True)
        elif active_tab == "Factor Correlation":
            st.subheader("Dynamic Factor Correlation")
            if hasattr(st.session_state.strategy, 'factor_data') and st.session_state.strategy.factor_data is not None:
                fd = st.session_state.strategy.factor_data
                a_range = st.slider("Period", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM-DD")
                try:
                    fd_slice = fd.loc[str(a_range[0]):str(a_range[1])]
                    if not fd_slice.empty:
                        corr_m = fd_slice.corr()
                        st.plotly_chart(FactorCharts.plot_correlation_matrix(corr_m), use_container_width=True)
                except Exception: st.warning("No data")
        elif active_tab == "Risk Analysis":
            st.subheader("Risk Analysis")
            if 'rolling_var_series' in m:
                st.plotly_chart(PerformanceCharts.plot_rolling_var(m['rolling_var_series']), use_container_width=True)
                st.markdown(f"**Metrics**: 95% Historical VaR: **{abs(m.get('VaR_95', 0)):.2%}**, 95% ES: **{abs(m.get('ES_95', 0)):.2%}**.")