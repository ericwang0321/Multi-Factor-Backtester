import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml
import os
import io
from datetime import datetime

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
    # 根据用户目录结构加载数据
    dh = DataHandler(csv_path=config['paths']['price_data_csv'], start_date="2016-01-01", end_date="2025-12-31")
    dh.load_data()
    return dh, config

# --- Sidebar: Parameters ---
with st.sidebar:
    st.header("Parameters")
    dh, config = load_framework('config.yaml')
    
    # Benchmark Selection
    bench_options = {
        "S&P 500 (SPXT)": "spxt_index_daily_return.csv",
        "Global Equity (MXWD)": "mxwd_index_daily_return.csv",
        "Commodity (BCOM)": "bcom_index_daily_return.csv",
        "Global Bond": "global_bond_index_daily_return.csv"
    }
    selected_bench_label = st.selectbox("Compare against Benchmark", list(bench_options.keys()))
    
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

# --- Backtest Logic & Session Persistence ---
if run_btn:
    if not selected_factors:
        st.error("Error: Please select at least one factor.")
    else:
        with st.spinner('Running multi-factor simulation...'):
            bt_config = {
                'INITIAL_CAPITAL': 1000000, 
                'COMMISSION_RATE': comm_rate, 
                'SLIPPAGE': slip_rate, 
                'REBALANCE_DAYS': rebalance_days
            }
            
            # 1. 执行策略回测
            u_df = dh.load_universe_data(config['paths']['universe_definition'])
            strategy = FactorTopNStrategy(universe_df=u_df, factor_weights=factor_weights, top_n=5)
            engine = BacktestEngine(
                start_date=start_date.strftime('%Y-%m-%d'), 
                end_date=end_date.strftime('%Y-%m-%d'), 
                config=bt_config, strategy=strategy, data_handler=dh
            )
            engine.factor_engine.current_weights = factor_weights 
            equity_df, final_portfolio = engine.run()

            # 2. 加载基准数据并计算净值
            bench_file = bench_options[selected_bench_label]
            bench_path = os.path.join("data", "processed", bench_file)
            b_raw = pd.read_csv(bench_path)
            b_raw['report_date'] = pd.to_datetime(b_raw['report_date'])
            b_raw = b_raw.set_index('report_date').sort_index()
            
            b_rets = b_raw.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d'), 'default']
            benchmark_equity = (1 + b_rets).cumprod() * bt_config['INITIAL_CAPITAL']

            # 3. 计算所有指标
            metrics = calculate_extended_metrics(equity_df['total_value'], benchmark_equity, final_portfolio)
            
            # 4. 存入 Session State 防止刷新丢失
            st.session_state.bt_ready = True
            st.session_state.equity_df = equity_df
            st.session_state.metrics = metrics
            st.session_state.strategy = strategy
            st.session_state.final_portfolio = final_portfolio
            st.session_state.engine = engine
            st.session_state.selected_factors = selected_factors
            st.session_state.bench_label = selected_bench_label

# --- Rendering Results ---
if st.session_state.get('bt_ready'):
    m = st.session_state.metrics
    
    # 1. Row: High-level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha (Excess)", f"{m.get('Alpha', 0):+.2%}")
    c2.metric("Sharpe Ratio", f"{m.get('Sharpe Ratio', 0):.2f}")
    c3.metric("Info Ratio", f"{m.get('Info Ratio', 0):.2f}")
    c4.metric("Beta", f"{m.get('Beta', 0):.2f}")

    # 2. Row: Transaction Cost Attribution ($)
    st.divider()
    st.subheader("Transaction Cost Attribution")
    ct1, ct2, ct3, ct4 = st.columns(4)
    ct1.metric("Total Cost", f"${m.get('Total Cost', 0):,.0f}")
    ct2.metric("Commission", f"${m.get('Commission', 0):,.0f}")
    ct3.metric("Slippage", f"${m.get('Slippage', 0):,.0f}")
    ct4.metric("Max Drawdown", f"{m.get('Max Drawdown', 0):.2%}")

    # 3. Excel Download Button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        summary_df = pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value'])
        summary_df.to_excel(writer, sheet_name='Summary')
        pd.DataFrame({
            'Strategy': m['strategy_curve'], 
            'Benchmark': m['benchmark_curve'], 
            'Excess': m['excess_curve']
        }).to_excel(writer, sheet_name='Comparison')
    
    st.download_button("Download Excel Report", buffer.getvalue(), f"Backtest_{datetime.now().strftime('%H%M')}.xlsx", use_container_width=True)

    # 4. Strategy vs Benchmark Dual-Axis Chart
    st.subheader(f"Strategy vs {st.session_state.bench_label}")
    fig = go.Figure()
    
    # 策略线 (左轴 Y1): 深蓝色实线
    fig.add_trace(go.Scatter(
        x=m['strategy_curve'].index, y=m['strategy_curve'], 
        name='Strategy (Left Axis)', 
        line=dict(color='#0B3D59', width=2.5)
    ))
    
    # 基准线 (左轴 Y1): 浅蓝色点线
    fig.add_trace(go.Scatter(
        x=m['benchmark_curve'].index, y=m['benchmark_curve'], 
        name=f'{st.session_state.bench_label} (Left Axis)', 
        line=dict(color='#5EA9CE', width=2, dash='dot')
    ))
    
    # 超额收益 (右轴 Y2): 紫色线并填充
    fig.add_trace(go.Scatter(
        x=m['excess_curve'].index, y=m['excess_curve'], 
        name='Excess Return (Right Axis)', 
        yaxis='y2', 
        fill='tozeroy', 
        line=dict(color='#8E44AD', width=1.5),
        fillcolor='rgba(142, 68, 173, 0.2)' 
    ))
    
    # 更新Layout（已修复 titlefont 错误）
    fig.update_layout(
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # 左侧Y轴
        yaxis=dict(
            title=dict(text="Normalized Value (Base 1.0)", font=dict(color="#0B3D59")),
            tickfont=dict(color="#0B3D59")
        ),
        # 右侧Y轴
        yaxis2=dict(
            title=dict(text="Cumulative Excess Return", font=dict(color="#8E44AD")),
            tickfont=dict(color="#8E44AD"),
            overlaying="y", 
            side="right"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5. Detail Tabs
    t1, t2, t3, t4, t5 = st.tabs(["Performance", "Signals", "Holdings", "Factor Correlation", "Risk Analysis"])
    
    with t1:
        st.table(pd.DataFrame.from_dict({k: v for k, v in m.items() if not isinstance(v, pd.Series)}, orient='index', columns=['Value']).astype(str))
    with t2:
        st.dataframe(st.session_state.strategy.get_trade_log(), use_container_width=True)
    with t3:
        st.dataframe(st.session_state.final_portfolio.get_holdings_history(), use_container_width=True)
    
    with t4:
        st.subheader("Dynamic Factor Correlation Analysis")
        current_factors = st.session_state.get('selected_factors', [])
        if len(current_factors) > 1:
            a_range = st.slider("Select Analysis Period", min_value=start_date, max_value=end_date, value=(start_date, end_date), format="YYYY-MM-DD")
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
            st.warning("Please select at least two factors.")

    with t5:
        st.subheader("Daily Risk Exposure (95% Confidence)")
        if 'rolling_var_series' in m:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=m['rolling_var_series'].index, y=m['rolling_var_series'].values * 100, 
                fill='tozeroy', name='95% Rolling VaR', line=dict(color='rgba(255, 0, 0, 0.6)')
            ))
            fig_r.update_layout(yaxis_title="Potential Loss (%)", template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True)
            st.markdown(f"**Metrics**: 95% Historical VaR is **{abs(m.get('VaR_95', 0)):.2%}**, 95% ES is **{abs(m.get('ES_95', 0)):.2%}**.")

else:
    st.info("Configure the parameters and click 'Run Backtest' to see results.")