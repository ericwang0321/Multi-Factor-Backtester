import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  
import yaml
import os
import io  
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
    
    available_factors = [
        'trend_score', 'momentum', 'volatility', 'turnover_mean', 
        'alpha001', 'breakout_quality_score', 'rsi', 'stochastic_k',
        'amount_mean', 'amihud_illiquidity'
    ]
    selected_factors = st.multiselect("Select Factors", available_factors, default=['trend_score', 'momentum'])
    
    factor_weights = {}
    if selected_factors:
        st.write("Set Factor Weights:")
        for factor in selected_factors:
            weight = st.number_input(f"Weight: {factor}", 0.0, 1.0, 1.0/len(selected_factors), 0.05)
            factor_weights[factor] = weight

    st.divider()
    st.header("Transaction Costs")
    comm_rate = st.number_input("Commission Rate", 0.0, 0.01, 0.001, format="%.4f")
    slip_rate = st.number_input("Slippage Rate", 0.0, 0.01, 0.0005, format="%.4f")

    rebalance_days = st.slider("Rebalance Frequency (Trading Days)", 1, 60, 20)
    top_n = st.number_input("Top N Selection", 1, 15, 5)
    
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
            'COMMISSION_RATE': comm_rate, 
            'SLIPPAGE': slip_rate,       
            'REBALANCE_DAYS': rebalance_days
        }

        with st.spinner('Running multi-factor simulation...'):
            u_df = dh.load_universe_data(config['paths']['universe_definition'])
            strategy = FactorTopNStrategy(universe_df=u_df, factor_weights=factor_weights, top_n=top_n, ascending=False)
            engine = BacktestEngine(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),
                                   config=bt_config, strategy=strategy, data_handler=dh)
            
            engine.factor_engine.current_weights = factor_weights 
            equity_df, final_portfolio = engine.run()
            
            metrics = calculate_extended_metrics(equity_df['total_value'], equity_df['total_value'], final_portfolio)

        # --- Dashboard Metrics ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Return", f"{metrics.get('总回报率', 0):.2%}")
        m_col2.metric("Annual Return", f"{metrics.get('年化回报率', 0):.2%}")
        m_col3.metric("Sharpe", f"{metrics.get('夏普比率', 0):.2f}")
        m_col4.metric("Max Drawdown", f"{metrics.get('最大回撤', 0):.2%}")

        st.divider()
        st.subheader("Transaction Cost Attribution")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cost", f"${metrics.get('总交易成本', 0):,.0f}")
        c2.metric("Commission", f"${metrics.get('累计佣金支出', 0):,.0f}")
        c3.metric("Slippage", f"${metrics.get('累计滑点支出', 0):,.0f}")
        c4.metric("Return Drag", f"-{metrics.get('交易成本对收益损耗', 0):.2%}", delta_color="inverse")

        # --- Export Excel (最小改动：过滤 Series) ---
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # 仅写入数值型指标
            clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, pd.Series)}
            pd.DataFrame.from_dict(clean_metrics, orient='index', columns=['Value']).to_excel(writer, sheet_name='Summary')
            strategy.get_trade_log().to_excel(writer, sheet_name='Trades', index=False)
            final_portfolio.get_holdings_history().to_excel(writer, sheet_name='Holdings', index=False)
        
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), 
                          file_name=f"Backtest_{datetime.now().strftime('%Y%m%d')}.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        # --- Main Chart ---
        st.divider()
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['total_value'], name='Equity Curve', line=dict(color='#00d1b2', width=2)))
        fig.update_layout(hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # --- Tabs (新增 Risk Analysis) ---
        t1, t2, t3, t4, t5 = st.tabs(["Performance", "Signals", "Holdings", "Factor Correlation", "Risk Analysis"])
        
        with t1:
            # 过滤掉绘图用的 Series 后显示表格
            st.table(pd.DataFrame.from_dict({k: v for k, v in metrics.items() if not isinstance(v, pd.Series)}, 
                                           orient='index', columns=['Value']).astype(str))
        with t2:
            st.dataframe(strategy.get_trade_log(), use_container_width=True)
        with t3:
            st.dataframe(final_portfolio.get_holdings_history(), use_container_width=True)
        with t4:
            st.subheader("Factor Cross-Correlation")
            factor_list = []
            for f in selected_factors:
                if f in engine.factor_engine._factor_cache:
                    s = engine.factor_engine._factor_cache[f].loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')].stack()
                    s.name = f
                    factor_list.append(s)
            if len(factor_list) > 1:
                st.plotly_chart(px.imshow(pd.concat(factor_list, axis=1).corr(), text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)

        with t5:
            st.subheader("🛡️ Daily Risk Exposure (95% Confidence)")
            if 'rolling_var_series' in metrics:
                r_var = metrics['rolling_var_series']
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=r_var.index, y=r_var.values * 100, fill='tozeroy', 
                                         name='95% Rolling VaR', line=dict(color='rgba(255, 0, 0, 0.7)')))
                fig_v.update_layout(yaxis_title="Potential Loss (%)", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_v, use_container_width=True)
                st.markdown(f"> **指标解读**: 历史 VaR(95%) 为 **{abs(metrics['历史 VaR (95%)']):.2%}**，预期缺口 ES(95%) 为 **{abs(metrics['预期缺口 ES (95%)']):.2%}**。")

else:
    st.info("Configure and Run Backtest.")