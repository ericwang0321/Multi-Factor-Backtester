import streamlit as st
import streamlit.components.v1 as components

class TradingViewWidgets:
    """
    复刻 OpenStock 的视觉组件，使用 TradingView 原生 Embed 代码。
    """

    @staticmethod
    def render_ticker_tape():
        """顶部滚动行情条"""
        html_code = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
          {
          "symbols": [
            {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
            {"proName": "FOREXCOM:NSXUSD", "title": "Nasdaq 100"},
            {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
            {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
            {"proName": "BITSTAMP:ETHUSD", "title": "Ethereum"}
          ],
          "showSymbolLogo": true,
          "colorTheme": "light",
          "isTransparent": false,
          "displayMode": "adaptive",
          "locale": "en"
        }
          </script>
        </div>
        """
        components.html(html_code, height=75)

    @staticmethod
    def render_advanced_chart(symbol="NASDAQ:AAPL"):
        """高级 K 线图 (强制高度版)"""
        html_code = f"""
        <div class="tradingview-widget-container" style="height:100%;width:100%">
          <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
          {{
          "width": "100%",
          "height": "800",   
          "symbol": "{symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "calendar": false,
          "support_host": "https://www.tradingview.com"
        }}
          </script>
        </div>
        """
        # 注意：这里去掉了 "autosize": true，并显式添加了 "height": "800"
        
        # Streamlit 容器高度
        components.html(html_code, height=810)

    @staticmethod
    def render_market_heatmap():
        """股市热力图"""
        html_code = """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
          {
          "exchanges": [],
          "dataSource": "SPX500",
          "grouping": "sector",
          "blockSize": "market_cap_basic",
          "blockColor": "change",
          "locale": "en",
          "symbolUrl": "",
          "colorTheme": "light",
          "hasTopBar": false,
          "isTransparent": false,
          "width": "100%",
          "height": "500"
        }
          </script>
        </div>
        """
        components.html(html_code, height=500)

    @staticmethod
    def render_company_profile(symbol="NASDAQ:AAPL"):
        """公司简介"""
        html_code = f"""
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-profile.js" async>
          {{
          "width": "100%",
          "height": 500,
          "colorTheme": "light",
          "isTransparent": false,
          "symbol": "{symbol}",
          "locale": "en"
        }}
          </script>
        </div>
        """
        components.html(html_code, height=500)