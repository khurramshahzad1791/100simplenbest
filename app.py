"""
MEXC ULTIMATE MULTI-TIMEFRAME SCANNER
- Selects 100 coins across categories: top volume, high volume up, high volume down, near breakout
- Shows signals as they are found, sorted by confidence
- Grades: A1 (≥80), A (60‑79), B (40‑59)
- Full position planning with account balance & leverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')
import ta

st.set_page_config(page_title="MEXC ULTIMATE SCANNER", layout="wide")
st.markdown("""
<style>
    .main-title { font-family: 'Orbitron', sans-serif; text-align: center; font-size: 48px;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; padding: 20px; }
    .signal-card { background: #1e1e1e; padding: 20px; border-radius: 15px; border-left: 5px solid;
        margin: 10px 0; color: white; transition: transform 0.3s; }
    .signal-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.3); }
    .long-card { border-left-color: #00ff00; }
    .short-card { border-left-color: #ff4444; }
    .metric-box { background: #2d2d2d; padding: 10px; border-radius: 8px; text-align: center; }
    .grade-a1 { background: gold; color: black; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
    .grade-a { background: silver; color: black; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
    .grade-b { background: #cd7f32; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-title">📊 MEXC ULTIMATE MULTI-TF SCANNER</h1>', unsafe_allow_html=True)

# ==================== DATA FETCHER ====================
class MEXCDataFetcher:
    def __init__(self):
        self.exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
        self.timeframes = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'}

    def fetch_tickers(self):
        """Get 24h volume and percentage change for all symbols"""
        try:
            tickers = self.exchange.fetch_tickers()
            data = []
            for sym, t in tickers.items():
                if '/USDT' in sym and t['quoteVolume'] and t['percentage'] is not None:
                    data.append({
                        'symbol': sym,
                        'volume': t['quoteVolume'],
                        'change': t['percentage']
                    })
            return pd.DataFrame(data).sort_values('volume', ascending=False)
        except Exception as e:
            st.error(f"Ticker fetch failed: {e}")
            return pd.DataFrame()

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        try:
            tf = self.timeframes.get(timeframe, '1h')
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except:
            return None

    def get_multi_timeframe_data(self, symbol, timeframes):
        data = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit=200)
            if df is not None and len(df) >= 50:
                data[tf] = df
        return data

# ==================== INDICATORS ====================
class TechnicalIndicators:
    @staticmethod
    def calculate_all(df):
        if df is None or len(df) < 50:
            return df
        for p in [9,20,50,100,200]:
            df[f'ma_{p}'] = df['c'].rolling(p).mean()
        df['rsi'] = ta.momentum.rsi(df['c'], 14)
        macd = ta.trend.MACD(df['c'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        df['volume_sma'] = df['v'].rolling(20).mean()
        df['volume_ratio'] = df['v'] / df['volume_sma']
        df['atr'] = ta.volatility.average_true_range(df['h'], df['l'], df['c'], 14)
        return df

# ==================== SUPPORT/RESISTANCE ====================
class SupportResistanceDetector:
    @staticmethod
    def find_swing_points(df, window=5):
        highs, lows = df['h'].values, df['l'].values
        swing_highs, swing_lows = [], []
        for i in range(window, len(df)-window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append((i, lows[i]))
        return swing_highs, swing_lows

    @staticmethod
    def detect_levels(df, tolerance=0.01):
        swing_highs, swing_lows = SupportResistanceDetector.find_swing_points(df)
        high_prices = [p[1] for p in swing_highs]
        low_prices = [p[1] for p in swing_lows]
        def cluster(prices, tol):
            if not prices: return []
            prices = sorted(prices)
            clusters, cur = [], [prices[0]]
            for p in prices[1:]:
                if abs(p - np.mean(cur)) / np.mean(cur) < tol:
                    cur.append(p)
                else:
                    clusters.append(np.mean(cur))
                    cur = [p]
            clusters.append(np.mean(cur))
            return clusters
        resistance = cluster(high_prices, tolerance)
        support = cluster(low_prices, tolerance)
        current = df['c'].iloc[-1]
        nearest_res = min([r for r in resistance if r > current], default=None)
        nearest_sup = max([s for s in support if s < current], default=None)
        return {'resistance': resistance, 'support': support,
                'nearest_resistance': nearest_res, 'nearest_support': nearest_sup}

# ==================== TREND DETECTOR ====================
class TrendlineDetector:
    @staticmethod
    def detect_trend(df):
        if len(df) < 20: return "neutral"
        y = df['c'].values[-20:]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.001: return "uptrend"
        if slope < -0.001: return "downtrend"
        return "sideways"

# ==================== POSITION SIZER ====================
class PositionSizer:
    @staticmethod
    def calculate(account, risk_pct, entry, stop, leverage=1.0):
        risk_amount = account * (risk_pct / 100)
        stop_dist = abs(entry - stop)
        if stop_dist == 0: return {}
        size = risk_amount / stop_dist
        value = size * entry
        return {'position_size': size, 'position_value': value,
                'required_margin': value / leverage, 'risk_amount': risk_amount}

# ==================== MULTI-TIMEFRAME ANALYZER ====================
class MultiTimeframeAnalyzer:
    def __init__(self, timeframes=['1M','1w','1d','4h','1h','15m']):
        self.timeframes = timeframes

    def analyze(self, symbol, data):
        if not data: return None
        # Determine higher‑timeframe bias
        bias = "neutral"
        for tf in ['1M','1w','1d','4h','1h']:
            if tf in data and data[tf] is not None and len(data[tf]) > 20:
                trend = TrendlineDetector.detect_trend(data[tf])
                if trend != "neutral":
                    bias = trend
                    break
        # Choose entry timeframe
        entry_tf = '15m' if '15m' in data else ('1h' if '1h' in data else None)
        if not entry_tf or entry_tf not in data:
            return None
        df = data[entry_tf]
        if len(df) < 20:
            return None
        df = TechnicalIndicators.calculate_all(df)
        cur = df.iloc[-1]

        ma_align = (cur['ma_9'] > cur['ma_20'] > cur['ma_50']) if bias=="uptrend" else (cur['ma_9'] < cur['ma_20'] < cur['ma_50']) if bias=="downtrend" else True
        rsi_ok = 30 < cur['rsi'] < 70
        vol_ok = cur['volume_ratio'] > 1.2
        macd_bull = cur['macd'] > cur['macd_signal'] and cur['macd_hist'] > 0
        macd_bear = cur['macd'] < cur['macd_signal'] and cur['macd_hist'] < 0

        sr = SupportResistanceDetector.detect_levels(df)
        near_support = sr['nearest_support'] and cur['c'] <= sr['nearest_support'] * 1.02
        near_resistance = sr['nearest_resistance'] and cur['c'] >= sr['nearest_resistance'] * 0.98

        signal, conf, reasons = "neutral", 0, []
        if bias == "uptrend" and near_support and ma_align and rsi_ok and vol_ok and macd_bull:
            signal, conf, reasons = "long", 80, ["Higher TF uptrend","Near support","MA aligned","RSI healthy","Volume spike","MACD bullish"]
        elif bias == "downtrend" and near_resistance and ma_align and rsi_ok and vol_ok and macd_bear:
            signal, conf, reasons = "short", 80, ["Higher TF downtrend","Near resistance","MA aligned","RSI healthy","Volume spike","MACD bearish"]
        elif bias == "uptrend" and near_support:
            signal, conf, reasons = "long", 60, ["Higher TF uptrend","Near support"]
        elif bias == "downtrend" and near_resistance:
            signal, conf, reasons = "short", 60, ["Higher TF downtrend","Near resistance"]

        if conf >= 80: grade = "A1"
        elif conf >= 60: grade = "A"
        elif conf >= 40: grade = "B"
        else: grade = "C"

        return {
            'symbol': symbol.replace('/USDT',''),
            'price': cur['c'],
            'signal': signal,
            'confidence': conf,
            'grade': grade,
            'reasons': reasons,
            'bias': bias,
            'entry_tf': entry_tf,
            'near_support': sr['nearest_support'],
            'near_resistance': sr['nearest_resistance'],
            'atr': cur['atr'],
            'rsi': cur['rsi'],
            'volume_ratio': cur['volume_ratio'],
            'trendline': TrendlineDetector.detect_trend(df)
        }

# ==================== SCANNER ENGINE (with diverse selection) ====================
class Scanner:
    def __init__(self, fetcher, analyzer):
        self.fetcher = fetcher
        self.analyzer = analyzer

    def select_diverse_symbols(self, ticker_df, total=100):
        """
        Selects:
        - top 30 by volume (famous)
        - next 30 with high volume & positive change (up movers)
        - next 20 with high volume & negative change (down movers)
        - remaining 20 (breakout candidates) from highest volume among leftover
        """
        if ticker_df.empty:
            return []
        df = ticker_df.copy()
        # Ensure we have enough coins
        if len(df) < total:
            total = len(df)

        # Famous (top 30 by volume)
        famous = df.head(30)['symbol'].tolist()
        used = set(famous)
        remaining = df[~df['symbol'].isin(used)]

        # Up movers: volume > median of remaining & change > 2%
        if not remaining.empty:
            median_vol = remaining['volume'].median()
            up = remaining[(remaining['volume'] >= median_vol) & (remaining['change'] > 2)]
            up_selected = up.head(30)['symbol'].tolist()
        else:
            up_selected = []
        used.update(up_selected)
        remaining = df[~df['symbol'].isin(used)]

        # Down movers: volume > median of remaining & change < -2%
        if not remaining.empty:
            median_vol = remaining['volume'].median()
            down = remaining[(remaining['volume'] >= median_vol) & (remaining['change'] < -2)]
            down_selected = down.head(20)['symbol'].tolist()
        else:
            down_selected = []
        used.update(down_selected)

        # Fill the rest (up to total) with highest volume among what's left
        remaining = df[~df['symbol'].isin(used)]
        needed = total - len(used)
        if needed > 0 and not remaining.empty:
            others = remaining.head(needed)['symbol'].tolist()
        else:
            others = []

        # Combine in desired order
        final = famous + up_selected + down_selected + others
        return final[:total]

    def scan(self, ticker_df, filters, account, risk_pct, leverage, timeframes):
        symbols_to_scan = self.select_diverse_symbols(ticker_df, total=100)
        if not symbols_to_scan:
            st.warning("No symbols selected. Check ticker data.")
            return []

        results = []
        total = len(symbols_to_scan)
        progress_bar = st.progress(0)
        status = st.status("🔄 Initializing scan...", expanded=False)
        start = time.time()

        for i, sym in enumerate(symbols_to_scan):
            elapsed = time.time() - start
            eta = (elapsed/(i+1))*(total-i-1) if i>0 else 0
            status.update(label=f"🔍 Scanning {i+1}/{total}: {sym} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            data = self.fetcher.get_multi_timeframe_data(sym, timeframes)
            if data:
                signal = self.analyzer.analyze(sym, data)
                if signal:
                    # apply filters
                    if filters.get('near_support_only') and not signal.get('near_support'):
                        pass
                    elif filters.get('near_resistance_only') and not signal.get('near_resistance'):
                        pass
                    else:
                        results.append(signal)
                        st.toast(f"📈 New {signal['grade']} signal: {signal['symbol']} ({signal['signal']})", icon="🎯")
            progress_bar.progress((i+1)/total)

        status.update(label=f"✅ Scan completed in {time.time()-start:.1f}s", state="complete")
        progress_bar.empty()
        return results

# ==================== STREAMLIT UI ====================
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = MEXCDataFetcher()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MultiTimeframeAnalyzer()
if 'scanner' not in st.session_state:
    st.session_state.scanner = Scanner(st.session_state.fetcher, st.session_state.analyzer)
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

with st.sidebar:
    st.header("🔍 SCAN SETTINGS")
    available_tfs = ['15m','1h','4h','1d','1w','1M']
    selected_tfs = st.multiselect("Timeframes", available_tfs, default=['1d','4h','1h','15m'])
    if selected_tfs:
        st.session_state.analyzer.timeframes = selected_tfs

    st.divider()
    st.subheader("🎯 Filters")
    filter_near_support = st.checkbox("Near Support only")
    filter_near_resistance = st.checkbox("Near Resistance only")
    filters = {
        'near_support_only': filter_near_support,
        'near_resistance_only': filter_near_resistance
    }

    st.divider()
    st.subheader("💰 Position Sizing")
    account = st.number_input("Balance (USDT)", 10, value=1000, step=100)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    leverage = st.number_input("Leverage", 1, value=1, step=1)

    st.divider()
    if st.button("🚀 START DIVERSE SCAN (100 coins)", use_container_width=True, type="primary"):
        st.session_state.scanning = True
    auto_refresh = st.checkbox("Auto-refresh every 30s", False)

if st.session_state.scanning:
    with st.spinner("Fetching ticker data..."):
        ticker_df = st.session_state.fetcher.fetch_tickers()
    if ticker_df.empty:
        st.error("Failed to get ticker data. Please try again.")
        st.session_state.scanning = False
    else:
        st.info(f"Fetched {len(ticker_df)} coins. Selecting 100 diverse symbols...")
        results = st.session_state.scanner.scan(ticker_df, filters, account, risk_pct, leverage,
                                                st.session_state.analyzer.timeframes)
        st.session_state.scan_results = results
        st.session_state.scanning = False
        st.rerun()

if st.session_state.scan_results:
    results = st.session_state.scan_results
    results.sort(key=lambda x: x['confidence'], reverse=True)

    st.success(f"✅ Found {len(results)} opportunities")

    long = [r for r in results if r['signal']=='long']
    short = [r for r in results if r['signal']=='short']
    col1, col2, col3 = st.columns(3)
    col1.metric("LONG", len(long))
    col2.metric("SHORT", len(short))
    col3.metric("NEUTRAL", len(results)-len(long)-len(short))

    tab_long, tab_short = st.tabs(["📈 LONG SIGNALS", "📉 SHORT SIGNALS"])

    def display_signals(sig_list, is_long):
        for sig in sig_list:
            if sig['atr'] and not np.isnan(sig['atr']):
                stop = sig['price'] - 2*sig['atr'] if is_long else sig['price'] + 2*sig['atr']
                target1 = sig['price'] + 3*sig['atr'] if is_long else sig['price'] - 3*sig['atr']
                target2 = sig['price'] + 5*sig['atr'] if is_long else sig['price'] - 5*sig['atr']
            else:
                stop = sig['price']*0.98 if is_long else sig['price']*1.02
                target1 = sig['price']*1.03 if is_long else sig['price']*0.97
                target2 = sig['price']*1.05 if is_long else sig['price']*0.95

            pos = PositionSizer.calculate(account, risk_pct, sig['price'], stop, leverage)
            grade_class = "grade-a1" if sig['grade']=="A1" else "grade-a" if sig['grade']=="A" else "grade-b"
            sup_str = f"${sig['near_support']:.4f}" if sig['near_support'] else "N/A"
            res_str = f"${sig['near_resistance']:.4f}" if sig['near_resistance'] else "N/A"

            st.markdown(f"""
            <div class='signal-card {"long-card" if is_long else "short-card"}'>
                <div style='display: flex; justify-content: space-between;'>
                    <h2>{sig['symbol']}</h2>
                    <span><span class='{grade_class}'>{sig['grade']}</span> | {sig['confidence']}%</span>
                </div>
                <h3>{'LONG' if is_long else 'SHORT'} at ${sig['price']:.4f}</h3>
                <p>{' | '.join(sig['reasons'])}</p>
                <div style='display: grid; grid-template-columns: repeat(4,1fr); gap:10px; margin:15px 0;'>
                    <div class='metric-box'>RSI: {sig['rsi']:.1f}</div>
                    <div class='metric-box'>Vol: {sig['volume_ratio']:.1f}x</div>
                    <div class='metric-box'>ATR: ${sig['atr']:.4f}</div>
                    <div class='metric-box'>Trend: {sig['trendline']}</div>
                </div>
                <p>Near support: {sup_str} | Near resistance: {res_str}</p>
            """, unsafe_allow_html=True)

            if pos:
                st.markdown(f"""
                <div style='background:#2d2d2d; padding:10px; border-radius:8px; margin-top:10px;'>
                    <b>Position Plan</b><br>
                    Entry: ${sig['price']:.4f} | Stop: ${stop:.4f}<br>
                    Target1: ${target1:.4f} | Target2: ${target2:.4f}<br>
                    Size: {pos['position_size']:.4f} units | Value: ${pos['position_value']:.2f}<br>
                    Margin: ${pos['required_margin']:.2f} | Risk: ${pos['risk_amount']:.2f} ({risk_pct}%)
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_long:
        if long:
            display_signals(long, is_long=True)
        else:
            st.info("No long signals")
    with tab_short:
        if short:
            display_signals(short, is_long=False)
        else:
            st.info("No short signals")
else:
    st.info("👈 Click 'START DIVERSE SCAN' to begin")

st.divider()
st.caption(f"🔄 Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total signals: {len(st.session_state.scan_results)}")

if auto_refresh:
    time.sleep(30)
    st.rerun()
