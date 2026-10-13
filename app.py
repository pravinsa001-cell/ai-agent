import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Independent Signal Lab", page_icon="📊", layout="centered")

st.title("📊 Independent Signal Lab")
st.caption("Educational chart analysis • RSI + ATR + EMA • No broker connection or order execution")
st.warning(
    "For education and simulation only. Signals are not predictions or financial advice. "
    "This starter app uses uploaded candle data or clearly labelled synthetic demo data; "
    "it does not connect to Binomo or place trades."
)

def make_demo_data(n=180, seed=7):
    """Create synthetic OHLC candles for interface testing only, not real market data."""
    rng = np.random.default_rng(seed)
    changes = rng.normal(loc=0.015, scale=0.22, size=n)
    close = 100 + np.cumsum(changes)
    open_ = np.r_[close[0] - changes[0], close[:-1]]
    spread = rng.uniform(0.03, 0.22, size=n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"date": "timestamp", "time": "timestamp", "datetime": "timestamp",
               "o": "open", "h": "high", "l": "low", "c": "close"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV must contain columns: open, high, low, close. Missing: " + ", ".join(missing))
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    else:
        df["timestamp"] = np.arange(len(df))
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("Please upload at least 30 valid candles for this indicator setup.")
    if (df["high"] < df["low"]).any():
        raise ValueError("Some rows have high below low. Please check the candle data.")
    return df

def indicators(df):
    out = df.copy()
    close = out["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100
    out.loc[(avg_loss == 0) & (avg_gain == 0), "rsi14"] = 50

    prev_close = close.shift(1)
    true_range = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    out["atr14"] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["atr_pct"] = (out["atr14"] / close) * 100
    return out

def decide(last, previous, atr_limit_pct):
    reasons = []
    if pd.isna(last["rsi14"]) or pd.isna(last["atr14"]):
        return "NO TRADE", ["Not enough valid candles to calculate all indicators."]
    if last["atr_pct"] > atr_limit_pct:
        return "NO TRADE", [f"ATR is high relative to price ({last['atr_pct']:.3f}%). Wait for calmer conditions."]
    bullish = last["ema9"] > last["ema21"]
    bearish = last["ema9"] < last["ema21"]
    rsi = last["rsi14"]
    if bullish:
        reasons.append("EMA 9 is above EMA 21 (short-term trend filter is bullish).")
    elif bearish:
        reasons.append("EMA 9 is below EMA 21 (short-term trend filter is bearish).")
    else:
        reasons.append("EMA 9 and EMA 21 are equal; trend is unclear.")

    if bullish and 50 <= rsi <= 65 and last["close"] >= last["ema9"]:
        reasons.append(f"RSI is {rsi:.1f}, within the example bullish momentum band (50–65).")
        return "CALL SETUP", reasons
    if bearish and 35 <= rsi <= 50 and last["close"] <= last["ema9"]:
        reasons.append(f"RSI is {rsi:.1f}, within the example bearish momentum band (35–50).")
        return "PUT SETUP", reasons
    reasons.append(f"RSI is {rsi:.1f}; the complete example conditions are not aligned.")
    return "NO TRADE", reasons

with st.sidebar:
    st.header("Settings")
    atr_limit = st.slider("Skip if ATR / price is above (%)", min_value=0.05, max_value=2.0, value=0.60, step=0.05)
    st.caption("This threshold is a configurable example, not a universal market rule.")

source = st.radio("Choose candle data", ["Synthetic demo data", "Upload my candle CSV"], horizontal=False)

if source == "Upload my candle CSV":
    file = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: open, high, low, close. Optional: timestamp.")
    if file is None:
        st.info("Upload a CSV to analyse your own historical candle data.")
        st.code("timestamp,open,high,low,close\n2026-01-01 10:00:00,1.1000,1.1010,1.0995,1.1005")
        st.stop()
    try:
        raw = normalize_columns(pd.read_csv(file))
    except Exception as e:
        st.error(str(e))
        st.stop()
    source_note = "Uploaded CSV"
else:
    raw = make_demo_data()
    source_note = "SYNTHETIC DEMO DATA — not live prices"

data = indicators(raw)
valid = data.dropna(subset=["rsi14", "atr14", "ema9", "ema21"])
if valid.empty:
    st.error("Not enough data to calculate indicators.")
    st.stop()

last = valid.iloc[-1]
previous = valid.iloc[-2] if len(valid) > 1 else last
signal, reasons = decide(last, previous, atr_limit)

st.subheader("Current analysis")
st.caption(source_note)
if "SYNTHETIC" in source_note:
    st.info("Demo mode: the prices below are randomly generated for testing the app interface only.")

if signal == "CALL SETUP":
    st.success("Example rule result: CALL SETUP")
elif signal == "PUT SETUP":
    st.error("Example rule result: PUT SETUP")
else:
    st.warning("Example rule result: NO TRADE")

m1, m2 = st.columns(2)
m1.metric("Latest close", f"{last['close']:.5f}")
m2.metric("RSI (14)", f"{last['rsi14']:.1f}")
m3, m4 = st.columns(2)
m3.metric("ATR (14)", f"{last['atr14']:.5f}")
m4.metric("ATR / price", f"{last['atr_pct']:.3f}%")

st.markdown("**EMA trend**")
st.write("Bullish filter" if last["ema9"] > last["ema21"] else "Bearish filter" if last["ema9"] < last["ema21"] else "Unclear")
st.markdown("**Why the app returned this result**")
for reason in reasons:
    st.write(f"- {reason}")

chart_data = data.tail(100).set_index("timestamp")[["close", "ema9", "ema21"]]
st.subheader("Close and EMA lines")
st.line_chart(chart_data, use_container_width=True)

st.subheader("Recent candles and indicators")
show_cols = ["timestamp", "open", "high", "low", "close", "rsi14", "atr14", "ema9", "ema21"]
st.dataframe(data[show_cols].tail(15), use_container_width=True, hide_index=True)

csv_bytes = data.to_csv(index=False).encode("utf-8")
st.download_button("Download analysed data (CSV)", data=csv_bytes, file_name="analysed_candles.csv", mime="text/csv")

with st.expander("How the sample rules work"):
    st.markdown("""
    - **CALL SETUP:** EMA 9 > EMA 21, RSI between 50 and 65, latest close at/above EMA 9, and ATR/price below the chosen threshold.
    - **PUT SETUP:** EMA 9 < EMA 21, RSI between 35 and 50, latest close at/below EMA 9, and ATR/price below the chosen threshold.
    - **NO TRADE:** any required condition is missing or volatility is above the chosen threshold.

    These are simple educational rules, not a validated trading strategy. They do not estimate a reliable probability of winning.
    """)

st.caption("No live data feed, no Binomo connection, no trade execution. Validate indicator calculations and backtest with clean historical data before relying on any analysis.")
