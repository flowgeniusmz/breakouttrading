import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go


# Load the data
data_path = "data/EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv"
df = pd.read_csv(data_path)


# Step 2 Detecting Pivot Points

### Implementing Pivot Point Detection Function
def isPivot(candle, window):
    """
    Function that detects if a candle is a pivot/fractal point.
    Args: 
        candle: index of the candle in the DataFrame
        window: number of candles before and after the current candle to consider
    Returns: 
        1 if pivot high, 2 if pivot low, 3 if both, and 0 by default
    """
    if candle - window < 0 or candle + window >= len(df):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivotLow = 0
        if df.iloc[candle].High < df.iloc[i].High:
            pivotHigh = 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0
    
### Using IsPivot function
window = 5 #ex window size
df['Pivot'] = [isPivot(candle=i, window=window) for i in range(len(df))]


# Step 3 Visualizing
### Implementing Pivot Point Position Function
def pointpos(x):
    if x['isPivot'] == 2:
        return x['Low'] - 1e-3
    elif x['isPivot'] == 1:
        return x['High'] + 1e-3
    else:
        return np.nan
    
### applying pointpos function
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)


### visualizing pivot points
# Select a subset of the DataFrame for visualization
dfpl = df[0:100]

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

# Add the pivot points as scatter plot markers
fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")

# Display the chart
fig.show()


# Step 4: Price Channels
#### Implementing channel detection function
def collect_channel(candle, backcandles, window):
    localdf = df[candle-backcandles-window:candle-window]
    localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name, window), axis=1)
    highs = localdf[localdf['isPivot'] == 1].High.values
    idxhighs = localdf[localdf['isPivot'] == 1].High.index
    lows = localdf[localdf['isPivot'] == 2].Low.values
    idxlows = localdf[localdf['isPivot'] == 2].Low.index
    
    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows, lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs, highs)
    
        return (sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return (0, 0, 0, 0, 0, 0)
    
### using the channel detection function
candle = 100
backcandles = 20
window = 5
channel_info = collect_channel(candle, backcandles, window)


#Step p5: Visualizaing Detected Channels
### Setting up visualizagion
candle = 75
backcandles = 40
window = 3

# Select a subset of the DataFrame for visualization
dfpl = df[candle-backcandles-window-5:candle+200]

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

# Add the pivot points as scatter plot markers
fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")

# Detect the channels using the collect_channel function
sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window)
print(f"R-squared for lows: {r_sq_l}, R-squared for highs: {r_sq_h}")

# Generate x values for the channel lines
x = np.array(range(candle-backcandles-window, candle+1))

# Add the lower channel line to the chart
fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='Lower Channel'))

# Add the upper channel line to the chart
fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='Upper Channel'))

# Display the chart
fig.show()


#Step 6 Detecting Breakouts
def isBreakOut(candle, backcandles, window):
    if (candle - backcandles - window) < 0:
        return 0
    
    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window)
    
    prev_idx = candle - 1
    prev_high = df.iloc[candle - 1].High
    prev_low = df.iloc[candle - 1].Low
    prev_close = df.iloc[candle - 1].Close
    
    curr_idx = candle
    curr_high = df.iloc[candle].High
    curr_low = df.iloc[candle].Low
    curr_close = df.iloc[candle].Close
    curr_open = df.iloc[candle].Open

    if (prev_high > (sl_lows * prev_idx + interc_lows) and
        prev_close < (sl_lows * prev_idx + interc_lows) and
        curr_open < (sl_lows * curr_idx + interc_lows) and
        curr_close < (sl_lows * prev_idx + interc_lows)): #and r_sq_l > 0.9
        return 1
    
    elif (prev_low < (sl_highs * prev_idx + interc_highs) and
          prev_close > (sl_highs * prev_idx + interc_highs) and
          curr_open > (sl_highs * curr_idx + interc_highs) and
          curr_close > (sl_highs * prev_idx + interc_highs)): #and r_sq_h > 0.9
        return 2
    
    else:
        return 0
    
### Using is breakdout
breakout_signals = [isBreakOut(i, backcandles, window) for i in range(len(df))]
df['BreakOut'] = breakout_signals

# Step 7 Visualizing Breakouts
def breakpointpos(x):
    if x['isBreakOut'] == 2:
        return x['Low'] - 3e-3
    elif x['isBreakOut'] == 1:
        return x['High'] + 3e-3
    else:
        return np.nan
    
candle = 75
backcandles = 40
window = 3

# Select a subset of the DataFrame for visualization
dfpl = df[candle-backcandles-window-5:candle+20]

# Detect breakouts for the selected subset
dfpl["isBreakOut"] = [isBreakOut(candle, backcandles, window) for candle in dfpl.index]

# Determine the positions of the breakouts
dfpl['breakpointpos'] = dfpl.apply(lambda row: breakpointpos(row), axis=1)

candle = 59

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])

# Add the pivot points as scatter plot markers
fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")

# Add the breakout points as scatter plot markers
fig.add_scatter(x=dfpl.index, y=dfpl['breakpointpos'], mode="markers",
                marker=dict(size=8, color="Black"), marker_symbol="hexagram",
                name="breakout")

# Detect the channels using the collect_channel function
sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window)
print(f"R-squared for lows: {r_sq_l}, R-squared for highs: {r_sq_h}")

# Generate x values for the channel lines
x = np.array(range(candle-backcandles-window, candle+1))

# Add the lower channel line to the chart
fig.add_trace(go.Scatter(x=x, y=sl_lows*x + interc_lows, mode='lines', name='Lower Channel'))

# Add the upper channel line to the chart
fig.add_trace(go.Scatter(x=x, y=sl_highs*x + interc_highs, mode='lines', name='Upper Channel'))

# Display the chart
fig.show()