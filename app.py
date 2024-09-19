import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="EURUSD Candlestick Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Data Loading and Caching
# ------------------------------
@st.cache_data
def load_data(path):
    """
    Load CSV data into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(path)
    return df

# ------------------------------
# Pivot Point Detection
# ------------------------------
def is_pivot(df, candle, window):
    """
    Determine if a candle is a pivot high, pivot low, both, or neither.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        candle (int): Index of the current candle.
        window (int): Number of candles before and after to consider.

    Returns:
        int: 1 if pivot high, 2 if pivot low, 3 if both, 0 otherwise.
    """
    if candle - window < 0 or candle + window >= len(df):
        return 0

    current_high = df.at[candle, 'High']
    current_low = df.at[candle, 'Low']
    
    # Check for pivot high
    pivot_high = all(current_high > df['High'].iloc[candle - window:candle + window + 1].drop(candle))
    
    # Check for pivot low
    pivot_low = all(current_low < df['Low'].iloc[candle - window:candle + window + 1].drop(candle))
    
    if pivot_high and pivot_low:
        return 3
    elif pivot_high:
        return 1
    elif pivot_low:
        return 2
    else:
        return 0

def detect_pivots(df, window):
    """
    Apply pivot detection across the entire DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        window (int): Number of candles before and after to consider.

    Returns:
        pd.Series: Series indicating pivot type for each candle.
    """
    pivots = df.apply(lambda row: is_pivot(df, row.name, window), axis=1)
    return pivots

def calculate_point_position(row):
    """
    Determine the y-position for plotting pivot points.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        float or np.nan: Y-position for the pivot point marker.
    """
    if row['Pivot'] == 2:
        return row['Low'] - 1e-3
    elif row['Pivot'] == 1:
        return row['High'] + 1e-3
    else:
        return np.nan

# ------------------------------
# Price Channel Detection
# ------------------------------
def collect_channel(df, candle, backcandles, window):
    """
    Collect channel information using linear regression on pivot points.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        candle (int): Current candle index.
        backcandles (int): Number of candles to look back.
        window (int): Window size for pivot detection.

    Returns:
        tuple: (slope_lows, intercept_lows, slope_highs, intercept_highs, r_squared_l, r_squared_h)
    """
    start = candle - backcandles - window
    end = candle - window
    if start < 0:
        return (0, 0, 0, 0, 0, 0)
    
    local_df = df.iloc[start:end].copy()
    local_df['Pivot'] = local_df.apply(lambda row: is_pivot(df, row.name, window), axis=1)
    
    # Extract pivot highs and lows
    pivot_highs = local_df[local_df['Pivot'] == 1]
    pivot_lows = local_df[local_df['Pivot'] == 2]
    
    if len(pivot_lows) >= 2 and len(pivot_highs) >= 2:
        # Linear regression for lows
        slope_lows, intercept_lows, r_value_l, _, _ = stats.linregress(pivot_lows.index, pivot_lows['Low'])
        # Linear regression for highs
        slope_highs, intercept_highs, r_value_h, _, _ = stats.linregress(pivot_highs.index, pivot_highs['High'])
        return (slope_lows, intercept_lows, slope_highs, intercept_highs, r_value_l**2, r_value_h**2)
    else:
        return (0, 0, 0, 0, 0, 0)

# ------------------------------
# Breakout Detection
# ------------------------------
def is_breakout(df, candle, backcandles, window, channel_info):
    """
    Determine if there is a breakout at the current candle.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        candle (int): Current candle index.
        backcandles (int): Number of candles to look back.
        window (int): Window size for pivot detection.
        channel_info (tuple): Channel information from collect_channel.

    Returns:
        int: 1 for breakout down, 2 for breakout up, 0 otherwise.
    """
    if candle - backcandles - window < 0:
        return 0

    slope_lows, intercept_lows, slope_highs, intercept_highs, r_sq_l, r_sq_h = channel_info
    
    prev_idx = candle - 1
    prev_high = df.at[prev_idx, 'High']
    prev_low = df.at[prev_idx, 'Low']
    prev_close = df.at[prev_idx, 'Close']
    
    curr_idx = candle
    curr_high = df.at[curr_idx, 'High']
    curr_low = df.at[curr_idx, 'Low']
    curr_close = df.at[curr_idx, 'Close']
    curr_open = df.at[curr_idx, 'Open']
    
    # Calculate channel lines
    lower_channel_prev = slope_lows * prev_idx + intercept_lows
    lower_channel_curr = slope_lows * curr_idx + intercept_lows
    upper_channel_prev = slope_highs * prev_idx + intercept_highs
    upper_channel_curr = slope_highs * curr_idx + intercept_highs
    
    # Breakout down
    if (prev_high > lower_channel_prev and
        prev_close < lower_channel_prev and
        curr_open < lower_channel_curr and
        curr_close < lower_channel_prev):
        return 1
    # Breakout up
    elif (prev_low < upper_channel_prev and
          prev_close > upper_channel_prev and
          curr_open > upper_channel_curr and
          curr_close > upper_channel_prev):
        return 2
    else:
        return 0

def determine_breakouts(df, backcandles, window):
    """
    Apply breakout detection across the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        backcandles (int): Number of candles to look back.
        window (int): Window size for pivot detection.

    Returns:
        pd.Series: Series indicating breakout type for each candle.
    """
    breakouts = []
    for i in range(len(df)):
        channel_info = collect_channel(df, i, backcandles, window)
        breakout = is_breakout(df, i, backcandles, window, channel_info)
        breakouts.append(breakout)
    return pd.Series(breakouts, index=df.index)

def calculate_breakpoint_position(row):
    """
    Determine the y-position for plotting breakout points.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        float or np.nan: Y-position for the breakout point marker.
    """
    if row['BreakOut'] == 2:
        return row['Low'] - 3e-3
    elif row['BreakOut'] == 1:
        return row['High'] + 3e-3
    else:
        return np.nan

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_candlestick_with_pivots(df_subset):
    """
    Plot candlestick chart with pivot points.

    Args:
        df_subset (pd.DataFrame): Subset of the DataFrame to visualize.

    Returns:
        plotly.graph_objects.Figure: Candlestick figure with pivots.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df_subset.index,
        open=df_subset['Open'],
        high=df_subset['High'],
        low=df_subset['Low'],
        close=df_subset['Close'],
        name='Candlestick'
    )])
    
    # Add pivot points
    fig.add_trace(go.Scatter(
        x=df_subset.index,
        y=df_subset['pointpos'],
        mode='markers',
        marker=dict(size=5, color='MediumPurple'),
        name='Pivot Points'
    ))
    
    fig.update_layout(
        title="Candlestick Chart with Pivot Points",
        xaxis_title="Candle Index",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        height=600
    )
    
    return fig

def plot_channels(fig, channel_info, start, end):
    """
    Add channel lines to the candlestick figure.

    Args:
        fig (plotly.graph_objects.Figure): Existing candlestick figure.
        channel_info (tuple): Channel information from collect_channel.
        start (int): Starting index for channel lines.
        end (int): Current candle index.
    """
    slope_lows, intercept_lows, slope_highs, intercept_highs, r_sq_l, r_sq_h = channel_info
    if slope_lows == 0 and slope_highs == 0:
        return  # No valid channels to plot
    
    x = np.arange(start, end + 1)
    lower_channel = slope_lows * x + intercept_lows
    upper_channel = slope_highs * x + intercept_highs
    
    # Add lower channel line
    fig.add_trace(go.Scatter(
        x=x,
        y=lower_channel,
        mode='lines',
        name='Lower Channel',
        line=dict(color='blue', dash='dash')
    ))
    
    # Add upper channel line
    fig.add_trace(go.Scatter(
        x=x,
        y=upper_channel,
        mode='lines',
        name='Upper Channel',
        line=dict(color='red', dash='dash')
    ))

def plot_breakouts(fig, df_subset):
    """
    Add breakout points to the candlestick figure.

    Args:
        fig (plotly.graph_objects.Figure): Existing candlestick figure.
        df_subset (pd.DataFrame): Subset of the DataFrame with breakouts.
    """
    fig.add_trace(go.Scatter(
        x=df_subset.index,
        y=df_subset['breakpointpos'],
        mode='markers',
        marker=dict(size=8, color='Black', symbol='hexagram'),
        name='Breakouts'
    ))

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    # Application Title
    st.title("📈 EURUSD Candlestick Analysis")

    # Load Data
    data_path = "data/EURUSD_Candlestick_1_D_BID_04.05.2003-21.01.2023.csv"
    df = load_data(data_path)
    
    # Sidebar for User Inputs
    st.sidebar.header("🔧 Parameters")
    pivot_window = st.sidebar.slider("Pivot Detection Window", min_value=1, max_value=20, value=5, step=1)
    backcandles = st.sidebar.slider("Back Candles for Channels", min_value=10, max_value=100, value=40, step=1)
    channel_window = st.sidebar.slider("Channel Detection Window", min_value=1, max_value=20, value=3, step=1)
    visualize_candles = st.sidebar.slider("Number of Candles to Visualize", min_value=50, max_value=len(df), value=200, step=10)
    
    # ------------------------------
    # Step 2: Detecting Pivot Points
    # ------------------------------
    st.header("🔍 Step 2: Detecting Pivot Points")
    df['Pivot'] = detect_pivots(df, pivot_window)
    df['pointpos'] = df.apply(calculate_point_position, axis=1)
    st.write(f"Total Pivots Detected: {df['Pivot'].isin([1,2,3]).sum()}")

    # ------------------------------
    # Step 3: Visualizing Pivot Points
    # ------------------------------
    st.subheader("📊 Step 3: Visualizing Pivot Points")
    df_pivots = df.iloc[:visualize_candles].copy()
    fig_pivots = plot_candlestick_with_pivots(df_pivots)
    st.plotly_chart(fig_pivots, use_container_width=True)
    
    # ------------------------------
    # Step 4: Price Channels
    # ------------------------------
    st.header("📈 Step 4: Price Channels Detection")
    candle = st.number_input(
        "Select Candle Index for Channel Detection",
        min_value=backcandles + channel_window,
        max_value=len(df)-1,
        value=100,
        step=1
    )
    channel_info = collect_channel(df, candle, backcandles, channel_window)
    st.write(f"**R-squared for Lows:** {channel_info[4]:.2f}")
    st.write(f"**R-squared for Highs:** {channel_info[5]:.2f}")
    
    # ------------------------------
    # Step 5: Visualizing Detected Channels
    # ------------------------------
    st.subheader("📉 Step 5: Visualizing Detected Channels")
    start_vis = max(candle - backcandles - channel_window - 5, 0)
    end_vis = candle + 200
    df_channels = df.iloc[start_vis:end_vis].copy()
    fig_channels = plot_candlestick_with_pivots(df_channels)
    start_channel = candle - backcandles - channel_window
    plot_channels(fig_channels, channel_info, start_channel, candle)
    st.plotly_chart(fig_channels, use_container_width=True)
    
    # ------------------------------
    # Step 6: Detecting Breakouts
    # ------------------------------
    st.header("🚀 Step 6: Detecting Breakouts")
    df['BreakOut'] = determine_breakouts(df, backcandles, channel_window)
    st.write(f"Total Breakouts Detected: {df['BreakOut'].isin([1,2]).sum()}")

    # ------------------------------
    # Step 7: Visualizing Breakouts
    # ------------------------------
    st.subheader("💥 Step 7: Visualizing Breakouts")
    breakout_candle = st.number_input(
        "Select Candle Index for Breakout Visualization",
        min_value=backcandles + channel_window + 5,
        max_value=len(df)-20,
        value=75,
        step=1
    )
    df_breakout = df.iloc[breakout_candle - backcandles - channel_window -5 : breakout_candle + 20].copy()
    df_breakout['breakpointpos'] = df_breakout.apply(calculate_breakpoint_position, axis=1)
    
    fig_breakouts = plot_candlestick_with_pivots(df_breakout)
    plot_breakouts(fig_breakouts, df_breakout)
    
    # Add channels for breakout visualization
    channel_info_breakout = collect_channel(df, breakout_candle, backcandles, channel_window)
    start_channel_b = breakout_candle - backcandles - channel_window
    plot_channels(fig_breakouts, channel_info_breakout, start_channel_b, breakout_candle)
    
    st.plotly_chart(fig_breakouts, use_container_width=True)

    # ------------------------------
    # Additional Insights (Optional)
    # ------------------------------
    st.header("📄 Additional Insights")
    st.write("You can further enhance this application by adding more analytical features such as:")
    st.markdown("""
    - **Statistical Summaries**: Display summaries of pivot points, channels, and breakouts.
    - **Interactive Filtering**: Allow users to filter data based on date ranges or pivot types.
    - **Export Options**: Provide options to download analysis results or figures.
    - **Alerts and Notifications**: Implement real-time alerts for detected breakouts.
    """)

if __name__ == "__main__":
    main()
