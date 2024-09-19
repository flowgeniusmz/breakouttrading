import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Stock Candlestick Analysis with Real-Time Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Helper Functions
# ------------------------------

@st.cache_data
def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance using yfinance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        if df.empty:
            st.error("No data fetched. Please check the ticker symbol and date range.")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Dividends': 'Dividends',
            'Stock Splits': 'Stock Splits'
        }, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def is_pivot(df: pd.DataFrame, candle: int, window: int) -> int:
    """
    Determine if a candle is a pivot high, pivot low, both, or neither.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
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

def detect_pivots(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Apply pivot detection across the entire DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        window (int): Number of candles before and after to consider.

    Returns:
        pd.Series: Series indicating pivot type for each candle.
    """
    pivots = df.apply(lambda row: is_pivot(df, row.name, window), axis=1)
    return pivots

def calculate_point_position(row: pd.Series) -> float:
    """
    Determine the y-position for plotting pivot points.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        float or np.nan: Y-position for the pivot point marker.
    """
    if row['Pivot'] == 2:
        return row['Low'] - (row['Low'] * 0.01)  # 1% below the low
    elif row['Pivot'] == 1:
        return row['High'] + (row['High'] * 0.01)  # 1% above the high
    else:
        return np.nan

@st.cache_data
def collect_channel(df: pd.DataFrame, candle: int, backcandles: int, window: int) -> tuple:
    """
    Collect channel information using linear regression on pivot points.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
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

@st.cache_data
def determine_breakouts(df: pd.DataFrame, backcandles: int, window: int) -> pd.Series:
    """
    Apply breakout detection across the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        backcandles (int): Number of candles to look back.
        window (int): Window size for pivot detection.

    Returns:
        pd.Series: Series indicating breakout type for each candle.
    """
    breakouts = []
    total = len(df)
    progress_bar = st.progress(0)
    for i in range(len(df)):
        channel_info = collect_channel(df, i, backcandles, window)
        breakout = is_breakout(df, i, backcandles, window, channel_info)
        breakouts.append(breakout)
        if i % 100 == 0:
            progress = i / total
            progress_bar.progress(min(progress, 1.0))
    progress_bar.empty()
    return pd.Series(breakouts, index=df.index)

def is_breakout(df: pd.DataFrame, candle: int, backcandles: int, window: int, channel_info: tuple) -> int:
    """
    Determine if there is a breakout at the current candle.

    Args:
        df (pd.DataFrame): DataFrame containing stock data.
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

    if slope_lows == 0 and slope_highs == 0:
        return 0  # No valid channels

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

def calculate_breakpoint_position(row: pd.Series) -> float:
    """
    Determine the y-position for plotting breakout points.

    Args:
        row (pd.Series): A row of the DataFrame.

    Returns:
        float or np.nan: Y-position for the breakout point marker.
    """
    if row['BreakOut'] == 2:
        return row['Low'] - (row['Low'] * 0.02)  # 2% below the low
    elif row['BreakOut'] == 1:
        return row['High'] + (row['High'] * 0.02)  # 2% above the high
    else:
        return np.nan

def plot_candlestick_with_pivots(df_subset: pd.DataFrame) -> go.Figure:
    """
    Plot candlestick chart with pivot points.

    Args:
        df_subset (pd.DataFrame): Subset of the DataFrame to visualize.

    Returns:
        plotly.graph_objects.Figure: Candlestick figure with pivots.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df_subset['Date'],
        open=df_subset['Open'],
        high=df_subset['High'],
        low=df_subset['Low'],
        close=df_subset['Close'],
        name='Candlestick'
    )])

    # Add pivot points
    pivots = df_subset.dropna(subset=['pointpos'])
    fig.add_trace(go.Scatter(
        x=pivots['Date'],
        y=pivots['pointpos'],
        mode='markers',
        marker=dict(size=7, color='MediumPurple', symbol='diamond'),
        name='Pivot Points'
    ))

    fig.update_layout(
        title="Candlestick Chart with Pivot Points",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        height=600
    )

    return fig

def plot_channels(fig: go.Figure, channel_info: tuple, start: int, end: int, df: pd.DataFrame):
    """
    Add channel lines to the candlestick figure.

    Args:
        fig (plotly.graph_objects.Figure): Existing candlestick figure.
        channel_info (tuple): Channel information from collect_channel.
        start (int): Starting index for channel lines.
        end (int): Current candle index.
        df (pd.DataFrame): Complete DataFrame for date referencing.
    """
    slope_lows, intercept_lows, slope_highs, intercept_highs, r_sq_l, r_sq_h = channel_info
    if slope_lows == 0 and slope_highs == 0:
        return  # No valid channels to plot

    # Generate date range for channel lines
    date_range = df['Date'].iloc[start:end+1]
    x = date_range

    lower_channel = slope_lows * np.arange(start, end + 1) + intercept_lows
    upper_channel = slope_highs * np.arange(start, end + 1) + intercept_highs

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

def plot_breakouts(fig: go.Figure, df_subset: pd.DataFrame) -> go.Figure:
    """
    Add breakout points to the candlestick figure.

    Args:
        fig (plotly.graph_objects.Figure): Existing candlestick figure.
        df_subset (pd.DataFrame): Subset of the DataFrame with breakouts.

    Returns:
        plotly.graph_objects.Figure: Updated figure with breakouts.
    """
    breakouts = df_subset.dropna(subset=['breakpointpos'])
    if not breakouts.empty:
        fig.add_trace(go.Scatter(
            x=breakouts['Date'],
            y=breakouts['breakpointpos'],
            mode='markers',
            marker=dict(size=10, color='Black', symbol='triangle-up'),
            name='Breakouts'
        ))
    return fig

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    # Initialize session state for DataFrame
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

    # Application Title
    st.title("📈 Stock Candlestick Analysis with Real-Time Data")

    # Sidebar for User Inputs
    st.sidebar.header("🔧 Parameters")

    # Stock Ticker Selection
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()

    # Date Range Selection
    today = datetime.today()
    default_start = today - timedelta(days=365 * 5)  # 5 years ago
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=today)

    # Validate Date Inputs
    if start_date > end_date:
        st.sidebar.error("Error: End Date must fall after Start Date.")

    # Pivot Detection Parameters
    pivot_window = st.sidebar.slider("Pivot Detection Window", min_value=1, max_value=20, value=5, step=1)

    # Price Channel Parameters
    backcandles = st.sidebar.slider("Back Candles for Channels", min_value=10, max_value=100, value=40, step=1)
    channel_window = st.sidebar.slider("Channel Detection Window", min_value=1, max_value=20, value=3, step=1)

    # Visualization Parameters
    visualize_candles = st.sidebar.slider("Number of Candles to Visualize", min_value=50, max_value=1000, value=200, step=10)

    # Load Data Button
    if st.sidebar.button("Load Data"):
        with st.spinner("Fetching data..."):
            df = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not df.empty:
                st.session_state['df'] = df
                st.success("Data loaded successfully!")
    else:
        # If data is already loaded in session state, use it
        df = st.session_state['df']

    if not df.empty:
        # ------------------------------
        # Step 2: Detecting Pivot Points
        # ------------------------------
        st.header("🔍 Step 2: Detecting Pivot Points")
        df['Pivot'] = detect_pivots(df, pivot_window)
        df['pointpos'] = df.apply(calculate_point_position, axis=1)
        total_pivots = df['Pivot'].isin([1, 2, 3]).sum()
        st.write(f"**Total Pivots Detected:** {total_pivots}")

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
        max_candle = len(df) - 1
        default_candle = min(100, max_candle)
        candle = st.number_input(
            "Select Candle Index for Channel Detection",
            min_value=backcandles + channel_window,
            max_value=max_candle,
            value=default_candle,
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
        end_vis = min(candle + 200, len(df) - 1)
        df_channels = df.iloc[start_vis:end_vis].copy()
        fig_channels = plot_candlestick_with_pivots(df_channels)
        start_channel = candle - backcandles - channel_window
        plot_channels(fig_channels, channel_info, start_channel, candle, df)
        st.plotly_chart(fig_channels, use_container_width=True)

        # ------------------------------
        # Step 6: Detecting Breakouts
        # ------------------------------
        st.header("🚀 Step 6: Detecting Breakouts")
        with st.spinner("Detecting breakouts..."):
            df['BreakOut'] = determine_breakouts(df, backcandles, channel_window)
        total_breakouts = df['BreakOut'].isin([1, 2]).sum()
        st.write(f"**Total Breakouts Detected:** {total_breakouts}")

        if total_breakouts > 0:
            st.success(f"{total_breakouts} breakout(s) detected.")
            # Display a table of detected breakouts
            breakouts_df = df[df['BreakOut'].isin([1, 2])]
            st.subheader("📋 Detected Breakouts")
            st.dataframe(breakouts_df[['Date', 'Open', 'High', 'Low', 'Close', 'BreakOut']].reset_index(drop=True))
        else:
            st.warning("No breakouts detected based on the current parameters.")

        # ------------------------------
        # Step 7: Visualizing Breakouts
        # ------------------------------
        st.subheader("💥 Step 7: Visualizing Breakouts")
        if total_breakouts > 0:
            # Allow user to filter breakouts by type
            breakout_type = st.selectbox("Select Breakout Type to Visualize", options=["All", "Breakout Down", "Breakout Up"])

            if breakout_type == "All":
                filtered_breakouts = df[df['BreakOut'].isin([1, 2])]
            elif breakout_type == "Breakout Down":
                filtered_breakouts = df[df['BreakOut'] == 1]
            else:
                filtered_breakouts = df[df['BreakOut'] == 2]

            # Select a subset of the DataFrame for visualization
            # To visualize all breakouts, we can plot the entire dataset with markers
            df_vis = df.copy()
            df_vis['breakpointpos'] = df_vis.apply(calculate_breakpoint_position, axis=1)

            fig_breakouts = plot_candlestick_with_pivots(df_vis)
            fig_breakouts = plot_breakouts(fig_breakouts, df_vis)

            # Add all breakouts
            breakouts_to_plot = filtered_breakouts.dropna(subset=['breakpointpos'])
            if not breakouts_to_plot.empty:
                fig_breakouts.add_trace(go.Scatter(
                    x=breakouts_to_plot['Date'],
                    y=breakouts_to_plot['breakpointpos'],
                    mode='markers',
                    marker=dict(size=10, color='Black', symbol='triangle-up'),
                    name='Breakouts'
                ))

            # Optionally, plot channel lines for each breakout
            # This can be computationally intensive; hence, we skip it or make it optional

            fig_breakouts.update_layout(
                title="Candlestick Chart with Breakouts",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(x=0, y=1),
                height=600
            )

            st.plotly_chart(fig_breakouts, use_container_width=True)
        else:
            st.info("No breakouts to visualize.")

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
        - **Technical Indicators**: Incorporate additional technical indicators like RSI, MACD, etc.
        """)

    else:
        st.warning("Please load data by entering a valid ticker and selecting the date range.")

if __name__ == "__main__":
    main()
