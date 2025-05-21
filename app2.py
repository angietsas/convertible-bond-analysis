import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(layout="wide")
st.title("Convertible Bond CoC Explorer")

# Load and clean data
df_data = pd.read_csv("bond_data.csv")
df_data["Pricing Date"] = pd.to_datetime(df_data["Pricing Date"], errors="coerce")
df_data["Maturity Date"] = pd.to_datetime(df_data["Maturity Date"], errors="coerce")

bond_row = None
col1, col2, col3 = st.columns([2, 2, 1])

# UI inputs
with col1:
    ticker = st.text_input("Enter Ticker (e.g. CRM)").upper()

# Default values for dropdown and button
bond_name = None
pressedButton = False

# Check if a ticker is entered
if ticker:
    ticker_bonds = df_data[df_data["Ticker from BQNT"] == ticker]

    if ticker_bonds.empty:
        st.warning(f"No bonds found for ticker: {ticker}")

        with col2:
            st.selectbox("Select Bond", ["No bonds available"], disabled=True)
        with col3:
            st.markdown("<div style='height: 28px' />", unsafe_allow_html=True)
            st.button("Create Graph", disabled=True)

    else:
        with col2:
            bond_name = st.selectbox(
                "Select Bond", ticker_bonds["Name"].tolist(), disabled=False
            )
        with col3:
            st.markdown("<div style='height: 28px' />", unsafe_allow_html=True)
            pressedButton = st.button("Create Graph", disabled=False)
else:
    # Show disabled dropdown and button by default
    with col2:
        st.selectbox("Select Bond", ["No bonds available"], disabled=True)
    with col3:
        st.markdown("<div style='height: 28px' />", unsafe_allow_html=True)
        st.button("Create Graph", disabled=True)

# Create a placeholder for the graphs
graph_placeholder = st.empty()

# Proceed if the button is pressed
if pressedButton and bond_name:
    bond_row = ticker_bonds[ticker_bonds["Name"] == bond_name].iloc[0]

    # Extract bond-specific values
    start_date = bond_row["Pricing Date"] - pd.DateOffset(months=3)
    end_date = bond_row["Maturity Date"]
    conv_premium = float(bond_row["Conversion Premium"])
    upper_premium = float(bond_row["Effective Premium"])
    derivative_cost_percent = float(bond_row["Cost in Bond Points"])
    principal = float(bond_row["Issue Size ($mm)"])
    coupon = float(bond_row["Coupon"])
    conversion_price = float(bond_row["Conversion Price (adjusted)"])
    term = float(bond_row["Term"])

    @st.cache_data(show_spinner=False)
    def get_data(ticker, start_date, end_date):
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.stack(level=1, future_stack=True)
        df = df.reset_index()[["Date", "Close"]].rename(
            columns={"Date": "Date", "Close": "Last Price"}
        )
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.error("No stock data available.")
        st.stop()

    initial_price = df["Last Price"].iloc[0]
    upper_strike = initial_price * (1 + upper_premium / 100)
    conversion_price = initial_price * (1 + conv_premium / 100)
    derivative_cost = principal * (derivative_cost_percent / 100)
    underlying_shares = principal / conversion_price

    def coc(_, stock_price):
        semi_coupon = principal * (coupon / 100) / 2
        n_periods = int(term * 2)
        max_conv = max(0, stock_price - conversion_price)
        max_upper = max(0, stock_price - upper_strike)
        dilution = max(
            0,
            (stock_price * underlying_shares - principal)
            - underlying_shares * (max_conv - max_upper),
        )
        flows = (
            [principal - derivative_cost]
            + [-semi_coupon] * (n_periods - 1)
            + [-(semi_coupon + principal + dilution)]
        )
        return npf.irr(flows) * 2 * 100

    def effective_premium(
        principal, conversion_price, upper_strike, current_price, initial_price
    ):
        if current_price <= upper_strike:
            return (upper_strike - initial_price) / initial_price
        hedge = (
            (upper_strike - conversion_price)
            * (principal / conversion_price)
            / current_price
        )
        net_shares = (principal / conversion_price) - hedge
        if net_shares <= 0:
            return np.nan
        return (principal / net_shares - initial_price) / initial_price

    # Precompute CoC and Effective Premium
    df["CoC"] = df.apply(lambda row: coc(initial_price, row["Last Price"]), axis=1)
    df["Effective_Premium"] = df.apply(
        lambda row: effective_premium(
            principal, conversion_price, upper_strike, row["Last Price"], initial_price
        )
        * 100,
        axis=1,
    )

    # Plotly animated chart
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.4, 0.4],
        row_heights=[0.7, 0.7],
        specs=[
            [{"colspan": 2, "type": "table"}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        shared_xaxes=True,
    )

    # First trace - Table
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Date",
                    "Price",
                    "CoC (%)",
                    "Eff. Premium (%)",
                ],
                fill_color="#111",
                font=dict(color="white", size=16, family="Arial Black"),
                align="center",
                height=28,
            ),
            cells=dict(
                values=[
                    [df["Date"][0].strftime("%Y-%m-%d")],
                    [f"{df['Last Price'].iloc[0]:.2f}"],
                    [f"{df['CoC'].iloc[0]:.2f}%"],
                    [f"{df['Effective_Premium'].iloc[0]:.2f}%"],
                ],
                fill_color="#111",
                font=dict(color="lightgrey", size=16),
                align="center",
                height=28,
            ),
            domain=dict(x=[0.2, 0.8]),
        ),
        row=1,
        col=1,
    )

    # Second trace - Stock Price
    fig.add_trace(
        go.Scatter(
            x=[df["Date"][0]],
            y=[df["Last Price"].iloc[0]],
            mode="lines",
            name="Stock Price",
            line=dict(color="cyan"),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Third trace - Effective Premium
    fig.add_trace(
        go.Scatter(
            x=[df["Date"][0]],
            y=[df["Effective_Premium"].iloc[0]],
            mode="lines",
            name="Effective Premium",
            line=dict(color="lightgrey"),
            xaxis="x3",
            yaxis="y3",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Fourth trace - Current date line for stock price
    fig.add_trace(
        go.Scatter(
            x=[df["Date"][0], df["Date"][0]],
            y=[df["Last Price"].min() * 0.95, df["Last Price"].max() * 1.05],
            mode="lines",
            name="Current Date",
            line=dict(color="white", width=2),
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Fifth trace - Current date line for premium
    fig.add_trace(
        go.Scatter(
            x=[df["Date"][0], df["Date"][0]],
            y=[0, 1],
            mode="lines",
            name="Current Date",
            line=dict(color="white", width=2),
            showlegend=False,
            xaxis="x3",
            yaxis="y3",
        ),
        row=2,
        col=2,
    )

    # Sample dataframe to limit number of frames
    max_frames = 100
    if len(df) > max_frames:
        # Calculate step size to get ~100 evenly spaced samples
        step = len(df) // max_frames
        df_sampled = df.iloc[::step]
        # Always include the last row
        if df_sampled.index[-1] != df.index[-1]:
            df_sampled = pd.concat([df_sampled, df.iloc[[-1]]])
    else:
        df_sampled = df

    # Frames
    frames = []
    slider_steps = []
    for i, (idx, row) in enumerate(df_sampled.iterrows()):
        date = row["Date"]
        price = row["Last Price"]
        df_until_now = df[df["Date"] <= date]

        frames.append(
            go.Frame(
                data=[
                    go.Table(
                        header=dict(
                            values=[
                                "Date",
                                "Price",
                                "CoC (%)",
                                "Eff. Premium (%)",
                            ],
                            fill_color="#111",
                            font=dict(color="white", size=16, family="Arial Black"),
                            align="center",
                            height=28,
                        ),
                        cells=dict(
                            values=[
                                [date.strftime("%Y-%m-%d")],
                                [f"{price:.2f}"],
                                [f"{row['CoC'] * 100:.0f}%"],
                                [f"{row['Effective_Premium'] * 100:.0f}%"],
                            ],
                            fill_color="#111",
                            font=dict(color="lightgrey", size=16),
                            align="center",
                            height=28,
                        ),
                        domain=dict(x=[0.2, 0.8]),
                    ),
                    go.Scatter(
                        x=df_until_now["Date"],
                        y=df_until_now["Last Price"],
                        mode="lines",
                        line=dict(color="cyan"),
                        xaxis="x2",
                        yaxis="y2",
                    ),
                    go.Scatter(
                        x=[date, date],
                        y=[
                            df["Last Price"].min() * 0.95,
                            df["Last Price"].max() * 1.05,
                        ],
                        mode="lines",
                        line=dict(color="white", width=2),
                        xaxis="x2",
                        yaxis="y2",
                    ),
                    go.Scatter(
                        x=df_until_now["Date"],
                        y=df_until_now["Effective_Premium"],
                        mode="lines",
                        line=dict(color="lightgrey"),
                        xaxis="x3",
                        yaxis="y3",
                    ),
                    go.Scatter(
                        x=[date, date],
                        y=[0, 1],
                        mode="lines",
                        line=dict(color="white", width=2),
                        xaxis="x3",
                        yaxis="y3",
                    ),
                ],
                name=str(i),
            )
        )

        slider_steps.append(
            {
                "args": [
                    [str(i)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    },
                ],
                "label": date.strftime("%Y-%m-%d"),
                "method": "animate",
            }
        )

    fig.frames = frames
    fig.update(data=frames[0].data)
    fig.update_layout(
        xaxis=dict(title="Date", range=[df["Date"].min(), df["Date"].max()]),
        xaxis2=dict(
            title="Date",
            domain=[0.05, 0.65],  # Allocate the left half for Stock Price
            range=[df["Date"].min(), df["Date"].max()],
        ),
        xaxis3=dict(
            title="Date",
            domain=[0.75, 0.95],  # Allocate the right half for Effective Premium
            range=[df["Date"].min(), df["Date"].max()],
        ),
        yaxis2=dict(
            title="Stock Price",
            domain=[0.0, 0.7],  # Ensure the same vertical domain
            anchor="x2",
        ),
        yaxis3=dict(
            title="Effective Premium (%)",
            tickformat=".0%",
            domain=[0.0, 0.7],  # Match the vertical domain of Stock Price
            anchor="x3",
        ),
        font=dict(color="white"),
        plot_bgcolor="#222",
        paper_bgcolor="#222",
        width=1500,
        height=900,
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="center",
                currentvalue={
                    "font": {"size": 18},
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "right",
                },
                transition={"duration": 300, "easing": "cubic-in-out"},
                pad={"b": 30, "t": 50},
                len=0.6,
                x=0.5,
                y=0,
                steps=slider_steps,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="right",
                bordercolor="white",
                x=0.7,
                y=0.9,
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                            },
                        ],
                    ),
                    dict(
                        label="⏪ Rewind",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                                "direction": "reverse",
                            },
                        ],
                    ),
                ],
            )
        ],
        shapes=[
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=df["Date"].min(),
                x1=df["Date"].max(),
                y0=conversion_price,
                y1=upper_strike,
                fillcolor="rgba(0, 255, 0, 0.15)",
                line=dict(width=0),
                layer="below",
            ),
            dict(
                type="rect",
                xref="x2",
                yref="y2",
                x0=df["Date"].min(),
                x1=df["Date"].max(),
                y0=upper_strike,
                y1=df["Last Price"].max() + 5,
                fillcolor="rgba(255, 0, 0, 0.15)",
                line=dict(width=0),
                layer="below",
            ),
        ],
        transition={"duration": 0},
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    with graph_placeholder:
        st.info(
            "Select a ticker and bond, then press 'Create Graph' to view the analysis."
        )
