import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# Predefined list of KSE-100 stocks
kse100_stocks = [
    {"name": "Habib Bank Limited", "ticker": "HBL"},
    {"name": "Lucky Cement Limited", "ticker": "LUCK"},
    {"name": "Engro Corporation Limited", "ticker": "ENGRO"},
    {"name": "Pakistan Petroleum Limited", "ticker": "PPL"},
    {"name": "Oil & Gas Development Company", "ticker": "OGDC"},
    {"name": "Hub Power Company", "ticker": "HUBC"},
    # Add the rest of the KSE-100 stock list here
]

# Create a DataFrame for easy lookup
stocks_df = pd.DataFrame(kse100_stocks)

# Function Definitions
def fetch_stock_data(ticker, period="5y"):
    """Fetch stock data for the given ticker and period."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df['Close']  # Extract closing prices
    df.index = pd.to_datetime(df.index)  # Ensure datetime index
    return df

def resample_monthly(data):
    """Resample daily data to monthly closing prices."""
    return data.resample('M').last()

def fit_sarimax(data):
    """Fit a SARIMAX model on the data."""
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return model.fit(disp=False)

def forecast(model, steps):
    """Generate forecasts using the SARIMAX model."""
    return model.forecast(steps=steps)

# Streamlit App
st.title("PSX Stock Forecasting System")
st.write("Forecast the next 1 to 6 months of stock prices for PSX tickers.")

# Searchable dropdown for KSE-100 stocks
selected_stock = st.selectbox(
    "Search and Select a KSE-100 Stock:",
    options=stocks_df["name"] + " (" + stocks_df["ticker"] + ")",
    index=0,
    help="Start typing the stock name or ticker to filter the dropdown."
)

# Extract the ticker symbol from the dropdown
ticker_symbol = stocks_df.loc[
    stocks_df["name"] + " (" + stocks_df["ticker"] + ")" == selected_stock, "ticker"
].values[0]

# Text input for manual entry (fallback)
manual_ticker = st.text_input(
    "Or Enter a Ticker Manually (if not in the list):",
    value="",
    placeholder="e.g., PSO"
).upper()

# Decide which ticker to use
final_ticker = manual_ticker if manual_ticker else ticker_symbol
ticker = f"{final_ticker}.KA"  # Append .KA for PSX tickers

if st.button("Generate Forecast"):
    try:
        # Fetch stock data
        st.write(f"Fetching data for **{final_ticker}**...")
        df = fetch_stock_data(ticker)
        if df.empty:  # Check if data is empty
            st.error("No data found for the selected stock. Please check the ticker.")
        else:
            # Display the current price
            current_price = df.iloc[-1]  # Get the most recent closing price
            st.markdown(f"###### Current Price of **{final_ticker}**: PKR **{current_price:.2f}**")

            # Resample data and train model
            monthly_data = resample_monthly(df)
            st.write("Forecasting using AI Model...")
            model = fit_sarimax(monthly_data)
            # Generate Forecast
            forecast_steps = 6
            future_forecast = forecast(model, forecast_steps)
            future_dates = pd.date_range(monthly_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]
            forecast_df = pd.DataFrame({
                "Date": future_dates.strftime('%d-%m-%Y'),
                "Forecast": future_forecast
            })
            # Display Forecast
            st.write(f"Forecast for the next 6 months:")
            st.dataframe(forecast_df)
            # Recommendations
            st.subheader("Recommendations")
            if future_forecast[-1] > monthly_data.iloc[-1]:
                st.write("✅ **Recommendation:** The forecast suggests a potential upward trend. Consider holding or buying.")
            else:
                st.write("⚠️ **Recommendation:** The forecast suggests a potential downward trend. Consider selling or monitoring closely.")
            # Visualization
            st.subheader("Visualization")
            fig = go.Figure()

            # Filter the monthly data to the most recent months
            monthly_data = monthly_data[-36:]  # Keep only the last few months

            # Add historical data
            fig.add_trace(go.Scatter(
                x=monthly_data.index, y=monthly_data,
                mode='lines', name='Historical Data',
                line=dict(color='blue')
            ))

            # Add future forecast
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_forecast,
                mode='lines+markers', name='Future Forecast',
                line=dict(color='red', dash='dash')
            ))
            # Update layout
            fig.update_layout(
                title=f"Forecast for {final_ticker}",
                xaxis_title="Date",
                yaxis_title="Closing Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
# Developer Information
st.markdown("---")
st.markdown(
    """
    **Developed by [Basharat Hussain](https://basharathussain.com/)**  
    For more projects and details, visit the website.
    """
)
