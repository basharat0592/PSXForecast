import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
# Function Definitions
def fetch_stock_data(ticker, period="10y"):
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
# User Input
user_input = st.text_input("Enter PSX Ticker Symbol (e.g., HUBC):", value="HUBC").upper()
ticker = f"{user_input}.KA"  # Append .KA to the input
if st.button("Generate Forecast"):
    try:
        # Fetch stock data
        st.write(f"Fetching data for **{user_input}**...")
        df = fetch_stock_data(ticker)
        if df.empty:  # Check if data is empty
            st.error("Wrong Ticker! Please enter a valid PSX ticker.")
        else:
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
                title=f"Forecast for {user_input}",
                xaxis_title="Date",
                yaxis_title="Closing Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )
            st.plotly_chart(fig)
    except Exception:
        st.error("Wrong Ticker! Please enter a valid PSX ticker.")
# Developer Information
st.markdown("---")
st.markdown(
    """
    **Developed by [Basharat Hussain](https://basharathussain.com/)**  
    For more projects and details, visit the website.
    """
)