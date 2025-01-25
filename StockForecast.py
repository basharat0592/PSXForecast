import streamlit as st
import yfinance as yf
import pandas as pd
import wbdata
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import datetime


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


def fetch_kse100_data():
    """Fetch historical KSE 100 Index data from Yahoo Finance."""
    kse100 = yf.Ticker("^KSE")  # KSE 100 index ticker symbol
    df = kse100.history(period="10y")
    df = df['Close']
    df.index = pd.to_datetime(df.index)
    return resample_monthly(df)


def fetch_world_bank_data():
    """Fetch inflation and interest rate data from World Bank."""
    # Define the time range
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime.now()

    # Fetch inflation data for Pakistan
    inflation = wbdata.get_dataframe("FP.CPI.TOTL.ZG", country="PK", data_date=(start_date, end_date))
    # Fetch interest rate data for Pakistan
    interest_rate = wbdata.get_dataframe("FR.INR.RINR", country="PK", data_date=(start_date, end_date))

    # Resample to monthly and fill missing values
    inflation = inflation.resample('M').mean().fillna(method='ffill')
    interest_rate = interest_rate.resample('M').mean().fillna(method='ffill')

    return inflation, interest_rate


def fit_sarimax(data, exog):
    """Fit a SARIMAX model on the data with exogenous variables."""
    model = SARIMAX(data, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return model.fit(disp=False)


def forecast(model, steps, exog_future):
    """Generate forecasts using the SARIMAX model."""
    return model.forecast(steps=steps, exog=exog_future)


# Streamlit App
st.title("PSX Stock Forecasting System")
st.write("Forecast the next 1 to 6 months of stock prices for PSX tickers with real exogenous variables.")

# User Input
user_input = st.text_input("Enter PSX Ticker Symbol (e.g., HUBC):", value="HUBC").upper()
ticker = f"{user_input}.KA"  # Append .KA to the input

if st.button("Generate Forecast"):
    try:
        # Fetch stock data
        st.write(f"Fetching data for **{ticker}**...")
        df = fetch_stock_data(ticker)

        if df.empty:  # Check if data is empty
            st.error("Wrong Ticker! Please enter a valid PSX ticker.")
        else:
            # Resample data to monthly
            monthly_data = resample_monthly(df)

            # Fetch real exogenous variables
            st.write("Fetching exogenous variables (KSE100 Index, Inflation, Interest Rates)...")
            kse100_data = fetch_kse100_data()
            inflation, interest_rate = fetch_world_bank_data()

            # Align exogenous variables to the stock data timeline
            exog_data = pd.concat([kse100_data, inflation, interest_rate], axis=1)
            exog_data.columns = ['KSE100_Index', 'Inflation', 'Interest_Rate']
            exog_data = exog_data.fillna(method='ffill').fillna(method='bfill')  # Handle missing values

            # Ensure exog aligns with stock data
            exog_data = exog_data.loc[monthly_data.index]

            # Train the SARIMAX model
            st.write("Training the SARIMAX model with real exogenous variables...")
            model = fit_sarimax(monthly_data, exog=exog_data)

            # Generate forecast for the next 6 months
            future_index = pd.date_range(monthly_data.index[-1], periods=7, freq='M')[1:]
            kse100_future = pd.Series([kse100_data.iloc[-1]] * 6, index=future_index)  # Mock future KSE100
            inflation_future = pd.Series([inflation.iloc[-1]] * 6, index=future_index)  # Mock future inflation
            interest_rate_future = pd.Series([interest_rate.iloc[-1]] * 6, index=future_index)  # Mock future rates
            future_exog = pd.concat([kse100_future, inflation_future, interest_rate_future], axis=1)
            future_exog.columns = ['KSE100_Index', 'Inflation', 'Interest_Rate']

            forecast_steps = 6
            forecast_values = forecast(model, steps=forecast_steps, exog_future=future_exog)

            # Prepare forecast dataframe
            forecast_df = pd.DataFrame({
                "Date": future_index,
                "Forecast": forecast_values
            })

            # Display Forecast
            st.write(f"Forecast for the next 6 months:")
            st.dataframe(forecast_df)

            # Recommendations
            st.subheader("Recommendations")
            if forecast_values[-1] > monthly_data.iloc[-1]:
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
                x=future_index, y=forecast_values,
                mode='lines+markers', name='Future Forecast',
                line=dict(color='red', dash='dash')
            ))

            # Update layout
            fig.update_layout(
                title=f"SARIMAX Forecast for {ticker} with Real Exogenous Variables",
                xaxis_title="Date",
                yaxis_title="Closing Price",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )

            st.plotly_chart(fig)

    except Exception as e:
        st.error("Wrong Ticker! Please enter a valid PSX ticker.")

# Developer Information
st.markdown("---")
st.markdown(
    """
    **Developed by [Basharat Hussain](https://basharathussain.com/)**  
    For more projects and details, visit the website.
    """
)