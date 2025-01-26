import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import hashlib
import datetime
import plotly.graph_objects as go
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase using Streamlit secrets
if "firebase_initialized" not in st.session_state:
    try:
        if len(firebase_admin._apps) == 0:  # Check if Firebase is already initialized
            # Fetch Firebase credentials from Streamlit secrets
            firebase_creds = st.secrets["firebase_creds"]

            # Correctly handle line breaks in private_key if necessary
            private_key = firebase_creds["private_key"].replace(r'\n', '\n')  # Ensure line breaks are fixed

            # Rebuild the credential dictionary with the fixed private_key
            firebase_creds_fixed = {
                "type": firebase_creds["type"],
                "project_id": firebase_creds["project_id"],
                "private_key_id": firebase_creds["private_key_id"],
                "private_key": private_key,
                "client_email": firebase_creds["client_email"],
                "client_id": firebase_creds["client_id"],
                "auth_uri": firebase_creds["auth_uri"],
                "token_uri": firebase_creds["token_uri"],
                "auth_provider_x509_cert_url": firebase_creds["auth_provider_x509_cert_url"],
                "client_x509_cert_url": firebase_creds["client_x509_cert_url"]
            }

            # Initialize Firebase with the fixed credentials
            cred = credentials.Certificate(firebase_creds_fixed)
            firebase_admin.initialize_app(cred)

        st.session_state["firebase_initialized"] = True  # Mark Firebase as initialized
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        st.session_state["firebase_initialized"] = False  # Mark as failed to initialize

# Firestore client
if "firebase_initialized" in st.session_state and st.session_state["firebase_initialized"]:
    db = firestore.client()  # Connect to Firestore
else:
    st.error("Firebase is not initialized.")

# Utility Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Streamlit App Initialization
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None

# Login Persistence
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login(email, password):
    user_doc = db.collection("users").where("email", "==", email).limit(1).stream()
    user = next(user_doc, None)
    if user and user.to_dict()["password"] == hash_password(password):
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        st.session_state["logged_in"] = True
        return True
    return False

def register(email, password, phone):
    password_hash = hash_password(password)
    user_doc = db.collection("users").where("email", "==", email).limit(1).stream()
    if next(user_doc, None):
        return False  # User already exists
    db.collection("users").add({
        "email": email,
        "password": password_hash,
        "phone": phone,
        "plan": "Free"
    })
    return True

def logout():
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None
    st.session_state["logged_in"] = False
    st.session_state["portfolio"] = []
    st.session_state.clear()  # Clear the session state
    st.rerun()  # Re-run the app after logout

# Portfolio Management Functions
def fetch_portfolio(email):
    portfolio_ref = db.collection("portfolios").document(email)
    portfolio_doc = portfolio_ref.get()
    if portfolio_doc.exists:
        return portfolio_doc.to_dict().get("stocks", [])
    return []

def save_portfolio(email, portfolio):
    db.collection("portfolios").document(email).set({"stocks": portfolio})

def calculate_current_value(ticker, quantity):
    try:
        stock = yf.Ticker(f"{ticker}.KA")
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        return current_price * quantity
    except Exception:
        return None

# Forecasting Functions
def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df["Close"]

def resample_monthly(data):
    return data.resample("M").last()

def fit_sarimax(data):
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return model.fit(disp=False)

def forecast(model, steps):
    return model.forecast(steps=steps)

# App Layout
st.title("PSX Stock Forecasting & Portfolio Manager")
st.sidebar.title("Authentication")

# Check if already authenticated
if st.session_state["authenticated"]:
    st.sidebar.success(f"Welcome back, {st.session_state['user_email']}!")

    # Logout Button
    if st.sidebar.button("Logout"):
        logout()

    # Portfolio Management
    st.sidebar.title("Portfolio Management")
    portfolio = st.session_state.get("portfolio", [])

    st.sidebar.write("### Add Stock to Portfolio")
    ticker = st.sidebar.text_input("Ticker Symbol (e.g., HUBC)").upper()
    quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
    purchase_price = st.sidebar.number_input("Purchase Price (PKR)", min_value=0.0, value=0.0)
    purchase_date = st.sidebar.date_input("Purchase Date", value=datetime.date.today())

    if st.sidebar.button("Add Stock"):
        if ticker and purchase_price > 0:
            current_value = calculate_current_value(ticker, quantity)
            if current_value:
                portfolio.append({
                    "ticker": ticker,
                    "quantity": quantity,
                    "purchase_value": purchase_price * quantity,
                    "purchase_date": purchase_date.strftime("%Y-%m-%d"),
                    "current_value": current_value
                })
                save_portfolio(st.session_state["user_email"], portfolio)
                st.session_state["portfolio"] = portfolio  # Update the portfolio in session state
                st.sidebar.success(f"Added {ticker} to portfolio!")
            else:
                st.sidebar.error("Invalid ticker symbol.")

    st.sidebar.write("### Current Portfolio Summary")
    total_value = 0.0
    total_purchase_value = 0.0
    total_gain_loss = 0.0

    if portfolio:
        for i, stock in enumerate(portfolio):
            total_value += stock['current_value']
            total_purchase_value += stock['purchase_value']
            gain_loss = stock['current_value'] - stock['purchase_value']
            total_gain_loss += gain_loss
            
            st.sidebar.write(f"**{stock['ticker']}**")
            st.sidebar.write(f"- Quantity: {stock['quantity']}")
            st.sidebar.write(f"- Purchase Date: {stock['purchase_date']}")
            st.sidebar.write(f"- Purchase Value: PKR {stock['purchase_value']:.2f}")
            st.sidebar.write(f"- Current Value: PKR {stock['current_value']:.2f}")
            st.sidebar.write(f"- Gain/Loss: PKR {gain_loss:.2f}")
            
            if st.sidebar.button(f"Remove {stock['ticker']}", key=f"remove_{i}"):
                portfolio.pop(i)
                save_portfolio(st.session_state["user_email"], portfolio)
                st.session_state["portfolio"] = portfolio  # Update portfolio after removal
                st.sidebar.success(f"Removed {stock['ticker']} from portfolio!")
                st.rerun()

        # Display Total Portfolio Summary
        st.sidebar.write(f"**Total Portfolio Summary:**")
        st.sidebar.write(f"- Total Portfolio Value: PKR {total_value:.2f}")
        st.sidebar.write(f"- Total Purchase Value: PKR {total_purchase_value:.2f}")
        st.sidebar.write(f"- Total Gain/Loss: PKR {total_gain_loss:.2f}")
    else:
        st.sidebar.write("No stocks in portfolio yet.")


    # Main Forecasting Section
    st.subheader("Stock Forecasting")
    user_input = st.text_input("Enter PSX Ticker Symbol (e.g., HUBC):", value="HUBC").upper()
    ticker = f"{user_input}.KA"

    if st.button("Generate Forecast"):
        try:
            df = fetch_stock_data(ticker)
            if df.empty:
                st.error("Invalid ticker symbol.")
            else:
                current_price = df.iloc[-1]
                st.write(f"Current Price of **{user_input}**: PKR {current_price:.2f}")
                st.write(f"Generating Forecast for **{user_input}** Please wait...")
                monthly_data = resample_monthly(df)
                model = fit_sarimax(monthly_data)
                forecast_steps = 6
                future_forecast = forecast(model, forecast_steps)
                future_dates = pd.date_range(monthly_data.index[-1], periods=forecast_steps + 1, freq="M")[1:]
                forecast_df = pd.DataFrame({
                    "Date": future_dates.strftime("%d-%m-%Y"),
                    "Forecast": future_forecast
                })
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
        except Exception as e:
            st.error(f"Error: {e}")

    # Recommendation Section
    st.subheader("Stock Recommendations")
    st.write("Here are some recommendations for stocks to watch:")
    st.write("1. **HUBC**: A strong performer with consistent growth.")
    st.write("2. **OGDC**: Promising due to the increasing oil prices.")
    st.write("3. **PSO**: A major player in the oil sector.")
    st.write("4. **Ufone**: Excellent growth potential in the telecom sector.")

else:
    st.markdown("*Please login/register to proceed with forecasting, portfolio management and stock recommendations...*")
    # Login & Register Section
    auth_option = st.sidebar.selectbox("Choose an option:", ["Login", "Register"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_option == "Register":
        phone = st.sidebar.text_input("Phone Number")
        if st.sidebar.button("Register"):
            if email and password and phone:
                if register(email, password, phone):
                    st.sidebar.success("Registered successfully! Please log in.")
                else:
                    st.sidebar.error("User already exists.")
            else:
                st.sidebar.error("All fields are mandatory.")

    elif auth_option == "Login":
        if st.sidebar.button("Login"):
            if login(email, password):
                st.sidebar.success("Logged in successfully!")
                st.session_state["user_email"] = email
                st.session_state["portfolio"] = fetch_portfolio(email)
                st.rerun()  # Rerun to show portfolio and other content after successful login
            else:
                st.sidebar.error("Invalid credentials.")
                
st.markdown("---")
st.markdown("**Developed by [Basharat Hussain](basharathussain.com)**")

