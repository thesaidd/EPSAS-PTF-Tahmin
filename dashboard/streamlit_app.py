import os

import streamlit as st


st.set_page_config(
    page_title="EPİAŞ PTF Forecasting MVP",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EPİAŞ PTF Forecasting MVP")
st.subheader("Turkey Day-Ahead Market Price Forecasting")

st.info(
    "This MVP will provide hourly PTF/MCP point forecasts for the next 24 hours "
    "and residual uncertainty estimates for B2B energy-market workflows."
)

st.metric(label="API health", value="Not checked")
st.caption(f"Configured API endpoint: {os.getenv('API_URL', 'http://localhost:8000')}")

