import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# App title
st.title("WattBuddy")
st.subheader("Your Electricity Advisor")
st.write('\n')

# Sidebar for navigation
if st.button("Home", use_container_width=True, on_click=set_page_selection, args=('home',)):
        st.session_state.page_selection = 'home'
elif st.sidebar.button("About"):
    page = "About"
else:
    page = "Home"  # Default to Home if no button is pressed

# Sidebar for budget and pricing input
st.sidebar.header("Budget and Pricing Input")
budget = st.sidebar.number_input("Enter your budget in Php:", min_value=0.0, value=1000.0, step=50.0)
price_per_kwh = st.sidebar.number_input("Enter Meralco's price per kWh in Php:", min_value=0.0, value=10.0, step=0.1)

# Home page content
if page == "Home":
    # Add appliances section
    st.write('\n')
    st.subheader("Add Appliances")
    if "appliances" not in st.session_state:
        st.session_state["appliances"] = []

