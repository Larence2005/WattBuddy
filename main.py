import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Initialize session state for page selection
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'Home'

def set_page_selection(page):
    st.session_state.page_selection = page

# App title
st.title("WattBuddy")
st.subheader("Your Electricity Advisor")
st.write('\n')

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Home", use_container_width=True, on_click=set_page_selection, args=("Home",)):
        pass  # Selection handled by callback
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=("About",)):
        pass  # Selection handled by callback

# Sidebar for budget and pricing input
st.sidebar.header("Budget and Pricing Input")
budget = st.sidebar.number_input("Enter your budget in Php:", min_value=0.0, value=1000.0, step=50.0)
price_per_kwh = st.sidebar.number_input("Enter Meralco's price per kWh in Php:", min_value=0.0, value=10.0, step=0.1)

# Home page content
if st.session_state.page_selection == "Home":
    # Add appliances section
    st.write('\n')
    st.subheader("Add Appliances")
    if "appliances" not in st.session_state:
        st.session_state["appliances"] = []

    # Form to add appliances
    with st.form("add_appliance_form"):
        appliance_name = st.text_input("Appliance Name:")
        wattage = st.number_input("Wattage (in Watts):", min_value=0.0, step=1.0)
        hours_used = st.number_input("Usage Time (in Hours):", min_value=0.0, step=0.1)
        add_appliance = st.form_submit_button("Add Appliance")
        if add_appliance:
            if appliance_name and wattage and hours_used:
                st.session_state["appliances"].append({
                    "Name": appliance_name,
                    "Wattage (W)": wattage,
                    "Hours Used": hours_used,
                    "kWh Consumed": wattage * hours_used / 1000,
                    "Cost (Php)": (wattage * hours_used / 1000) * price_per_kwh
                })
                st.success(f"{appliance_name} added successfully!")

    # Display appliances
    if st.session_state["appliances"]:
        st.subheader("Appliance List")
        df = pd.DataFrame(st.session_state["appliances"])
        st.dataframe(df)

        # Total consumption and cost
        total_cost = df["Cost (Php)"].sum()
        total_kwh = df["kWh Consumed"].sum()

        # Calculate monthly values
        monthly_cost = total_cost * 30  # Assuming 30 days in a month
        monthly_kwh = total_kwh * 30  # Assuming usage is similar every day

        # Display total and monthly stats
        st.write('\n')
        st.write(f"### Electric Cost (Per Day): Php {total_cost:.2f}")
        st.write('\n')
        st.write(f"### kWh Consumption (Per Day): {total_kwh:.2f} kWh")
        st.write('\n')
        st.write(f"### Electric Cost (Monthly): Php {monthly_cost:.2f}")
        st.write('\n')
        st.write(f"### kWh Consumption (Monthly): {monthly_kwh:.2f} kWh")
        st.write('\n')

        # Cost status (monthly cost vs. budget)
        if monthly_cost <= budget * 0.7:
            st.success("Your monthly electric cost is GOOD!")
        elif monthly_cost <= budget:
            st.warning("Your monthly electric cost is BALANCED!")
        else:
            st.error("Your monthly electric cost is HIGH!")

        # Wattage percentage graph
        st.write('\n')
        st.subheader("Wattage Percentage Graph")
        fig, ax = plt.subplots()
        wattage_percentages = (df["Wattage (W)"] / df["Wattage (W)"].sum()) * 100
        ax.pie(wattage_percentages, labels=df["Name"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    else:
        st.info("Add appliances to calculate and analyze.")

# About page content
elif st.session_state.page_selection == "About":
    st.header("About WattBuddy")
    st.write("""
        WattBuddy helps users manage their electricity consumption and budget. By entering the cost of electricity and 
        adding appliances, users can calculate the total cost and consumption based on their usage.
        The app also provides suggestions for adjusting appliance usage to stay within the given budget.
    """)
