import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from streamlit_navigation_bar import st_navbar

# Set page configuration
st.set_page_config(initial_sidebar_state="collapsed")

# Define navigation bar pages and styles
pages = ["About", "Budget and Pricing", "Suggest Appliances"]
styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

# Render the navigation bar
page = st_navbar(pages, styles=styles)

# Content based on selected page
if page == "Budget and Pricing":
    # Budget and Pricing Input Section
    st.title("Budget and Pricing")
    st.write('\n')
    budget = st.number_input("Enter your budget in Php:", min_value=0.0, value=1000.0, step=50.0)
    price_per_kwh = st.number_input("Enter Meralco's price per kWh in Php:", min_value=0.0, value=11.8569, step=0.1)
    st.write('\n')

    # Add appliances section
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

    # Display and process appliances list
    if st.session_state["appliances"]:
        df = pd.DataFrame(st.session_state["appliances"])
        st.subheader("Appliance List")
        st.dataframe(df)

        # Add remove buttons
        for idx, row in df.iterrows():
            remove_button = st.button(f"Remove {row['Name']}", key=f"remove_{idx}")
            if remove_button:
                st.session_state["appliances"].pop(idx)
                st.experimental_rerun()

        # Total consumption and cost
        total_cost = df["Cost (Php)"].sum()
        total_kwh = df["kWh Consumed"].sum()
        monthly_cost = total_cost * 30
        st.write(f"#### Monthly Electric Cost: Php {monthly_cost:.2f}")

elif page == "About":
    st.title("WattBuddy")
    st.subheader("Your Electricity Advisor")
    st.write('\n')
    st.header("About WattBuddy")
    st.write("""
        WattBuddy helps users manage their electricity consumption and budget. By entering the cost of electricity and 
        adding appliances, users can calculate the total cost and consumption based on their usage.
        The app also provides suggestions for adjusting appliance usage to stay within the given budget.
    """)

elif page == "Suggest Appliances":
    st.title("Suggest Appliances")
    st.write("This page suggests appliances based on your budget.")
    st.write('Data set used for this auto suggestion is from the year 2022.')
    st.write('Rating: Php10.4')
