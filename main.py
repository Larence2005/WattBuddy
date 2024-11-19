import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Initialize session state for page selection
if "page_selection" not in st.session_state:
    st.session_state.page_selection = "About"

def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=("About",)):
        pass

    if st.button("Budget and Pricing", use_container_width=True, on_click=set_page_selection, args=("Budget and Pricing",)):
        pass

    if st.button("Suggest Appliances", use_container_width=True, on_click=set_page_selection, args=("Suggest Appliances",)):
        pass

# Budget and Pricing Section
if st.session_state.page_selection == "Budget and Pricing":
    st.title("Budget and Pricing")
    st.write("\n")
    budget = st.number_input("Enter your budget in Php:", min_value=0.0, value=1000.0, step=50.0)
    price_per_kwh = st.number_input("Enter Meralco's price per kWh in Php:", min_value=0.0, value=11.8569, step=0.1)
    st.write("\n")

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
                    "Cost (Php)": (wattage * hours_used / 1000) * price_per_kwh,
                })
                st.success(f"{appliance_name} added successfully!")

    # Display appliances using AgGrid
    if st.session_state["appliances"]:
        st.subheader("Appliance List")
        df = pd.DataFrame(st.session_state["appliances"])

        # Configure AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)  # Allow all columns to be editable
        gb.configure_selection(selection_mode="single", use_checkbox=True)  # Add checkbox for selection
        gb.configure_grid_options(domLayout="normal")

        # Render AgGrid
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.MODEL_CHANGED,
            allow_unsafe_jscode=True,  # Allows Javascript code in AgGrid
            theme="streamlit",
        )

        # Handle edits and deletions
        updated_data = grid_response["data"]
        selected = grid_response["selected_rows"]

        # Update session state with edited data
        if updated_data is not None:
            st.session_state["appliances"] = updated_data

        # Remove selected appliance
        if selected:
            for item in selected:
                st.session_state["appliances"] = [
                    appliance for appliance in st.session_state["appliances"]
                    if appliance["Name"] != item["Name"]
                ]
            st.success(f"Selected appliance removed!")

        # Total consumption and cost
        df = pd.DataFrame(st.session_state["appliances"])
        total_cost = df["Cost (Php)"].sum()
        total_kwh = df["kWh Consumed"].sum()

        # Monthly values
        monthly_cost = total_cost * 30
        monthly_kwh = total_kwh * 30

        # Display total and monthly stats
        st.write("\n")
        st.write(f"#### Electric Cost (Per Day): Php {total_cost:.2f}")
        st.write(f"#### kWh Consumption (Per Day): {total_kwh:.2f} kWh")
        st.write(f"#### Electric Cost (Monthly): Php {monthly_cost:.2f}")
        st.write(f"#### kWh Consumption (Monthly): {monthly_kwh:.2f} kWh")

        # Display Pie Chart for Appliance Wattage
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
    st.title("WattBuddy")
    st.subheader("Your Electricity Advisor")
    st.write("\n")
    st.header("About WattBuddy")
    st.write("""
        WattBuddy helps users manage their electricity consumption and budget. By entering the cost of electricity and 
        adding appliances, users can calculate the total cost and consumption based on their usage.
        The app also provides suggestions for adjusting appliance usage to stay within the given budget.
    """)

# Suggest Appliances Page
elif st.session_state.page_selection == "Suggest Appliances":
    st.title("Suggest Appliances")
    st.write("This page suggests appliances based on your budget")
    st.write("\n")
    st.write("Data set used for this auto suggestion is from the year 2022")
    st.write("Rating: Php10.4")
