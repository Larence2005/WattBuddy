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
if st.sidebar.button("Home"):
    page = "Home"
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
            classification = "good"
        elif monthly_cost <= budget:
            st.warning("Your monthly electric cost is BALANCED!")
            classification = "balanced"
        else:
            st.error("Your monthly electric cost is HIGH!")
            classification = "high"

        # Train a linear regression model for suggestions
        st.write('\n')

        # Prepare data for linear regression
        X = df["Hours Used"].values.reshape(-1, 1)  # Feature: Hours Used
        y = df["Cost (Php)"].values.reshape(-1, 1)  # Target: Cost
        model = LinearRegression()
        model.fit(X, y)


        # Wattage percentage graph
        st.write('\n')
        st.subheader("Wattage Percentage Graph")
        fig, ax = plt.subplots()
        wattage_percentages = (df["Wattage (W)"] / df["Wattage (W)"].sum()) * 100
        ax.pie(wattage_percentages, labels=df["Name"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Predict suggested hours and calculate money saved
        st.write('\n')
        st.write('\n')
        df["Hours Suggested"] = 0  # Default is zero (no suggestion)
        money_saved_texts = []

        # Calculate total cost and total usage for percentage saving calculation
        total_cost = df["Cost (Php)"].sum()
        monthly_cost = total_cost * 30  # Assuming 30 days in a month

        # Calculate total monthly loss
        total_monthly_loss = max(monthly_cost - budget, 0)
        
        # Loop over each appliance to calculate suggestions and money saved
        for idx, row in df.iterrows():
            if classification == "high":
                # Calculate how much this appliance contributes to the excess cost
                appliance_monthly_cost = row["Cost (Php)"] * 30
                excess_cost = max(monthly_cost - budget, 0)

                if excess_cost > 0:
                    # Calculate the appliance's contribution to the excess cost
                    appliance_excess_ratio = appliance_monthly_cost / monthly_cost
                    percentage_lost = appliance_excess_ratio * 100
                else:
                    percentage_lost = 0

                # Calculate suggested hours (you can keep your existing logic)
                # Calculate suggested hours
                suggested_hours = (budget / monthly_cost) * row["Cost (Php)"] / (row["Wattage (W)"] * price_per_kwh / 1000)
                suggested_hours = max(suggested_hours, 0)
                df.at[idx, "Hours Suggested"] = suggested_hours

                # Add suggestion text for HIGH classification
                money_saved_texts.append(
                    f"\n{row['Name']}: Reduce usage! Potential loss: {percentage_lost:.2f}%"
                )
            else:
                # For BALANCED or GOOD classifications
                if monthly_cost <= budget * 0.7:
                    # Calculate money saved percentage based on individual appliance's cost ratio
                    appliance_cost_ratio = (row["Cost (Php)"] * 30) / monthly_cost
                    money_saved_percentage = appliance_cost_ratio * 100

                    money_saved_texts.append(
                        f"\n{row['Name']}: Money saved: {money_saved_percentage:.2f}%"
                    )
                elif monthly_cost <= budget:
                    # Calculate money saved percentage based on individual appliance's cost ratio
                    appliance_cost_ratio = (row["Cost (Php)"] * 30) / monthly_cost
                    money_saved_percentage = appliance_cost_ratio * 100

                    money_saved_texts.append(
                        f"{row['Name']}: Money saved: {money_saved_percentage:.2f}%"
                    )

        # Display the final results with the money saved or loss texts
        st.write("\n### Money Saved or Loss:")
        st.write(f"**Total Monthly Loss: Php {total_monthly_loss:.2f}**")
        for text in money_saved_texts:
            st.write(text)

        st.write('\n')
        st.write('\n')
        st.subheader("Suggestions")
        st.write("Based on your budget, here are usage suggestions:")
        st.dataframe(df[["Name", "Hours Used", "Hours Suggested"]])


    else:
        st.info("Add appliances to calculate and analyze.")

# About page content
elif page == "About":
    st.header("About WattBuddy")
    st.write("""
        WattBuddy helps users manage their electricity consumption and budget. By entering the cost of electricity and 
        adding appliances, users can calculate the total cost and consumption based on their usage.
        The app also provides suggestions for adjusting appliance usage to stay within the given budget.
    """)
