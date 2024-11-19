import streamlit as st
import pandas as pd
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

# Home page content
if st.session_state.page_selection == "Home":
    # Budget and Pricing Input Section
    st.subheader("Budget and Pricing Input")
    budget = st.number_input("Enter your budget in Php:", min_value=0.0, value=1000.0, step=50.0)
    price_per_kwh = st.number_input("Enter Meralco's price per kWh in Php:", min_value=0.0, value=10.0, step=0.1)
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
        st.write(f"### kWh Consumption (Per Day): {total_kwh:.2f} kWh")
        st.write(f"### Electric Cost (Monthly): Php {monthly_cost:.2f}")
        st.write(f"### kWh Consumption (Monthly): {monthly_kwh:.2f} kWh")

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

        # Money Saved or Loss
        st.write('\n')
        st.subheader("Money Saved or Loss")
        total_monthly_loss = max(monthly_cost - budget, 0)
        money_saved_texts = []

        for idx, row in df.iterrows():
            appliance_monthly_cost = row["Cost (Php)"] * 30
            if classification == "high":
                appliance_excess_ratio = appliance_monthly_cost / monthly_cost
                percentage_lost = appliance_excess_ratio * 100
                money_saved_texts.append(f"{row['Name']}: Reduce usage! Potential loss: {percentage_lost:.2f}%")
            else:
                appliance_cost_ratio = appliance_monthly_cost / monthly_cost
                money_saved_percentage = appliance_cost_ratio * 100
                money_saved_texts.append(f"{row['Name']}: Money saved: {money_saved_percentage:.2f}%")

        st.write(f"**Total Monthly Loss: Php {total_monthly_loss:.2f}**")
        for text in money_saved_texts:
            st.write(text)

        # Train Linear Regression Model
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

                # Predict suggested hours based on budget
        df["Hours Suggested"] = df.apply(
            lambda row: max(
                (budget / monthly_cost) * row["Cost (Php)"] / (row["Wattage (W)"] * price_per_kwh / 1000),
                0,
            )
            if classification == "high"
            else row["Hours Used"],
            axis=1,
        )

        st.write("### Suggested Appliance Usage")
        st.dataframe(df[["Name", "Hours Used", "Hours Suggested"]])
        

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
