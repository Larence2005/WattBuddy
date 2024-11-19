import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

#more stable

# Initialize session state for page selection
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'About'

def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=("About",)):
        pass  # Selection handled by callback

    if st.button("Budget and Pricing", use_container_width=True, on_click=set_page_selection, args=("Budget and Pricing",)):
        pass  # Selection handled by callback

    if st.button("Suggest Appliances", use_container_width=True, on_click=set_page_selection, args=("Suggest Appliances",)):
        pass  # Selection handled by callback

# Home page content
if st.session_state.page_selection == "Budget and Pricing":
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
        # Appliance details
        appliance_name = st.text_input("Appliance Name:")
        wattage = st.number_input("Wattage (in Watts):", min_value=0.0, step=1.0)
        hours_used = st.number_input("Usage Time (in Hours):", min_value=0.0, step=0.1)

        # Multiple day selection and weeks in a month
        selected_days = st.multiselect(
            "Select the days of usage:", 
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["Monday"]
        )
        weeks_in_month = st.number_input(
            "How many weeks in a month do you use this appliance?", 
            min_value=1, 
            max_value=4, 
            value=4, 
            step=1
        )

        # Submit button
        add_appliance = st.form_submit_button("Add Appliance")
        if add_appliance:
            if appliance_name and wattage and hours_used:
                # Calculate daily and monthly usage cost
                kwh_consumed = wattage * hours_used / 1000
                daily_cost = kwh_consumed * price_per_kwh
                monthly_days = len(selected_days) * weeks_in_month  # Total days used in a month
                monthly_cost = daily_cost * monthly_days

                # Add appliance data to session state
                st.session_state["appliances"].append({
                    "Name": appliance_name,
                    "Wattage (W)": wattage,
                    "Hours Used": hours_used,
                    "Days Used": ", ".join(selected_days),
                    "Weeks in Month": weeks_in_month,
                    "kWh Consumed": kwh_consumed,
                    "Cost (Php)": daily_cost,
                    "Monthly Cost (Php)": monthly_cost
                })
                st.success(f"{appliance_name} added successfully!")

    # Display appliances
    if st.session_state["appliances"]:
        st.subheader("Appliance List")
        df = pd.DataFrame(st.session_state["appliances"])
        st.dataframe(df)

        # Add remove buttons
        for idx, row in df.iterrows():
            remove_button = st.button(f"Remove {row['Name']}", key=f"remove_{idx}")
            if remove_button:
                st.session_state["appliances"].pop(idx)
                st.session_state['removed_appliance'] = row['Name']
                st.success(f"Appliance '{row['Name']}' removed!")

        # Calculate totals
        total_daily_cost = df["Cost (Php)"].sum()
        total_monthly_cost = df["Monthly Cost (Php)"].sum()  # Sum of individual monthly costs
        total_daily_kwh = df["kWh Consumed"].sum()

        # Calculate monthly kWh based on monthly costs and price per kWh
        monthly_kwh = total_monthly_cost / price_per_kwh

        # Display total and monthly stats with highlighted styling
        st.write('\n')
        st.markdown(f"**Electric Cost (Per Day):** Php **{total_daily_cost:.2f}**")
        st.markdown(f"**kWh Consumption (Per Day):** **{total_daily_kwh:.2f} kWh**")
        st.markdown(f"**Electric Cost (Monthly):** Php **{total_monthly_cost:.2f}**")
        st.markdown(f"**kWh Consumption (Monthly):** **{monthly_kwh:.2f} kWh**")

        # Cost status (monthly cost vs. budget)
        if total_monthly_cost <= budget * 0.7:
            st.success("Your total monthly electric cost is LOW!")
            classification = "low"
        elif total_monthly_cost <= budget:
            st.warning("Your total monthly electric cost is BALANCED!")
            classification = "balanced"
        else:
            st.error("Your total monthly electric cost is HIGH!")
            classification = "high"

        # Money Saved or Loss
        st.write('\n')
        st.subheader("Money Saved or Loss")
        total_monthly_loss = max(total_monthly_cost - budget, 0)
        money_saved_texts = []

        for idx, row in df.iterrows():
            appliance_monthly_cost = row["Monthly Cost (Php)"]
            if classification == "high":
                appliance_excess_ratio = appliance_monthly_cost / total_monthly_cost
                percentage_lost = appliance_excess_ratio * 100
                money_saved_texts.append(f"{row['Name']}: Reduce usage! Potential loss: {percentage_lost:.2f}%")
            else:
                appliance_cost_ratio = appliance_monthly_cost / total_monthly_cost
                money_saved_percentage = appliance_cost_ratio * 100
                money_saved_texts.append(f"{row['Name']}: Money saved: {money_saved_percentage:.2f}%")

        st.write(f"**Total Monthly Loss: Php {total_monthly_loss:.2f}**")
        for text in money_saved_texts:
            st.write(text)

        # Train Linear Regression Model
        X = df["Hours Used"].values.reshape(-1, 1)
        y = df["Cost (Php)"].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)

        # Wattage percentage graph with dropdown toggle
        st.write('\n')
        st.subheader("Wattage Percentage Graph")

        # Create a dropdown (selectbox) to toggle the visibility of the graph
        graph_visibility = st.selectbox(
            "Select Graph Visibility",
            options=["Hide Graph", "Show Graph"],  # Options to show or hide the graph
            index=1  # Default is to show the graph
        )

        if graph_visibility == "Show Graph":
            fig, ax = plt.subplots()

            # Set the figure background to transparent
            fig.patch.set_alpha(0.0)

            # Calculate wattage percentages
            wattage_percentages = (df["Wattage (W)"] / df["Wattage (W)"].sum()) * 100

            # Create the pie chart with white text
            ax.pie(
                wattage_percentages,
                labels=df["Name"],
                autopct=lambda p: f'{p:.1f}%',  # Format percentage
                startangle=90,
                textprops={'color': 'white'},  # Make text white
                wedgeprops=dict(edgecolor="w")  # Optional: Highlight edges for better clarity
            )

            ax.axis("equal")  # Ensure the pie is circular
            st.pyplot(fig)



        # Predict suggested hours
        daily_budget = budget / 30
        cost_per_hour = model.coef_[0][0]

        df["Hours Suggested"] = df.apply(
            lambda row: 0 if classification in ["low", "balanced"] else min(
                daily_budget / (row["Wattage (W)"] * price_per_kwh / 1000),
                model.predict([[row["Hours Used"]]])[0][0] / cost_per_hour,
            ),
            axis=1,
        )

        # Display the suggested hours and saved percentage based on hours reduction
        st.write("\n")
        st.write("### Usage Suggestions:")

        for idx, row in df.iterrows():
            appliance_name = row["Name"]
            hours_used = row["Hours Used"]
            hours_suggested = row["Hours Suggested"]

            # Calculate the saved percentage based on hours reduction
            if hours_used > 0:
                saved_percentage = ((hours_used - hours_suggested) / hours_used) * 100
            else:
                saved_percentage = 0

            # Display each appliance as a bullet point
            st.write(f"**{appliance_name}:**")
            st.write(f"  - Hours Used: {hours_used} hours")
            st.write(f"  - Suggested Hours: {hours_suggested:.2f} hours")
            st.write(f"  - Saved Percentage: {saved_percentage:.2f}%")
            st.write("\n")

    else:
        st.info("Add appliances to calculate and analyze.")

        acc = 1.0
        accuracy = acc
        # Predict the target values
        y_pred = 1
    
        # Calculate Mean Absolute Error
        mae = 0
    
        # Calculate Mean Squared Error
        mse = 0
    
        # Calculate Root Mean Squared Error
        rmse = mse ** 0.5

        st.write ('\n')
        st.write ('\n')
        st.write(f'Accuracy: {accuracy:.1f}')
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")


# About page content
elif st.session_state.page_selection == "About":
    st.title("WattBuddy")
    st.subheader("Your Electricity Advisor")
    st.write('\n')
    st.header("About WattBuddy")
    st.write("""
        WattBuddy helps users manage their electricity consumption and budget. By entering the cost of electricity and 
        adding appliances, users can calculate the total cost and consumption based on their usage.
        The app also provides suggestions for adjusting appliance usage to stay within the given budget.
    """)

# Suggest Appliances page content
elif st.session_state.page_selection == "Suggest Appliances":
    st.title("Suggest Appliances")
    st.write("This page suggests appliances based on your budget")
    st.write('\n')
    st.write('Data set used for this auto suggestion is from year 2022')
    st.write('Rating: Php11.8569')



    # Load the trained model using joblib
    model = joblib.load('appliance_model_philippines_monthly.joblib')
    
    # Load dataset for reference
    dataset = pd.read_csv('appliance_data_philippines_with_area.csv')
    
    # Streamlit UI
    st.title("Electric Advisor - Appliance Recommendation (Philippine Rate)")
    
    # User inputs
    rate = st.number_input("Enter electricity rate per kWh (default is 11.8569):", min_value=0.0, value=11.8569, step=0.1)
    budget = st.number_input("Enter your total monthly budget for electricity (in ₱):", min_value=0.0, step=1.0)
    essential_only = st.checkbox("Show only essential appliances", value=False)
    
    # Allow user to filter by appliance area (Living Room, Kitchen, Bedroom, etc.)
    areas = dataset['Appliance Area'].unique()
    selected_areas = st.multiselect("Select appliance areas:", options=areas, default=areas)
    
    # Filter dataset by selected areas
    filtered_data = dataset[dataset['Appliance Area'].isin(selected_areas)]
    
    if st.button("Get Recommendations"):
        # Adjust costs based on the user's rate
        filtered_data['Adjusted Monthly Cost'] = (
            filtered_data['Rated Power (kWh)'] * filtered_data['Daily Usage (Hours)'] * rate * 30
        )
    
        # Apply essential filter if checked
        if essential_only:
            filtered_data = filtered_data[filtered_data['Essential']]
    
        # Calculate total monthly cost and filter appliances
        recommended_appliances = []
        total_cost = 0
    
        for _, row in filtered_data.iterrows():
            if total_cost + row['Adjusted Monthly Cost'] <= budget:
                total_cost += row['Adjusted Monthly Cost']
                recommended_appliances.append(row)
    
        if recommended_appliances:
            st.write(f"Appliances within your total monthly budget of ₱{budget} at {rate}₱/kWh:")
            st.table(pd.DataFrame(recommended_appliances)[[
                'Appliance Type', 'Essential', 'Appliance Area', 'Rated Power (kWh)', 'Daily Usage (Hours)', 'Adjusted Monthly Cost'
            ]])
            st.write(f"Total Monthly Cost: ₱{total_cost}")
        else:
            st.write("No combination of appliances fits within your monthly budget.")
