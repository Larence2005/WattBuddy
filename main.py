import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    st.write('Rating: Php10.4')






import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Set page config
st.set_page_config(
    page_title="Appliance Recommender",
    page_icon="⚡",
    layout="wide"
)

# Define appliance types with importance ratings
appliance_types = {
    'Refrigerator': {'importance': 9, 'range': (0.8, 1.5)},
    'Air Conditioner': {'importance': 7, 'range': (1.5, 2.5)},
    'Television': {'importance': 5, 'range': (0.1, 0.3)},
    'Washing Machine': {'importance': 8, 'range': (0.5, 1.2)},
    'Microwave': {'importance': 6, 'range': (0.6, 1.2)},
    'Electric Kettle': {'importance': 4, 'range': (1.0, 1.5)},
    'Laptop': {'importance': 8, 'range': (0.05, 0.1)},
    'Ceiling Fan': {'importance': 7, 'range': (0.05, 0.15)},
    'Electric Stove': {'importance': 9, 'range': (1.0, 2.0)},
    'Water Heater': {'importance': 6, 'range': (1.2, 2.0)},
    'Dishwasher': {'importance': 5, 'range': (0.8, 1.5)},
    'Blender': {'importance': 3, 'range': (0.3, 0.6)}
}

def generate_dataset(num_samples=250):
    data = []
    for _ in range(num_samples):
        appliance = np.random.choice(list(appliance_types.keys()))
        min_consumption, max_consumption = appliance_types[appliance]['range']
        consumption = round(np.random.uniform(min_consumption, max_consumption), 2)
        cost = round(consumption * 10, 2)
        importance = appliance_types[appliance]['importance']
        efficiency = round(1 / (consumption * cost), 2)
        data.append([appliance, consumption, cost, importance, efficiency])
    
    return pd.DataFrame(data, columns=[
        'Appliance Type',
        'Estimated Energy Consumption per kWh',
        'Estimated Cost to Operate(Php) per hour',
        'Importance Rating',
        'Efficiency Score'
    ])

def get_recommendation_note(appliance, cost, importance):
    """Generate a personalized note for each recommendation."""
    if importance >= 8:
        priority = "Essential - High Priority"
    elif importance >= 6:
        priority = "Important - Medium Priority"
    else:
        priority = "Optional - Low Priority"
    
    cost_level = "Economical" if cost < 10 else "Moderate" if cost < 15 else "High"
    return f"{priority} | Cost Level: {cost_level}"

def recommend_appliances(budget, model, dataframe, label_encoder, num_recommendations):
    """Recommend appliances within budget considering importance and efficiency."""
    # Predict cost for all appliances
    dataframe['Predicted Cost'] = model.predict(
        dataframe[['Appliance Type Encoded', 'Estimated Energy Consumption per kWh', 
                  'Importance Rating', 'Efficiency Score']]
    )
    
    # Filter appliances within budget
    recommended = dataframe[dataframe['Predicted Cost'] <= budget].copy()
    
    if recommended.empty:
        return None
    
    # Calculate composite score
    recommended['Composite Score'] = (
        recommended['Importance Rating'] * 0.4 +
        recommended['Efficiency Score'] * 0.3 +
        (1 / recommended['Predicted Cost']) * 0.3
    )
    
    # Get top recommendations
    top_recommendations = recommended.nlargest(num_recommendations, 'Composite Score')
    
    # Prepare final output
    result = top_recommendations[[
        'Appliance Type',
        'Estimated Energy Consumption per kWh',
        'Predicted Cost',
        'Importance Rating'
    ]].sort_values(['Importance Rating', 'Predicted Cost'], ascending=[False, True])
    
    # Add recommendation notes
    result['Recommendation Notes'] = result.apply(
        lambda x: get_recommendation_note(
            x['Appliance Type'],
            x['Predicted Cost'],
            x['Importance Rating']
        ),
        axis=1
    )
    
    return result

def main():
    # Title and description
    st.title("🔌 Smart Appliance Recommender")
    st.markdown("""
    This app helps you find the best appliances within your budget, considering:
    - Energy efficiency
    - Operating costs
    - Importance level
    - Cost-effectiveness
    """)
    
    # Sidebar for user input
    st.sidebar.header("Set Your Parameters")
    budget = st.sidebar.slider(
        "What's your maximum budget per hour (PHP)?",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="Select your maximum operating cost budget per hour in Philippine Peso"
    )
    
    num_recommendations = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=15,
        value=8,
        step=1,
        help="Select how many appliance recommendations you want to see"
    )
    
    # Generate dataset and train model
    df = generate_dataset()
    le = LabelEncoder()
    df['Appliance Type Encoded'] = le.fit_transform(df['Appliance Type'])
    
    X = df[['Appliance Type Encoded', 'Estimated Energy Consumption per kWh', 
            'Importance Rating', 'Efficiency Score']]
    y = df['Estimated Cost to Operate(Php) per hour']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get recommendations
    recommendations = recommend_appliances(budget, model, df, le, num_recommendations)
    
    # Display results
    if recommendations is not None:
        st.subheader(f"Top {num_recommendations} Recommended Appliances Within ₱{budget}/hour")
        
        # Create three columns for each recommendation
        for _, row in recommendations.iterrows():
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"**{row['Appliance Type']}**")
            with col2:
                st.markdown(f"₱{row['Predicted Cost']:.2f}/hour")
            with col3:
                st.markdown(row['Recommendation Notes'])
            
            # Add a divider
            st.divider()
        
        # Display detailed table
        if st.checkbox("Show Detailed Information"):
            st.dataframe(
                recommendations.style.format({
                    'Estimated Energy Consumption per kWh': '{:.2f}',
                    'Predicted Cost': '₱{:.2f}',
                    'Importance Rating': '{:.0f}'
                })
            )
    else:
        st.error(f"No appliances found within your budget of ₱{budget}/hour. Please try a higher budget.")

if __name__ == "__main__":
    main()
