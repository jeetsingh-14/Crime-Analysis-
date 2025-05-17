"""
# National Crime Trends Dashboard

This Streamlit application visualizes crime data across the United States,
providing insights into crime trends at the national, state, city, county, and college levels.

Author: JUNIE
Date: April 2023
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import datetime
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="National Crime Trends Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    """
    Load all datasets needed for the dashboard
    """
    try:
        # Define the base path for data files
        base_path = os.path.join("ALL US Crime data-20250427T235517Z-001/ALL US Crime data/outputs")

        # Check if the directory exists
        if not os.path.exists(base_path):
            st.error(f"Data directory not found: {base_path}")
            # Try alternative path
            base_path = os.path.join("../ALL US Crime data-20250427T235517Z-001/ALL US Crime data/outputs")
            if not os.path.exists(base_path):
                st.error(f"Alternative data directory not found: {base_path}")
                # One more attempt with a different path structure
                base_path = os.path.join("../ALL US Crime data/outputs")
                if not os.path.exists(base_path):
                    raise FileNotFoundError(f"Could not find data directory in any of the expected locations")


        # Load city data with coordinates
        city_data_with_coords_path = os.path.join(base_path, "table8_with_coordinates.csv")
        city_data_with_coords = pd.read_csv(city_data_with_coords_path)

        # Process coordinates - ensure they are numeric and handle missing values
        city_data_with_coords['LATITUDE'] = pd.to_numeric(city_data_with_coords['LATITUDE'], errors='coerce')
        city_data_with_coords['LONGITUDE'] = pd.to_numeric(city_data_with_coords['LONGITUDE'], errors='coerce')

        # Log some debugging information (not displayed to user)
        print(f"Loaded {len(city_data_with_coords)} rows from table8_with_coordinates.csv")
        print(f"Number of rows with valid coordinates: {len(city_data_with_coords.dropna(subset=['LATITUDE', 'LONGITUDE']))}")

        # Load city summary data
        city_summary = pd.read_csv(os.path.join(base_path, "city_summary.csv"))

        # Load county summary data
        county_summary = pd.read_csv(os.path.join(base_path, "county_summary.csv"))

        # Load college summary data
        college_summary = pd.read_csv(os.path.join(base_path, "college_summary.csv"))

        # Load other datasets as needed
        table9_data = pd.read_csv(os.path.join(base_path, "table9_clean_final.csv"))
        table10_data = pd.read_csv(os.path.join(base_path, "table10_clean_final.csv"))

        # Log some debugging information about the data (not displayed to user)
        print(f"City summary data: {len(city_summary)} rows")
        print(f"County summary data: {len(county_summary)} rows")
        print(f"College summary data: {len(college_summary)} rows")

        return {
            "city_data_with_coords": city_data_with_coords,
            "city_summary": city_summary,
            "county_summary": county_summary,
            "college_summary": college_summary,
            "table9_data": table9_data,
            "table10_data": table10_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e

# Load all datasets
try:
    data = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# Custom CSS for styling with dark mode
st.markdown("""
<style>
    /* Dark mode theme */
    body {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Override Streamlit's default background */
    .stApp {
        background-color: #121212;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #90CAF9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #64B5F6;
        margin-bottom: 1rem;
    }

    /* Stat cards */
    .stat-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #333333;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #90CAF9;
    }
    .stat-label {
        font-size: 1rem;
        color: #B0B0B0;
    }

    /* Make sure text is readable */
    p, li, div {
        color: #E0E0E0;
    }

    /* Adjust widget colors */
    .stSelectbox label, .stRadio label {
        color: #E0E0E0 !important;
    }

    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #1E1E1E;
    }

    /* Animation for page transitions */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-header, .sub-header, .stat-card, .stButton, .stMarkdown, .element-container {
        animation: fadeIn 0.5s ease-out;
    }

    /* Staggered animations for elements */
    .stMarkdown {
        animation-delay: 0.1s;
    }

    .stat-card {
        animation-delay: 0.2s;
    }

    .element-container {
        animation-delay: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "LA Analysis", "National City Trends", "County Trends", "College Trends", "Time-Based Trends"]
)

# Filter Options section above Development Team
st.sidebar.markdown("---")
st.sidebar.markdown("## Filter Options")
# Note: The actual filter options are defined within each page condition

# Function to create a stat card
def stat_card(title, value, description=""):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{title}</div>
        <div>{description}</div>
    </div>
    """, unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.markdown('<h1 class="main-header">National Crime Trends Across the United States</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the National Crime Trends Dashboard. This interactive tool allows you to explore crime statistics 
    across the United States at various levels - from national trends down to specific cities, counties, and college campuses.

    Use the navigation menu on the left to explore different views and analyses of crime data.
    """)

    if data_loaded:
        # Calculate some overview statistics
        total_violent_crimes = data["city_summary"]["Violent_Crime_Total"].sum()
        total_property_crimes = data["city_summary"]["Property_Crime_Total"].sum()

        # Find state with highest violent crime rate
        state_crime_rates = data["city_summary"].groupby("State")[["Violent_Crime_Total", "Population"]].sum()
        state_crime_rates["Violent_Crime_Rate"] = (state_crime_rates["Violent_Crime_Total"] / state_crime_rates["Population"]) * 100
        highest_crime_state = state_crime_rates["Violent_Crime_Rate"].idxmax()
        highest_crime_rate = state_crime_rates["Violent_Crime_Rate"].max()

        # Find city with highest violent crime rate (for cities with population > 10000)
        large_cities = data["city_summary"][data["city_summary"]["Population"] > 10000]
        highest_crime_city = large_cities.loc[large_cities["Violent_Crime_Rate"].idxmax()]

        # Display overview statistics
        st.markdown('<h2 class="sub-header">Quick Statistics</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            stat_card("Total Violent Per Crimes", f"{int(total_violent_crimes):,}")

        with col2:
            stat_card("Total Property Per Crimes", f"{int(total_property_crimes):,}")

        with col3:
            stat_card("Highest Crime Rate Per State", f"{highest_crime_state}", f"{highest_crime_rate:.2f}% violent crime rate")

        with col4:
            stat_card("Highest Crime Rate Per City", f"{highest_crime_city['City']}, {highest_crime_city['State']}",
                     f"{highest_crime_city['Violent_Crime_Rate']:.2f}% violent crime rate")

        # Add interpretation for Quick Statistics
        st.markdown("""
        These statistics reveal that while violent crimes are less frequent than property crimes nationally, 
        there are significant regional variations with certain states and cities experiencing notably higher violent crime rates.
        """)

        # Show a sample visualization on the home page
        st.markdown('<h2 class="sub-header">National Overview</h2>', unsafe_allow_html=True)

        # Create a simple bar chart of top 10 states by violent crime
        top_states = state_crime_rates.sort_values("Violent_Crime_Rate", ascending=False).head(10)

        fig = px.bar(
            top_states.reset_index(), 
            x="State", 
            y="Violent_Crime_Rate",
            title="Top 10 States by Violent Crime Rate",
            labels={"Violent_Crime_Rate": "Violent Crime Rate (%)", "State": "State"},
            color="Violent_Crime_Rate",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Top 10 States chart
        st.markdown("""
        The chart reveals a concentration of high violent crime rates in certain regions, with the top states 
        showing rates significantly above the national average. This pattern suggests that crime prevention 
        strategies may need to be tailored to address specific regional factors contributing to these elevated rates.
        """)

    else:
        st.warning("Data could not be loaded. Please check the file paths and try again.")

# LA Crime Analytics Page
elif page == "LA Analysis":
    st.markdown('<h1 class="main-header">Los Angeles Crime Analytics</h1>', unsafe_allow_html=True)


    # Section 1: LA Crime Heatmap
    st.markdown('<h2 class="sub-header">Los Angeles Crime Heatmap</h2>', unsafe_allow_html=True)

    try:
        # Load and display the LA crime heatmap
        with open("outputs_1/crime_heatmap.html", 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600, width=900)

        # Add interpretation for LA Crime Heatmap
        st.markdown("""
        The heatmap identifies several high-concentration crime areas across Los Angeles, particularly in downtown and 
        central regions. These hotspots often correlate with areas of higher population density, commercial activity, 
        and socioeconomic challenges.
        """)
    except Exception as e:
        st.error(f"Error loading LA crime heatmap: {str(e)}")

    # Section 2: Crime Prediction Maps for Specific Areas
    st.markdown('<h2 class="sub-header">Crime Prediction by Area</h2>', unsafe_allow_html=True)

    try:
        # Get all PNG files from outputs_1 folder
        image_files = [file for file in os.listdir('outputs_1') if file.endswith('.png') and file.startswith('crime_prediction_')]

        if len(image_files) > 0:
            # Create a grid layout with 2 columns
            cols = st.columns(2)

            # Display each image with a caption
            for idx, img_file in enumerate(image_files):
                try:
                    # Extract area name from filename
                    area_name = img_file.replace('crime_prediction_', '').replace('.png', '').replace('_', ' ')

                    # Display image in the appropriate column
                    with cols[idx % 2]:
                        st.image(f'outputs_1/{img_file}', caption=area_name, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {img_file}: {str(e)}")
                    continue
        else:
            st.warning("No crime prediction images found in the outputs_1 folder.")
    except Exception as e:
        st.error(f"Error loading crime prediction images: {str(e)}")

    # Add interpretation for Crime Prediction Maps
    st.markdown("""
    These crime prediction maps highlight areas with elevated risk for future criminal activity based on historical patterns. 
    Areas shown in darker colors represent locations where preventative policing and community intervention may be most 
    effective in reducing crime rates.
    """)

    # Section 3: Weapons-Based Crime Analysis
    st.markdown('<h2 class="sub-header">Weapons-Based Crime Patterns in Los Angeles</h2>', unsafe_allow_html=True)

    st.markdown("""
    This visualization shows the distribution of crimes by weapon type across different areas of Los Angeles.
    The map highlights areas with higher concentrations of specific weapon-related crimes.
    """)

    try:
        # Load and display the weapon category map
        with open("outputs_1/weapon_category_map.html", 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600, width=900)

        # Add interpretation for Weapons-Based Crime Map
        st.markdown("""
        The weapons distribution map reveals that certain weapon types are more prevalent in specific neighborhoods. 
        Firearms tend to be more commonly used in higher-crime areas, while other weapons show different geographical 
        patterns, potentially reflecting differences in criminal activity types across the city.
        """)
    except Exception as e:
        st.error(f"Error loading weapon category map: {str(e)}")

    # Section 4: Crime Type Prediction Modeling and Insights
    st.markdown('<h2 class="sub-header">Crime Type Prediction Modeling and Insights</h2>', unsafe_allow_html=True)

    st.markdown("""
    This section presents insights from machine learning models used to classify and predict crime types in Los Angeles.
    The analysis includes model performance metrics, crime distribution by area, and predictive modeling results.
    """)

    # Base Model Performance Summary
    st.markdown('<h3>Base Model Performance Summary</h3>', unsafe_allow_html=True)

    base_model_data = {
        'Crime Type': ['Crimes against Persons', 'Crimes against Public Order', 'Other Crimes', 'Property Crimes'],
        'Precision': [0.78, 0.00, 0.77, 0.85],
        'Recall': [0.96, 0.00, 0.58, 0.88],
        'F1-Score': [0.86, 0.00, 0.66, 0.87],
        'Support': [15342, 3, 19498, 37201]
    }

    base_model_df = pd.DataFrame(base_model_data)
    st.dataframe(base_model_df)

    st.markdown("""
    **Model Performance Interpretation:**
    - The base model achieves an overall accuracy of 82%.
    - "Crimes against Persons" and "Property Crimes" are predicted with high precision and recall.
    - "Other Crimes" have good precision but lower recall, indicating some are misclassified.
    - "Crimes against Public Order" have 0 precision due to extreme class imbalance (only 3 samples).
    """)

    # Tuned Model Performance
    st.markdown('<h3>Tuned Model (GridSearchCV) Performance</h3>', unsafe_allow_html=True)

    st.markdown("""
    **Best Parameters Found:**
    - max_depth: None
    - min_samples_leaf: 2
    - min_samples_split: 10
    - n_estimators: 300
    - Best Cross-Validated Score: 0.604
    """)

    tuned_model_data = {
        'Crime Type': ['Crimes against Persons', 'Crimes against Public Order', 'Other Crimes', 'Property Crimes'],
        'Precision': [0.78, 0.00, 0.80, 0.86],
        'Recall': [0.98, 0.00, 0.58, 0.89],
        'F1-Score': [0.87, 0.00, 0.67, 0.87],
        'Support': [15342, 3, 19498, 37201]
    }

    tuned_model_df = pd.DataFrame(tuned_model_data)
    st.dataframe(tuned_model_df)

    st.markdown("""
    **Tuned Model Improvements:**
    - Overall accuracy improved slightly to 82% with better weighted average metrics.
    - Precision for "Other Crimes" improved from 0.77 to 0.80.
    - Recall for "Crimes against Persons" improved from 0.96 to 0.98.
    - The model still struggles with the rare "Crimes against Public Order" class.
    """)

    # Top 10 Areas with Highest Crime Counts
    st.markdown('<h3>Top 10 Areas with Highest Crime Counts</h3>', unsafe_allow_html=True)

    top_areas_data = {
        'Area Name': ['Central', 'Pacific', 'Southwest', '77th Street', 'N Hollywood', 
                     'Wilshire', 'Hollywood', 'Olympic', 'Newton', 'Rampart'],
        'Total Crime Count': [27184, 21961, 21496, 20735, 18807, 18020, 17601, 17528, 17327, 17132]
    }

    top_areas_df = pd.DataFrame(top_areas_data)
    st.dataframe(top_areas_df)

    st.markdown("""
    Central and Pacific areas have the highest crime counts in Los Angeles, with Central recording over 27,000 crimes.
    The top 10 areas account for a significant portion of all crimes in the city.
    """)

    # Crime Type Distribution in Top Areas
    st.markdown('<h3>Crime Type Distribution in Top Areas</h3>', unsafe_allow_html=True)

    # Create a selectbox for users to choose an area
    selected_area = st.selectbox(
        "Select an area to view crime type breakdown:",
        ['Central', 'Pacific', 'Southwest', '77th Street', 'N Hollywood', 
         'Wilshire', 'Hollywood', 'Olympic', 'Newton', 'Rampart']
    )

    # Define crime type distribution data for each area
    crime_distribution = {
        'Central': {'Property Crimes': 50.29, 'Other Crimes': 28.09, 'Crimes against Persons': 21.61, 'Crimes against Public Order': 0.01},
        'Pacific': {'Property Crimes': 62.67, 'Other Crimes': 24.27, 'Crimes against Persons': 13.05, 'Crimes against Public Order': 0.0},
        'Southwest': {'Property Crimes': 42.81, 'Other Crimes': 34.05, 'Crimes against Persons': 23.13, 'Crimes against Public Order': 0.01},
        '77th Street': {'Property Crimes': 35.43, 'Other Crimes': 31.06, 'Crimes against Persons': 33.51, 'Crimes against Public Order': 0.0},
        'N Hollywood': {'Property Crimes': 55.85, 'Other Crimes': 26.46, 'Crimes against Persons': 17.69, 'Crimes against Public Order': 0.0},
        'Wilshire': {'Property Crimes': 63.08, 'Other Crimes': 21.63, 'Crimes against Persons': 15.29, 'Crimes against Public Order': 0.01},
        'Hollywood': {'Property Crimes': 53.44, 'Other Crimes': 24.17, 'Crimes against Persons': 22.37, 'Crimes against Public Order': 0.02},
        'Olympic': {'Property Crimes': 50.09, 'Other Crimes': 26.79, 'Crimes against Persons': 23.13, 'Crimes against Public Order': 0.0},
        'Newton': {'Property Crimes': 42.51, 'Other Crimes': 31.32, 'Crimes against Persons': 26.17, 'Crimes against Public Order': 0.0},
        'Rampart': {'Property Crimes': 45.73, 'Other Crimes': 28.92, 'Crimes against Persons': 25.36, 'Crimes against Public Order': 0.0}
    }

    # Create a dataframe for the selected area
    selected_data = pd.DataFrame({
        'Crime Type': list(crime_distribution[selected_area].keys()),
        'Percentage': list(crime_distribution[selected_area].values())
    })

    # Create a bar chart for the selected area
    fig = px.bar(
        selected_data, 
        x='Crime Type', 
        y='Percentage',
        title=f'Crime Type Distribution in {selected_area}',
        color='Percentage',
        color_continuous_scale='Reds',
        text='Percentage'
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Crime Distribution Insights:**
    - Property Crimes dominate in most areas, particularly in Wilshire (63%) and Pacific (63%).
    - 77th Street has a more balanced distribution with a higher percentage of Crimes against Persons.
    - Crimes against Public Order are extremely rare across all areas.
    """)

    # Prediction Models for Each Area
    st.markdown('<h3>Prediction Models for Each Area</h3>', unsafe_allow_html=True)

    # Create a dataframe with model performance metrics
    model_performance_data = {
        'Area': ['Central', 'Central', 'Central', 'Pacific', 'Pacific', 'Pacific', 
                'Southwest', 'Southwest', 'Southwest', '77th Street', '77th Street', '77th Street',
                'N Hollywood', 'N Hollywood', 'N Hollywood', 'Wilshire', 'Wilshire', 'Wilshire',
                'Hollywood', 'Hollywood', 'Hollywood', 'Olympic', 'Olympic', 'Olympic',
                'Newton', 'Newton', 'Newton', 'Rampart', 'Rampart', 'Rampart'],
        'Crime Type': ['Property Crimes', 'Other Crimes', 'Crimes against Persons', 
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Crimes against Persons', 'Other Crimes',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons',
                      'Property Crimes', 'Other Crimes', 'Crimes against Persons'],
        'R² Score': [0.71, 0.40, 0.36, 0.32, 0.56, 0.06, -0.26, 0.69, 0.43, -7.92, 0.71, 0.65,
                    0.83, 0.83, 0.85, 0.81, -0.04, 0.89, 0.50, -0.01, 0.68, 0.37, 0.07, 0.42,
                    -0.55, 0.16, -1.73, 0.82, 0.79, 0.54],
        'MSE': [15359.99, 2924.21, 454.35, 11852.48, 1057.77, 290.18, 7158.52, 2658.33, 1007.03,
               3385.33, 1193.53, 1547.80, 3307.80, 825.05, 140.73, 7323.19, 1293.54, 120.89,
               3256.47, 1103.98, 249.18, 2787.68, 1140.90, 744.80, 3941.58, 972.68, 1770.44,
               1772.15, 637.67, 467.87]
    }

    model_performance_df = pd.DataFrame(model_performance_data)

    # Add a color column based on R² score
    def color_r2(val):
        if val >= 0.7:
            return 'good'
        elif val >= 0.4:
            return 'moderate'
        elif val >= 0:
            return 'poor'
        else:
            return 'negative'

    model_performance_df['Performance'] = model_performance_df['R² Score'].apply(color_r2)

    # Display the dataframe with styling
    st.dataframe(model_performance_df[['Area', 'Crime Type', 'R² Score', 'MSE']])

    st.markdown("""
    **Predictive Model Insights:**
    - Strong predictive models (R² > 0.7) exist for N Hollywood and Wilshire areas.
    - Some models show negative R² scores (e.g., Newton, 77th Street), indicating poor predictive power.
    - Property Crimes are generally easier to predict in most areas.
    - Models for Crimes against Persons vary widely in performance across different areas.
    """)

    # Downloadable CSV Reports
    st.markdown('<h3>Downloadable CSV Reports</h3>', unsafe_allow_html=True)

    try:
        # Load the CSV files
        location_crime_type_analysis = pd.read_csv("outputs_1/location_crime_type_analysis.csv")

        # Create a summary dataframe for crime prediction
        crime_prediction_summary = model_performance_df[['Area', 'Crime Type', 'R² Score', 'MSE']]
        crime_prediction_summary.to_csv("outputs_1/crime_prediction_summary.csv", index=False)

        # Create download buttons
        col1, col2 = st.columns(2)

        with col1:
            csv_location = location_crime_type_analysis.to_csv(index=False)
            st.download_button(
                label="Download Location Crime Type Analysis",
                data=csv_location,
                file_name="location_crime_type_analysis.csv",
                mime="text/csv",
            )

        with col2:
            csv_prediction = crime_prediction_summary.to_csv(index=False)
            st.download_button(
                label="Download Crime Prediction Summary",
                data=csv_prediction,
                file_name="crime_prediction_summary.csv",
                mime="text/csv",
            )

        st.markdown("""
        Download these reports for further analysis of crime patterns and predictive model performance across Los Angeles.
        """)
    except Exception as e:
        st.error(f"Error preparing downloadable reports: {str(e)}")


# City-Level Trends Page
elif page == "National City Trends":
    st.markdown('<h1 class="main-header">City-Level Crime Trends</h1>', unsafe_allow_html=True)

    if data_loaded:
        # Filter options
        st.sidebar.markdown("## Filter Options")

        # Filter by state
        states = sorted(data["city_summary"]["State"].unique())
        selected_state = st.sidebar.selectbox("Select State", ["All States"] + states)

        # Log information (not displayed to user)
        print(f"City summary data - total rows: {len(data['city_summary'])}")

        # Filter data based on selections
        filtered_data = data["city_summary"].copy()
        print(f"Total rows in filtered data: {len(filtered_data)}")

        # Apply state filter if not "All States"
        if selected_state != "All States":
            filtered_data = filtered_data[filtered_data["State"] == selected_state]
            print(f"Rows for state {selected_state}: {len(filtered_data)}")

        # Check for missing values in key columns
        missing_violent = filtered_data["Violent_Crime_Rate"].isna().sum()
        missing_property = filtered_data["Property_Crime_Rate"].isna().sum()
        print(f"Rows with missing Violent Crime Rate: {missing_violent}")
        print(f"Rows with missing Property Crime Rate: {missing_property}")

        # Add a status message for the user
        if len(filtered_data) == 0:
            st.warning(f"No data available" + (f" for {selected_state}" if selected_state != "All States" else "") + ". Please try different filters.")

        # Display visualizations
        st.markdown('<h2 class="sub-header">Top Dangerous Cities</h2>', unsafe_allow_html=True)

        # Add a spinner while generating the charts
        with st.spinner("Generating violent crime chart..."):
            # Filter out rows with missing violent crime rate
            violent_data = filtered_data.dropna(subset=["Violent_Crime_Rate"]).copy()
            print(f"Rows with valid Violent Crime Rate: {len(violent_data)}")

            # Ensure we have data to display
            if len(violent_data) > 0:
                # Top 10 cities by violent crime rate
                top_violent_cities = violent_data.sort_values("Violent_Crime_Rate", ascending=False).head(10)
                print(f"Top violent cities found: {len(top_violent_cities)}")

                if len(top_violent_cities) > 0:
                    fig = px.bar(
                        top_violent_cities, 
                        x="City", 
                        y="Violent_Crime_Rate",
                        title="Top 10 Cities by Violent Crime Rate",
                        labels={"Violent_Crime_Rate": "Violent Crime Rate", "City": "City"},
                        color="Violent_Crime_Rate",
                        color_continuous_scale="Reds",
                        hover_data=["State", "Population"]
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation for Top 10 Dangerous Cities
                    st.markdown("""
                    The chart highlights cities with persistently high violent crime rates, often characterized by urban density, 
                    socioeconomic disparities, and limited resources for law enforcement. Many of these cities face complex 
                    challenges that require comprehensive approaches beyond traditional policing.
                    """)
                else:
                    st.warning("No cities found with valid violent crime rate data.")
            else:
                st.warning("No data available for violent crime rate visualization.")

        # Add a spinner while generating the charts
        with st.spinner("Generating property crime chart..."):
            # Filter out rows with missing property crime rate
            property_data = filtered_data.dropna(subset=["Property_Crime_Rate"]).copy()
            print(f"Rows with valid Property Crime Rate: {len(property_data)}")

            # Ensure we have data to display
            if len(property_data) > 0:
                # Top 10 cities by property crime rate
                top_property_cities = property_data.sort_values("Property_Crime_Rate", ascending=False).head(10)
                print(f"Top property crime cities found: {len(top_property_cities)}")

                if len(top_property_cities) > 0:
                    fig = px.bar(
                        top_property_cities, 
                        x="City", 
                        y="Property_Crime_Rate",
                        title="Top 10 Cities by Property Crime Rate",
                        labels={"Property_Crime_Rate": "Property Crime Rate", "City": "City"},
                        color="Property_Crime_Rate",
                        color_continuous_scale="Blues",
                        hover_data=["State", "Population"]
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No cities found with valid property crime rate data.")
            else:
                st.warning("No data available for property crime rate visualization.")

    else:
        st.warning("Data could not be loaded. Please check the file paths and try again.")

# County-Level Trends Page
elif page == "County Trends":
    st.markdown('<h1 class="main-header">County-Level Crime Trends</h1>', unsafe_allow_html=True)

    if data_loaded:
        # Filter options
        st.sidebar.markdown("## Filter Options")

        # Filter by state
        states = sorted(data["county_summary"]["State"].str.strip().unique())
        selected_state = st.sidebar.selectbox("Select State", ["All States"] + states)

        # Filter by year
        years = sorted(data["county_summary"]["Year"].unique())
        selected_year = st.sidebar.selectbox("Select Year", years)

        # Filter data based on selections
        filtered_data = data["county_summary"][data["county_summary"]["Year"] == selected_year].copy()

        # Apply state filter if not "All States"
        if selected_state != "All States":
            filtered_data = filtered_data[filtered_data["State"].str.strip() == selected_state]

        # Display visualizations
        st.markdown('<h2 class="sub-header">County Crime Analysis</h2>', unsafe_allow_html=True)

        # Top 10 counties by violent crime
        top_violent_counties = filtered_data.sort_values("Violent_Crime", ascending=False).head(10)

        fig = px.bar(
            top_violent_counties, 
            x="County", 
            y="Violent_Crime",
            title=f"Top 10 Counties by Violent Crime ({selected_year})",
            labels={"Violent_Crime": "Violent Crime Count", "County": "County"},
            color="Violent_Crime",
            color_continuous_scale="Reds",
            hover_data=["State"]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Top 10 Counties by Violent Crime
        st.markdown("""
        The counties with highest violent crime counts typically contain major metropolitan areas or have large populations. 
        These counties often face unique challenges in crime prevention due to their size, demographic diversity, and 
        complex urban-suburban dynamics.
        """)

        # Top 10 counties by property crime
        top_property_counties = filtered_data.sort_values("Property_Crime", ascending=False).head(10)

        fig = px.bar(
            top_property_counties, 
            x="County", 
            y="Property_Crime",
            title=f"Top 10 Counties by Property Crime ({selected_year})",
            labels={"Property_Crime": "Property Crime Count", "County": "County"},
            color="Property_Crime",
            color_continuous_scale="Blues",
            hover_data=["State"]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Top 10 Counties by Property Crime
        st.markdown("""
        Property crimes show different patterns than violent crimes, with some counties ranking high in both categories 
        while others appear only in property crime rankings. Economic factors, tourism, and retail density often 
        contribute to higher property crime rates in these counties.
        """)

    else:
        st.warning("Data could not be loaded. Please check the file paths and try again.")

# College Campus Safety Page
elif page == "College Trends":
    st.markdown('<h1 class="main-header">College Campus Safety</h1>', unsafe_allow_html=True)

    if data_loaded:
        # Filter options
        st.sidebar.markdown("## Filter Options")

        # Filter by state
        states = sorted(data["college_summary"]["State"].unique())
        selected_state = st.sidebar.selectbox("Select State", ["All States"] + states)

        # Filter by year
        years = sorted(data["college_summary"]["Year"].unique())
        selected_year = st.sidebar.selectbox("Select Year", years)

        # Filter data based on selections
        filtered_data = data["college_summary"][data["college_summary"]["Year"] == selected_year].copy()

        # Apply state filter if not "All States"
        if selected_state != "All States":
            filtered_data = filtered_data[filtered_data["State"] == selected_state]

        # Display visualizations
        st.markdown('<h2 class="sub-header">College Campus Crime Analysis</h2>', unsafe_allow_html=True)

        # Top 10 colleges by violent crime rate
        top_violent_colleges = filtered_data.sort_values("Violent_Crime_Rate", ascending=False).head(10)

        fig = px.bar(
            top_violent_colleges, 
            x="University/College", 
            y="Violent_Crime_Rate",
            title=f"Top 10 Colleges by Violent Crime Rate ({selected_year})",
            labels={"Violent_Crime_Rate": "Violent Crime Rate", "University/College": "College/University"},
            color="Violent_Crime_Rate",
            color_continuous_scale="Reds",
            hover_data=["State", "Student_Enrollment"]
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Top 10 Colleges by Violent Crime Rate
        st.markdown("""
        Colleges with higher violent crime rates are often located in or near urban areas with higher overall crime. 
        Campus size, surrounding neighborhood characteristics, and the presence of robust security measures can all 
        significantly influence campus safety statistics.
        """)

        # Top 10 colleges by property crime rate
        top_property_colleges = filtered_data.sort_values("Property_Crime_Rate", ascending=False).head(10)

        fig = px.bar(
            top_property_colleges, 
            x="University/College", 
            y="Property_Crime_Rate",
            title=f"Top 10 Colleges by Property Crime Rate ({selected_year})",
            labels={"Property_Crime_Rate": "Property Crime Rate", "University/College": "College/University"},
            color="Property_Crime_Rate",
            color_continuous_scale="Blues",
            hover_data=["State", "Student_Enrollment"]
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Top 10 Colleges by Property Crime Rate
        st.markdown("""
        Property crimes on college campuses often involve theft of electronics, bicycles, and other personal items. 
        Institutions with higher property crime rates may benefit from enhanced security measures like surveillance 
        systems, improved lighting, and student education programs about protecting personal property.
        """)

    else:
        st.warning("Data could not be loaded. Please check the file paths and try again.")

# Time-Based Trends Page
elif page == "Time-Based Trends":
    st.markdown('<h1 class="main-header">Time-Based Crime Trends</h1>', unsafe_allow_html=True)

    if data_loaded:
        # Filter options
        st.sidebar.markdown("## Filter Options")

        # Filter by state
        states = sorted(data["city_summary"]["State"].unique())
        selected_state = st.sidebar.selectbox("Select State", ["All States"] + states)

        # Filter by crime type
        crime_type = st.sidebar.radio(
            "Select Crime Type",
            ["Violent Crime", "Property Crime"]
        )

        # Filter data based on selections
        filtered_data = data["city_summary"].copy()

        # Apply state filter if not "All States"
        if selected_state != "All States":
            filtered_data = filtered_data[filtered_data["State"] == selected_state]

        # Select the appropriate rate column based on crime type
        rate_column = "Violent_Crime_Rate" if crime_type == "Violent Crime" else "Property_Crime_Rate"

        # Display visualizations
        st.markdown('<h2 class="sub-header">Crime Trends Over Time</h2>', unsafe_allow_html=True)

        # Aggregate data by year
        yearly_data = filtered_data.groupby("Year")[rate_column].mean().reset_index()

        fig = px.line(
            yearly_data, 
            x="Year", 
            y=rate_column,
            title=f"Average {crime_type} Rate Over Time" + (f" in {selected_state}" if selected_state != "All States" else ""),
            labels={rate_column: f"Average {crime_type} Rate", "Year": "Year"},
            markers=True,
            line_shape="linear"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation for Crime Trends Over Time
        st.markdown(f"""
        The trend line shows how {crime_type.lower()} rates have changed over time
        {f" in {selected_state}" if selected_state != "All States" else " nationally"}. 
        Understanding these patterns helps identify whether crime prevention strategies have been effective and 
        where additional resources might be needed to address persistent or emerging crime issues.
        """)

        # Show top 5 cities with highest crime rates for each year
        st.markdown('<h2 class="sub-header">Top Cities by Year</h2>', unsafe_allow_html=True)

        years = sorted(filtered_data["Year"].unique())
        selected_year_for_top = st.selectbox("Select Year for Top Cities", years)

        year_data = filtered_data[filtered_data["Year"] == selected_year_for_top]
        top_cities = year_data.sort_values(rate_column, ascending=False).head(5)

        fig = px.bar(
            top_cities, 
            x="City", 
            y=rate_column,
            title=f"Top 5 Cities by {crime_type} Rate in {selected_year_for_top}" + (f" ({selected_state})" if selected_state != "All States" else ""),
            labels={rate_column: f"{crime_type} Rate", "City": "City"},
            color=rate_column,
            color_continuous_scale="Reds" if crime_type == "Violent Crime" else "Blues",
            hover_data=["State", "Population"]
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Data could not be loaded. Please check the file paths and try again.")

# Developers list in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Development Team")
developers_data = """
 **Jeet Singh Saini**
 -Role: Lead Data Scientist & Project Manager


 **Sylvano D'souza**
 -Role: Data Analyst – National Trends

 **Ashwinth Reddy Kondapalli**
-Role: ML Engineer – LA Crime Predictions

 **Vishal CV**
-Role: Dashboard & Visualization Specialist
 """
st.sidebar.markdown(developers_data)
