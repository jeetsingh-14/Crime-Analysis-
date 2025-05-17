"""
Los Angeles Crime Rate Heatmap Visualization
This script creates advanced heatmap visualizations for crime data in Los Angeles.
It focuses on creating visually appealing, professional-quality heatmaps showing crime rates by area.
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure output directory exists
os.makedirs("outputs_1", exist_ok=True)

# Load your feature engineered dataset
file_path = r"Data/processed/feature_engineered_data.csv"
try:
    crime_data = pd.read_csv(file_path)
    print("✅ Data Loaded: ", crime_data.shape)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Function to safely get coordinates, handling missing or invalid values
def safe_coordinates(df):
    """Filter out rows with missing or invalid coordinates."""
    return df.dropna(subset=['Latitude', 'Longitude']).copy()

# Function to create a custom color gradient for heatmaps
def get_custom_gradient():
    """Returns a custom color gradient for heatmaps."""
    return {
        0.0: '#0000FF',  # Blue for low crime
        0.25: '#00FFFF', # Cyan
        0.5: '#00FF00',  # Green
        0.75: '#FFFF00', # Yellow
        1.0: '#FF0000'   # Red for high crime
    }

# Function to categorize crime rates
def categorize_crime_rate(rate):
    """Categorize crime rates into risk levels."""
    if rate < 2:
        return "Low Risk"
    elif rate < 5:
        return "Moderate Risk"
    elif rate < 10:
        return "High Risk"
    else:
        return "Very High Risk"

# Step 1: Aggregate Crimes per Area
area_crime_counts = crime_data.groupby('Area_Name').size().reset_index(name='Crime_Count')

# Step 2: Normalize to % Crime Rate
total_crimes = area_crime_counts['Crime_Count'].sum()
area_crime_counts['Crime_Rate_Percent'] = (area_crime_counts['Crime_Count'] / total_crimes) * 100
area_crime_counts['Risk_Category'] = area_crime_counts['Crime_Rate_Percent'].apply(categorize_crime_rate)

# Step 3: Attach Area Center Lat/Lon
area_locations = crime_data.groupby('Area_Name')[['Latitude', 'Longitude']].mean().reset_index()

# Merge location with crime rates
area_crime_locations = pd.merge(area_crime_counts, area_locations, on='Area_Name', how='left')
area_crime_locations = safe_coordinates(area_crime_locations)

print(f"✅ Processed {len(area_crime_locations)} areas with valid coordinates")

# Step 4: Create Basic Crime Rate Heatmap
def create_basic_heatmap():
    """Create a basic crime rate heatmap."""
    map_center = [34.0522, -118.2437]  # Los Angeles coordinates
    m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

    # Prepare HeatMap data (Latitude, Longitude, Weight)
    heat_data = [
        [row['Latitude'], row['Longitude'], row['Crime_Rate_Percent']]
        for index, row in area_crime_locations.iterrows()
    ]

    # Add HeatMap Layer with custom gradient
    HeatMap(
        heat_data, 
        radius=20, 
        max_zoom=13, 
        blur=15, 
        gradient=get_custom_gradient(),
        min_opacity=0.5
    ).add_to(m)

    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Los Angeles Crime Rate Heatmap</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    output_path = "outputs_1/Basic_Crime_Rate_Heatmap.html"
    m.save(output_path)
    print(f"✅ Basic Crime Rate Heatmap saved to {output_path}")

    return m

# Step 5: Create Enhanced Heatmap with Tooltips and Risk Categories
def create_enhanced_heatmap():
    """Create an enhanced crime rate heatmap with tooltips and risk categories."""
    map_center = [34.0522, -118.2437]
    m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB dark_matter")

    # Add HeatMap Layer
    heat_data = [
        [row['Latitude'], row['Longitude'], row['Crime_Rate_Percent']]
        for index, row in area_crime_locations.iterrows()
    ]

    HeatMap(
        heat_data, 
        radius=25, 
        max_zoom=15, 
        blur=20, 
        gradient=get_custom_gradient(),
        min_opacity=0.6
    ).add_to(m)

    # Create a feature group for markers
    marker_group = folium.FeatureGroup(name="Area Markers")

    # Add CircleMarkers with enhanced tooltips
    for index, row in area_crime_locations.iterrows():
        # Determine marker color based on risk category
        if row['Risk_Category'] == "Low Risk":
            color = "blue"
        elif row['Risk_Category'] == "Moderate Risk":
            color = "green"
        elif row['Risk_Category'] == "High Risk":
            color = "orange"
        else:  # Very High Risk
            color = "red"

        # Create tooltip with more information
        tooltip_text = f"""
        <div style='font-family: Arial; font-size: 12px;'>
            <b>{row['Area_Name']}</b><br>
            Crime Rate: {row['Crime_Rate_Percent']:.2f}%<br>
            Crime Count: {row['Crime_Count']}<br>
            Risk Category: {row['Risk_Category']}
        </div>
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=folium.Tooltip(tooltip_text)
        ).add_to(marker_group)

    marker_group.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add title and legend
    title_html = '''
        <h3 align="center" style="font-size:18px"><b>Los Angeles Crime Rate Heatmap with Risk Categories</b></h3>
    '''

    legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color: white; padding: 10px; border-radius: 5px;">
            <b>Risk Categories</b><br>
            <i class="fa fa-circle" style="color:blue"></i> Low Risk<br>
            <i class="fa fa-circle" style="color:green"></i> Moderate Risk<br>
            <i class="fa fa-circle" style="color:orange"></i> High Risk<br>
            <i class="fa fa-circle" style="color:red"></i> Very High Risk<br>
        </div>
    '''

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    output_path = "outputs_1/Enhanced_Crime_Rate_Heatmap.html"
    m.save(output_path)
    print(f"✅ Enhanced Crime Rate Heatmap saved to {output_path}")

    return m

# Step 6: Create Time-based Heatmap (by hour of day)
def create_time_based_heatmap():
    """Create a time-based heatmap showing crime patterns by hour of day."""
    map_center = [34.0522, -118.2437]
    m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

    # Group data by hour
    hour_data = []

    for hour in range(24):
        # Filter data for this hour
        hour_crimes = crime_data[crime_data['Hour_Occurred'] == hour]

        if len(hour_crimes) > 0:
            # Group by area and calculate crime rates for this hour
            hour_area_counts = hour_crimes.groupby('Area_Name').size().reset_index(name='Hour_Crime_Count')
            hour_total = hour_area_counts['Hour_Crime_Count'].sum()
            hour_area_counts['Hour_Crime_Rate'] = (hour_area_counts['Hour_Crime_Count'] / hour_total) * 100

            # Merge with locations
            hour_locations = pd.merge(hour_area_counts, area_locations, on='Area_Name', how='left')
            hour_locations = safe_coordinates(hour_locations)

            # Create heat data for this hour
            hour_heat_data = [
                [row['Latitude'], row['Longitude'], row['Hour_Crime_Rate']]
                for index, row in hour_locations.iterrows()
            ]

            hour_data.append(hour_heat_data)
        else:
            # If no crimes at this hour, add empty list
            hour_data.append([])

    # Add time-based heatmap
    time_index = [f"{hour}:00" for hour in range(24)]

    HeatMapWithTime(
        hour_data,
        index=time_index,
        auto_play=True,
        max_opacity=0.8,
        radius=25,
        gradient=get_custom_gradient()
    ).add_to(m)

    # Add title
    title_html = '''
        <h3 align="center" style="font-size:16px"><b>Los Angeles Crime Rate by Hour of Day</b></h3>
        <p align="center" style="font-size:12px">Use the play button to see how crime patterns change throughout the day</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    output_path = "outputs_1/Time_Based_Crime_Heatmap.html"
    m.save(output_path)
    print(f"✅ Time-Based Crime Heatmap saved to {output_path}")

    return m

# Step 7: Create Predictive Crime Rate Heatmap
def create_predictive_heatmap():
    """Create a predictive crime rate heatmap using RandomForestRegressor."""
    try:
        print("🔍 Building predictive model for future crime rates...")

        # Prepare data for prediction
        # Group by Area and Month to see trends over time
        crime_data['Month_Year'] = crime_data['Year_Occurred'].astype(str) + '-' + crime_data['Month_Occurred'].astype(str)

        # Create time series data by area and month
        area_month_counts = crime_data.groupby(['Area_Name', 'Month_Year']).size().reset_index(name='Monthly_Crime_Count')

        # Pivot to get each month as a column
        area_time_series = area_month_counts.pivot(index='Area_Name', columns='Month_Year', values='Monthly_Crime_Count').reset_index()
        area_time_series = area_time_series.fillna(0)

        # Merge with area locations
        area_time_series = pd.merge(area_time_series, area_locations, on='Area_Name', how='left')
        area_time_series = safe_coordinates(area_time_series)

        # Prepare features (X) and target (y)
        # Use all month columns as features to predict the next month
        feature_cols = [col for col in area_time_series.columns if col not in ['Area_Name', 'Latitude', 'Longitude']]

        if len(feature_cols) < 2:
            print("❌ Not enough time series data for prediction")
            return None

        X = area_time_series[feature_cols]
        y = area_time_series[feature_cols[-1]]  # Use the last month as target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ Model trained with MSE: {mse:.2f}, R²: {r2:.2f}")

        # Predict next month for all areas
        X_all_scaled = scaler.transform(X)
        predictions = model.predict(X_all_scaled)

        # Add predictions to dataframe
        area_time_series['Predicted_Crime_Count'] = predictions

        # Calculate predicted crime rate percentage
        total_predicted = area_time_series['Predicted_Crime_Count'].sum()
        area_time_series['Predicted_Crime_Rate'] = (area_time_series['Predicted_Crime_Count'] / total_predicted) * 100

        # Create the map
        map_center = [34.0522, -118.2437]
        m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB dark_matter")

        # Prepare HeatMap data for predictions
        pred_heat_data = [
            [row['Latitude'], row['Longitude'], row['Predicted_Crime_Rate']]
            for index, row in area_time_series.iterrows()
        ]

        # Add HeatMap Layer for predictions
        HeatMap(
            pred_heat_data, 
            radius=25, 
            max_zoom=15, 
            blur=20, 
            gradient={
                0.0: '#00FFFF',  # Cyan for low predicted crime
                0.5: '#FFFF00',  # Yellow for medium
                1.0: '#FF00FF'   # Magenta for high predicted crime
            },
            min_opacity=0.7,
            name="Predicted Crime Rates"
        ).add_to(m)

        # Add markers with predicted values
        marker_cluster = MarkerCluster(name="Area Predictions").add_to(m)

        for index, row in area_time_series.iterrows():
            # Create tooltip with prediction information
            tooltip_text = f"""
            <div style='font-family: Arial; font-size: 12px;'>
                <b>{row['Area_Name']}</b><br>
                Predicted Crime Rate: {row['Predicted_Crime_Rate']:.2f}%<br>
                Predicted Crime Count: {row['Predicted_Crime_Count']:.0f}
            </div>
            """

            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                tooltip=folium.Tooltip(tooltip_text),
                icon=folium.Icon(color='purple', icon='info-sign')
            ).add_to(marker_cluster)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add title and explanation
        title_html = '''
            <h3 align="center" style="font-size:18px"><b>Predicted Future Crime Rates in Los Angeles</b></h3>
            <p align="center" style="font-size:12px">This map shows predicted crime rates based on historical patterns</p>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Save the map
        output_path = "outputs_1/Predicted_Crime_Rate_Heatmap.html"
        m.save(output_path)
        print(f"✅ Predicted Crime Rate Heatmap saved to {output_path}")

        return m

    except Exception as e:
        print(f"❌ Error creating predictive heatmap: {e}")
        return None

# Execute all visualization functions
print("\n🔍 Creating Basic Crime Rate Heatmap...")
create_basic_heatmap()

print("\n🔍 Creating Enhanced Crime Rate Heatmap with Risk Categories...")
create_enhanced_heatmap()

print("\n🔍 Creating Time-Based Crime Heatmap...")
create_time_based_heatmap()

print("\n🔍 Creating Predictive Crime Rate Heatmap...")
create_predictive_heatmap()

print("\n✅ All visualizations completed and saved to outputs_1/ directory")
