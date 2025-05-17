# visualize.py
# -------------------
# Visualizes outputs from analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== 1. Setup ==========
# Load the cleaned summary files
city_summary = pd.read_csv("outputs/city_summary.csv")
college_summary = pd.read_csv("outputs/college_summary.csv")
county_summary = pd.read_csv("outputs/county_summary.csv")

# Create output folder for plots
os.makedirs("outputs/figures", exist_ok=True)

# ========== 2. Visualizations ==========

# --- 2.1 Top 10 Cities by Violent Crime ---
top10_cities = city_summary.sort_values('Violent_Crime_Total', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(x='Violent_Crime_Total', y='City', data=top10_cities, palette='Reds_r')
plt.title('Top 10 Cities by Violent Crime (All Years)')
plt.xlabel('Total Violent Crimes')
plt.ylabel('City')
plt.tight_layout()
plt.savefig("outputs/figures/top10_cities_violent_crime.png")
plt.close()

# --- 2.2 Violent Crime Rate Trend by Year (Overall) ---
city_yearly = city_summary.groupby('Year').agg({'Violent_Crime_Total': 'sum'}).reset_index()

plt.figure(figsize=(10,5))
sns.lineplot(x='Year', y='Violent_Crime_Total', data=city_yearly, marker='o')
plt.title('Violent Crimes Over Years (All Cities Combined)')
plt.xlabel('Year')
plt.ylabel('Total Violent Crimes')
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/violent_crime_trend.png")
plt.close()

# --- 2.3 Top 10 Colleges by Violent Crime Rate ---
top10_colleges = college_summary.sort_values('Violent_Crime_Rate', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(x='Violent_Crime_Rate', y='University/College', data=top10_colleges, palette='Blues_r')
plt.title('Top 10 Colleges by Violent Crime Rate')
plt.xlabel('Violent Crime Rate (per 1000 Students)')
plt.ylabel('University/College')
plt.tight_layout()
plt.savefig("outputs/figures/top10_colleges_violent_crime_rate.png")
plt.close()

# --- 2.4 Property Crime by Metro vs Non-Metro Counties ---
if 'Metropolitan/Nonmetropolitan' in county_summary.columns:
    metro_nonmetro = county_summary.groupby('Metropolitan/Nonmetropolitan').agg({'Property_Crime': 'sum'}).reset_index()

    plt.figure(figsize=(8,5))
    sns.barplot(x='Metropolitan/Nonmetropolitan', y='Property_Crime', data=metro_nonmetro, palette='coolwarm')
    plt.title('Property Crimes: Metro vs Non-Metro Counties')
    plt.xlabel('County Type')
    plt.ylabel('Total Property Crimes')
    plt.tight_layout()
    plt.savefig("outputs/figures/property_crime_metro_nonmetro.png")
    plt.close()

# --- 2.5 BONUS: Heatmap: Violent Crime Rate by State and Year ---
pivot_heatmap = city_summary.pivot_table(index='State', columns='Year', values='Violent_Crime_Rate', aggfunc='mean')

plt.figure(figsize=(14,10))
sns.heatmap(pivot_heatmap, cmap='Reds', linewidths=0.5)
plt.title('Violent Crime Rate Heatmap (State vs Year)')
plt.xlabel('Year')
plt.ylabel('State')
plt.tight_layout()
plt.savefig("outputs/figures/violent_crime_heatmap.png")
plt.close()

# --- 2.6 BONUS: Custom City Trend Over Time ---
city_name = "Phoenix"  # <-- Change this to whatever city you want

city_trend = city_summary[city_summary['City'].str.upper() == city_name.upper()]

if not city_trend.empty:
    plt.figure(figsize=(10,5))
    sns.lineplot(x='Year', y='Violent_Crime_Total', data=city_trend, marker='o')
    plt.title(f'Violent Crimes Over Time - {city_name}')
    plt.xlabel('Year')
    plt.ylabel('Total Violent Crimes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{city_name}_violent_crime_trend.png")
    plt.close()
else:
    print(f"⚠️ City '{city_name}' not found in data!")

# --- 2.7 BONUS: Metro vs Non-Metro Trend Over Years ---
if 'Metropolitan/Nonmetropolitan' in county_summary.columns:
    metro_yearly = county_summary.groupby(['Year', 'Metropolitan/Nonmetropolitan']).agg({'Property_Crime': 'sum'}).reset_index()

    plt.figure(figsize=(10,6))
    sns.lineplot(x='Year', y='Property_Crime', hue='Metropolitan/Nonmetropolitan', data=metro_yearly, marker='o')
    plt.title('Property Crime Trend: Metro vs Non-Metro Counties')
    plt.xlabel('Year')
    plt.ylabel('Total Property Crimes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/figures/property_crime_metro_nonmetro_trend.png")
    plt.close()

print("✅ All visualizations and bonus plots saved inside 'outputs/figures/'!")
