# analysis.py
# -------------------
# Loads, cleans, and performs basic descriptive analysis on Table 8, 9, 10.

import pandas as pd
import os

# ========== 1. Setup ==========
# Create outputs folder if not exist
os.makedirs("outputs", exist_ok=True)

# Load CSVs (Original Names)
table8 = pd.read_csv("table8.csv")
table9 = pd.read_csv("table9.csv")
table10 = pd.read_csv("table10.csv")


# ========== 2. Cleaning Functions ==========

def clean_table8(df):
    # Clean city-level data
    df.columns = df.columns.str.strip().str.replace('\n', '_').str.replace(' ', '_')
    
    # Remove commas and convert to float
    numeric_cols = ['Population', 'Violent_Crime_Total', 'Property_Crime_Total']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = df[col].fillna(0).astype(float)
    
    # Crime rates per 1,000 people
    df['Violent_Crime_Rate'] = (df['Violent_Crime_Total'] / df['Population']) * 1000
    df['Property_Crime_Rate'] = (df['Property_Crime_Total'] / df['Population']) * 1000
    return df

def clean_table9(df):
    # Clean college-level data
    df.columns = df.columns.str.strip().str.replace('\n', '_').str.replace(' ', '_')
    
    # Remove commas and convert to float
    numeric_cols = ['Student__Enrollment1', 'Violent__Crime', 'Property__Crime']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = df[col].fillna(0).astype(float)
    
    # Rename for easier handling
    df.rename(columns={
        'Student__Enrollment1': 'Student_Enrollment',
        'Violent__Crime': 'Violent_Crime',
        'Property__Crime': 'Property_Crime'
    }, inplace=True)
    
    # Crime rates per 1,000 students
    df['Violent_Crime_Rate'] = (df['Violent_Crime'] / df['Student_Enrollment']) * 1000
    df['Property_Crime_Rate'] = (df['Property_Crime'] / df['Student_Enrollment']) * 1000
    return df

def clean_table10(df):
    # Clean county-level data
    df.columns = df.columns.str.strip().str.replace('\n', '_').str.replace(' ', '_')
    
    # Remove commas and convert to float
    numeric_cols = ['Violent__Crime', 'Property__Crime']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = df[col].fillna(0).astype(float)
    
    # Rename for consistency
    df.rename(columns={
        'Violent__Crime': 'Violent_Crime',
        'Property__Crime': 'Property_Crime'
    }, inplace=True)
    
    return df

# ========== 3. Apply Cleaning ==========

table8_clean = clean_table8(table8)
table9_clean = clean_table9(table9)
table10_clean = clean_table10(table10)

# Save cleaned versions (optional, for backup)
table8_clean.to_csv("outputs/table8_clean_final.csv", index=False)
table9_clean.to_csv("outputs/table9_clean_final.csv", index=False)
table10_clean.to_csv("outputs/table10_clean_final.csv", index=False)

# ========== 4. Descriptive Statistics ==========

# ---- City-level stats ----
city_summary = table8_clean.groupby(['State', 'Year', 'City']).agg({
    'Population': 'mean',
    'Violent_Crime_Total': 'sum',
    'Property_Crime_Total': 'sum',
    'Violent_Crime_Rate': 'mean',
    'Property_Crime_Rate': 'mean'
}).reset_index()
city_summary.to_csv("outputs/city_summary.csv", index=False)

# ---- College-level stats ----
college_summary = table9_clean.groupby(['State', 'Year', 'University/College']).agg({
    'Student_Enrollment': 'mean',
    'Violent_Crime': 'sum',
    'Property_Crime': 'sum',
    'Violent_Crime_Rate': 'mean',
    'Property_Crime_Rate': 'mean'
}).reset_index()
college_summary.to_csv("outputs/college_summary.csv", index=False)

# ---- County-level stats ----
county_summary = table10_clean.groupby(['State', 'Year', 'County']).agg({
    'Violent_Crime': 'sum',
    'Property_Crime': 'sum'
}).reset_index()
county_summary.to_csv("outputs/county_summary.csv", index=False)

print("✅ Analysis complete! Cleaned data and summaries saved in 'outputs/' folder.")
