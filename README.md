# US Crime Data Analysis and Visualization

## Project Overview
This project provides a comprehensive analysis of US crime data, with a focus on Los Angeles crime data and college campus crime statistics. It includes data cleaning, feature engineering, machine learning modeling, and interactive visualizations through a Streamlit dashboard.

## Features
- **Data Cleaning**: Process raw crime data, handle missing values, and standardize formats
- **Feature Engineering**: Create meaningful features from raw data for analysis and modeling
- **Machine Learning Modeling**: Predict crime categories and analyze crime patterns using Random Forest models
- **Interactive Visualizations**: Generate heatmaps, time-based visualizations, and predictive crime rate maps
- **Streamlit Dashboard**: User-friendly interface to explore crime data and visualizations

## Project Structure
- `data_cleaning.py`: Cleans and preprocesses the raw crime data
- `feature_engineering.py`: Creates features for analysis and modeling
- `ml_modeling.py`: Builds and evaluates machine learning models for crime prediction
- `visualization.py`: Creates various visualizations including heatmaps and time-based maps
- `streamlit_dashboard.py`: Interactive dashboard for exploring crime data and visualizations
- `Data/`: Directory containing raw and processed data
- `outputs_1/`: Directory containing model outputs and visualizations

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Processing Pipeline
1. Run the data cleaning script:
   ```
   python data_cleaning.py
   ```
2. Run the feature engineering script:
   ```
   python feature_engineering.py
   ```
3. Run the machine learning modeling script:
   ```
   python ml_modeling.py
   ```
4. Generate visualizations:
   ```
   python visualization.py
   ```

### Streamlit Dashboard
Launch the interactive dashboard:
```
streamlit run streamlit_dashboard.py
```

The dashboard provides:
- Crime statistics and trends
- Interactive maps and heatmaps
- Crime prediction visualizations
- College campus crime analysis

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- folium
- streamlit
- plotly
- altair

## Data Sources
- Los Angeles crime data (2020 to present)
- US college campus crime statistics

## Output Files
The project generates several output files in the `outputs_1/` directory:
- Crime prediction summaries
- Location-based crime type analyses
- Interactive heatmaps (HTML files)
- Visualizations of crime patterns
