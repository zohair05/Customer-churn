import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the raw telecom churn dataset.
    """
    # 1. Convert 'TotalCharges' from string to float. 
    # 'coerce' turns empty spaces into NaN.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 2. Handle missing values by imputing with the median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # 3. Drop customerID as it has no predictive power
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    return df