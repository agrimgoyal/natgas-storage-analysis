#!/usr/bin/env python3
"""
Price Data Preparation Utility
Helps format price data for use with the storage correlation analyzer
"""

import pandas as pd
import sys
from datetime import datetime

def prepare_price_data(input_file, output_file='prepared_prices.csv', 
                       date_col=None, price_col=None):
    """
    Prepare price data for analysis
    
    Parameters:
    -----------
    input_file : str
        Path to input price data (CSV or Excel)
    output_file : str
        Path for cleaned output file
    date_col : str, optional
        Name of date column (auto-detected if None)
    price_col : str, optional
        Name of price column (auto-detected if None)
    """
    
    print("="*70)
    print("PRICE DATA PREPARATION UTILITY")
    print("="*70)
    
    # Read input file
    print(f"\nReading: {input_file}")
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(input_file)
    else:
        raise ValueError("File must be CSV or Excel format")
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Auto-detect date column if not provided
    if date_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['date', 'time', 'day']):
                date_col = col
                break
    
    if date_col is None:
        print("\nWarning: Could not auto-detect date column. Using first column.")
        date_col = df.columns[0]
    
    print(f"\nUsing date column: '{date_col}'")
    
    # Auto-detect price column if not provided
    if price_col is None:
        for col in df.columns:
            if col == date_col:
                continue
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['price', 'settle', 'close', 'value', 'last']):
                price_col = col
                break
    
    if price_col is None:
        # Use second column if can't detect
        price_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    print(f"Using price column: '{price_col}'")
    
    # Parse dates
    print("\nParsing dates...")
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Check for parsing errors
    null_dates = df['Date'].isnull().sum()
    if null_dates > 0:
        print(f"  Warning: Could not parse {null_dates} dates")
    
    # Parse prices
    print("Parsing prices...")
    df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
    
    # Check for parsing errors
    null_prices = df['Price'].isnull().sum()
    if null_prices > 0:
        print(f"  Warning: Could not parse {null_prices} prices")
    
    # Create clean dataframe
    clean_df = df[['Date', 'Price']].dropna()
    clean_df = clean_df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates
    duplicates = clean_df.duplicated(subset=['Date']).sum()
    if duplicates > 0:
        print(f"  Removing {duplicates} duplicate dates (keeping first)")
        clean_df = clean_df.drop_duplicates(subset=['Date'], keep='first')
    
    # Summary statistics
    print("\n" + "-"*70)
    print("CLEANED DATA SUMMARY")
    print("-"*70)
    print(f"Total observations: {len(clean_df)}")
    print(f"Date range: {clean_df['Date'].min()} to {clean_df['Date'].max()}")
    print(f"Price range: ${clean_df['Price'].min():.2f} to ${clean_df['Price'].max():.2f}")
    print(f"Average price: ${clean_df['Price'].mean():.2f}")
    print(f"Median price: ${clean_df['Price'].median():.2f}")
    
    # Check data frequency
    date_diffs = clean_df['Date'].diff().dt.days.dropna()
    if len(date_diffs) > 0:
        avg_gap = date_diffs.median()
        if avg_gap <= 1:
            frequency = "Daily"
        elif avg_gap <= 7:
            frequency = "Weekly"
        elif avg_gap <= 31:
            frequency = "Monthly"
        else:
            frequency = "Other"
        print(f"Data frequency: {frequency} (median gap: {avg_gap:.0f} days)")
    
    # Save cleaned data
    clean_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned data to: {output_file}")
    
    # Preview
    print("\nFirst 5 rows:")
    print(clean_df.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(clean_df.tail().to_string(index=False))
    
    print("\n" + "="*70)
    print("PREPARATION COMPLETE")
    print("="*70)
    print(f"\nYou can now use this file with the analyzer:")
    print(f"  python storage_correlation_analysis.py ngshistory.xls {output_file}")
    print()
    
    return clean_df


def download_sample_data():
    """
    Create a sample price dataset for testing
    """
    print("Creating sample price data for testing...")
    
    # Generate sample dates (weekly, last 5 years)
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    
    # Generate synthetic prices (random walk around $3.00)
    import numpy as np
    np.random.seed(42)
    
    price = 3.0
    prices = []
    for _ in dates:
        change = np.random.normal(0, 0.15)  # Random change
        price = max(1.5, min(6.0, price + change))  # Keep between 1.5 and 6.0
        prices.append(price)
    
    # Create dataframe
    sample_df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    output_file = 'sample_prices.csv'
    sample_df.to_csv(output_file, index=False)
    
    print(f"✓ Created sample price data: {output_file}")
    print(f"  {len(sample_df)} weekly observations")
    print(f"  Date range: {sample_df['Date'].min()} to {sample_df['Date'].max()}")
    print("\nNote: This is synthetic data for testing purposes only!")
    
    return sample_df


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Price Data Preparation Utility")
        print("="*70)
        print("\nUsage:")
        print("  python prepare_price_data.py <input_file> [output_file]")
        print("\nOptions:")
        print("  python prepare_price_data.py sample    - Create sample data")
        print("\nExamples:")
        print("  python prepare_price_data.py henry_hub.csv")
        print("  python prepare_price_data.py prices.xlsx cleaned_prices.csv")
        print()
        return
    
    if sys.argv[1].lower() == 'sample':
        download_sample_data()
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'prepared_prices.csv'
    
    try:
        prepare_price_data(input_file, output_file)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. File exists and is readable")
        print("  2. File is in CSV or Excel format")
        print("  3. File contains date and price columns")
        sys.exit(1)


if __name__ == "__main__":
    main()
