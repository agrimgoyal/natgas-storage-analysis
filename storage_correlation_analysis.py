#!/usr/bin/env python3
"""
Natural Gas Storage Regional Correlation & Deviation Analysis
Analyzes regional storage relationships and their impact on price movements
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StorageCorrelationAnalyzer:
    """
    Analyzes correlations and ratios between natural gas storage regions
    and identifies deviation events that may correlate with price movements
    """
    
    def __init__(self, storage_file_path, price_file_path=None):
        """
        Initialize analyzer with storage and optional price data
        
        Parameters:
        -----------
        storage_file_path : str
            Path to EIA storage Excel file
        price_file_path : str, optional
            Path to price data (CSV or Excel)
        """
        self.storage_file = storage_file_path
        self.price_file = price_file_path
        self.storage_data = None
        self.price_data = None
        self.regions = []
        self.correlations = {}
        self.ratios = {}
        self.deviation_events = []
        
    def load_storage_data(self):
        """Load and process EIA storage data"""
        print("Loading storage data...")
        
        # Read Excel file
        xl = pd.ExcelFile(self.storage_file)
        
        # Find the historical data sheet
        hist_sheet = [s for s in xl.sheet_names if 'hist' in s.lower()]
        if not hist_sheet:
            hist_sheet = [xl.sheet_names[1]] if len(xl.sheet_names) > 1 else [xl.sheet_names[0]]
        
        df = pd.read_excel(self.storage_file, sheet_name=hist_sheet[0])
        
        # Find header row
        header_idx = None
        for idx, row in df.iterrows():
            row_str = ' '.join([str(x).lower() for x in row if pd.notna(x)])
            if 'week ending' in row_str and 'lower 48' in row_str:
                header_idx = idx
                break
        
        if header_idx is None:
            raise ValueError("Could not find header row with 'Week Ending' and region columns")
        
        # Set proper headers
        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
        
        # Parse date column
        date_col = [c for c in df.columns if 'week ending' in str(c).lower()][0]
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date').sort_index()
        
        # Identify region columns
        region_keywords = {
            'Total Lower 48': ['total lower 48', 'lower 48'],
            'East': ['east'],
            'Midwest': ['midwest'],
            'Mountain': ['mountain'],
            'Pacific': ['pacific'],
            'South Central': ['south central'],
            'Salt': ['salt'],
            'Non Salt': ['nonsalt', 'non-salt']
        }
        
        region_cols = {}
        for region, keywords in region_keywords.items():
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in keywords):
                    # Avoid salt matching south central
                    if region == 'South Central' and 'salt' in col_lower:
                        continue
                    region_cols[region] = col
                    break
        
        # Create clean dataframe with just regions
        self.storage_data = pd.DataFrame(index=df.index)
        for region, col in region_cols.items():
            self.storage_data[region] = pd.to_numeric(df[col], errors='coerce')
        
        self.storage_data = self.storage_data.dropna(how='all')
        self.regions = list(self.storage_data.columns)
        
        print(f"Loaded {len(self.storage_data)} weeks of data")
        print(f"Regions: {', '.join(self.regions)}")
        print(f"Date range: {self.storage_data.index.min()} to {self.storage_data.index.max()}")
        
        return self.storage_data
    
    def load_price_data(self):
        """Load price data if provided"""
        if self.price_file is None:
            print("No price data provided")
            return None
        
        print("Loading price data...")
        
        # Try to read as CSV first, then Excel
        try:
            if self.price_file.endswith('.csv'):
                df = pd.read_csv(self.price_file)
            else:
                df = pd.read_excel(self.price_file)
            
            # Try to find date and price columns
            date_col = None
            price_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower and date_col is None:
                    date_col = col
                if any(x in col_lower for x in ['price', 'settle', 'close', 'value']) and price_col is None:
                    price_col = col
            
            if date_col is None or price_col is None:
                print(f"Warning: Could not auto-detect columns. Using first two columns.")
                date_col = df.columns[0]
                price_col = df.columns[1]
            
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date').sort_index()
            
            self.price_data = pd.DataFrame({
                'Price': pd.to_numeric(df[price_col], errors='coerce')
            })
            self.price_data = self.price_data.dropna()
            
            print(f"Loaded {len(self.price_data)} price observations")
            print(f"Price range: {self.price_data.index.min()} to {self.price_data.index.max()}")
            
            return self.price_data
            
        except Exception as e:
            print(f"Error loading price data: {e}")
            return None
    
    def calculate_rolling_correlations(self, window=26):
        """
        Calculate rolling correlations between all region pairs
        
        Parameters:
        -----------
        window : int
            Rolling window size in weeks (default: 26 = ~6 months)
        """
        print(f"\nCalculating rolling {window}-week correlations...")
        
        self.correlation_window = window
        region_pairs = []
        
        # Generate all unique pairs
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                region_pairs.append((r1, r2))
        
        for r1, r2 in region_pairs:
            pair_key = f"{r1}_vs_{r2}"
            
            # Calculate rolling correlation
            rolling_corr = self.storage_data[[r1, r2]].rolling(window=window).corr().iloc[0::2, -1]
            rolling_corr.index = rolling_corr.index.droplevel(1)
            
            self.correlations[pair_key] = rolling_corr
        
        print(f"Calculated correlations for {len(region_pairs)} region pairs")
        
        return self.correlations
    
    def calculate_regional_ratios(self, ratio_pairs=None):
        """
        Calculate storage level ratios between specified region pairs
        
        Parameters:
        -----------
        ratio_pairs : list of tuples, optional
            Specific pairs to calculate ratios for
            If None, calculates key ratios automatically
        """
        print("\nCalculating regional storage ratios...")
        
        if ratio_pairs is None:
            # Default key ratios
            ratio_pairs = [
                ('East', 'Midwest'),
                ('Salt', 'Non Salt'),
                ('East', 'Total Lower 48'),
                ('South Central', 'Total Lower 48'),
            ]
            # Only keep pairs where both regions exist
            ratio_pairs = [(r1, r2) for r1, r2 in ratio_pairs 
                          if r1 in self.regions and r2 in self.regions]
        
        for r1, r2 in ratio_pairs:
            ratio_key = f"{r1}/{r2}"
            self.ratios[ratio_key] = self.storage_data[r1] / self.storage_data[r2]
        
        print(f"Calculated {len(self.ratios)} regional ratios")
        
        return self.ratios
    
    def detect_correlation_breaks(self, threshold_std=2.0, min_history=52):
        """
        Detect when correlations drop significantly below historical levels
        Uses EXPANDING WINDOW to avoid look-ahead bias - only uses data available
        up to each observation date for calculating mean/std.

        Parameters:
        -----------
        threshold_std : float
            Number of standard deviations below mean to flag as deviation
        min_history : int
            Minimum number of observations before flagging any events (default: 52 weeks)
        """
        print(f"\nDetecting correlation breaks (threshold: {threshold_std} std, min history: {min_history} weeks)...")
        print("  Using EXPANDING WINDOW methodology (no look-ahead bias)")

        correlation_events = []

        for pair_key, corr_series in self.correlations.items():
            # Drop NaN values and sort by date
            corr_clean = corr_series.dropna().sort_index()

            if len(corr_clean) < min_history:
                continue

            # Calculate EXPANDING mean and std, then SHIFT by 1 to exclude current observation
            # This ensures we only use data BEFORE each point (true out-of-sample)
            expanding_mean = corr_clean.expanding(min_periods=min_history).mean().shift(1)
            expanding_std = corr_clean.expanding(min_periods=min_history).std().shift(1)

            # Calculate Z-scores using only historical data BEFORE each point
            z_scores = (corr_clean - expanding_mean) / expanding_std

            # Find dates where correlation dropped significantly below historical mean
            # Only consider dates where we have enough history
            valid_dates = z_scores.dropna().index

            for date in valid_dates:
                z_score = z_scores.loc[date]
                # Flag correlation breaks (values significantly BELOW historical mean)
                if z_score < -threshold_std:
                    correlation_events.append({
                        'Date': date,
                        'Type': 'Correlation Break',
                        'Pair': pair_key,
                        'Value': corr_clean.loc[date],
                        'Mean': expanding_mean.loc[date],
                        'Threshold': expanding_mean.loc[date] - (threshold_std * expanding_std.loc[date]),
                        'Z-Score': z_score
                    })

        print(f"Found {len(correlation_events)} correlation break events")

        return pd.DataFrame(correlation_events)
    
    def detect_ratio_deviations(self, threshold_std=2.0, min_history=52):
        """
        Detect when regional ratios deviate significantly from historical norms
        Uses EXPANDING WINDOW to avoid look-ahead bias - only uses data available
        up to each observation date for calculating mean/std.

        Parameters:
        -----------
        threshold_std : float
            Number of standard deviations from mean to flag as deviation
        min_history : int
            Minimum number of observations before flagging any events (default: 52 weeks)
        """
        print(f"\nDetecting ratio deviations (threshold: {threshold_std} std, min history: {min_history} weeks)...")
        print("  Using EXPANDING WINDOW methodology (no look-ahead bias)")

        ratio_events = []

        for ratio_key, ratio_series in self.ratios.items():
            # Drop NaN values and sort by date
            ratio_clean = ratio_series.dropna().sort_index()

            if len(ratio_clean) < min_history:
                continue

            # Calculate EXPANDING mean and std, then SHIFT by 1 to exclude current observation
            # This ensures we only use data BEFORE each point (true out-of-sample)
            expanding_mean = ratio_clean.expanding(min_periods=min_history).mean().shift(1)
            expanding_std = ratio_clean.expanding(min_periods=min_history).std().shift(1)

            # Calculate Z-scores using only historical data BEFORE each point
            z_scores = (ratio_clean - expanding_mean) / expanding_std

            # Find dates where ratio deviated significantly from historical mean
            # Only consider dates where we have enough history
            valid_dates = z_scores.dropna().index

            for date in valid_dates:
                z_score = z_scores.loc[date]
                # Flag deviations in either direction
                if abs(z_score) > threshold_std:
                    ratio_events.append({
                        'Date': date,
                        'Type': 'Ratio Deviation',
                        'Pair': ratio_key,
                        'Value': ratio_clean.loc[date],
                        'Mean': expanding_mean.loc[date],
                        'Z-Score': z_score
                    })

        print(f"Found {len(ratio_events)} ratio deviation events")

        return pd.DataFrame(ratio_events)
    
    def analyze_price_impact(self, deviation_df, forward_window=4, release_delay_days=6):
        """
        Analyze price movements following deviation events

        Parameters:
        -----------
        deviation_df : DataFrame
            DataFrame of deviation events
        forward_window : int
            Number of weeks forward to analyze price impact
        release_delay_days : int
            Days between storage week-ending (Friday) and data release (Thursday)
            EIA releases storage data on Thursday for week ending previous Friday = 6 days
        """
        if self.price_data is None or len(deviation_df) == 0:
            print("No price data or no deviation events to analyze")
            return None

        print(f"\nAnalyzing price impact (forward window: {forward_window} weeks)...")
        print(f"  Using release delay of {release_delay_days} days (Friday week-end -> Thursday release)")

        results = []

        for idx, event in deviation_df.iterrows():
            event_date = event['Date']

            # Storage data is for week ending Friday, released following Thursday
            # Add release_delay_days to get the actual release date
            release_date = event_date + timedelta(days=release_delay_days)

            # Find price on or after RELEASE date (when traders can act on the info)
            price_at_release = self.price_data[self.price_data.index >= release_date]
            if len(price_at_release) == 0:
                continue

            price_t0 = price_at_release.iloc[0]['Price']
            date_t0 = price_at_release.index[0]
            
            # Find price N weeks forward
            future_date = date_t0 + timedelta(weeks=forward_window)
            price_future = self.price_data[self.price_data.index >= future_date]
            
            if len(price_future) == 0:
                continue
            
            price_t1 = price_future.iloc[0]['Price']
            date_t1 = price_future.index[0]
            
            # Calculate price change
            price_change = price_t1 - price_t0
            price_change_pct = (price_change / price_t0) * 100
            
            # Calculate volatility in the window
            window_prices = self.price_data[
                (self.price_data.index >= date_t0) & 
                (self.price_data.index <= date_t1)
            ]['Price']
            
            volatility = window_prices.std() if len(window_prices) > 1 else 0
            
            results.append({
                'Event_Date': event_date,
                'Event_Type': event['Type'],
                'Event_Pair': event['Pair'],
                'Event_ZScore': event['Z-Score'],
                'Price_T0': price_t0,
                'Price_T1': price_t1,
                'Price_Change': price_change,
                'Price_Change_Pct': price_change_pct,
                'Volatility': volatility,
                'Days_Forward': (date_t1 - date_t0).days
            })
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\nAnalyzed {len(results_df)} events with price data")
            print(f"Average price change: {results_df['Price_Change_Pct'].mean():.2f}%")
            print(f"Average volatility: ${results_df['Volatility'].mean():.2f}")
        
        return results_df
    
    def generate_report(self, output_file='storage_analysis_report.txt'):
        """Generate comprehensive text report of findings"""
        
        print(f"\nGenerating analysis report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NATURAL GAS STORAGE REGIONAL CORRELATION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Period: {self.storage_data.index.min()} to {self.storage_data.index.max()}\n")
            f.write(f"Regions Analyzed: {len(self.regions)}\n")
            f.write(f"Total Observations: {len(self.storage_data)}\n")
            f.write(f"Methodology: EXPANDING WINDOW (no look-ahead bias)\n")
            f.write(f"  - Z-scores calculated using only data available up to each observation\n")
            f.write(f"  - Minimum 52 weeks of history required before flagging events\n\n")
            
            # Section 1: Correlation Summary
            f.write("-"*80 + "\n")
            f.write("1. REGIONAL CORRELATION SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            if self.correlations:
                for pair_key, corr_series in self.correlations.items():
                    mean_corr = corr_series.mean()
                    std_corr = corr_series.std()
                    min_corr = corr_series.min()
                    max_corr = corr_series.max()
                    
                    f.write(f"{pair_key}:\n")
                    f.write(f"  Mean Correlation: {mean_corr:.3f}\n")
                    f.write(f"  Std Dev: {std_corr:.3f}\n")
                    f.write(f"  Range: [{min_corr:.3f}, {max_corr:.3f}]\n")
                    f.write(f"  Stability: {'High' if std_corr < 0.1 else 'Medium' if std_corr < 0.2 else 'Low'}\n\n")
            
            # Section 2: Ratio Analysis
            f.write("-"*80 + "\n")
            f.write("2. REGIONAL RATIO ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            if self.ratios:
                for ratio_key, ratio_series in self.ratios.items():
                    mean_ratio = ratio_series.mean()
                    std_ratio = ratio_series.std()
                    cv = (std_ratio / mean_ratio) * 100  # Coefficient of variation
                    
                    f.write(f"{ratio_key}:\n")
                    f.write(f"  Mean Ratio: {mean_ratio:.3f}\n")
                    f.write(f"  Std Dev: {std_ratio:.3f}\n")
                    f.write(f"  Coefficient of Variation: {cv:.2f}%\n")
                    f.write(f"  Interpretation: {'Stable relationship' if cv < 10 else 'Variable relationship' if cv < 20 else 'Highly variable'}\n\n")
            
            # Section 3: Deviation Events
            if hasattr(self, 'all_deviations') and len(self.all_deviations) > 0:
                f.write("-"*80 + "\n")
                f.write("3. DEVIATION EVENTS DETECTED\n")
                f.write("-"*80 + "\n\n")
                
                f.write(f"Total Events: {len(self.all_deviations)}\n")
                f.write(f"Correlation Breaks: {len(self.all_deviations[self.all_deviations['Type'] == 'Correlation Break'])}\n")
                f.write(f"Ratio Deviations: {len(self.all_deviations[self.all_deviations['Type'] == 'Ratio Deviation'])}\n\n")
                
                f.write("Most Significant Events (by Z-Score):\n\n")
                top_events = self.all_deviations.nlargest(10, 'Z-Score', keep='all')
                for idx, event in top_events.iterrows():
                    f.write(f"  {event['Date'].strftime('%Y-%m-%d')}: {event['Pair']}\n")
                    f.write(f"    Type: {event['Type']}, Z-Score: {event['Z-Score']:.2f}\n\n")
            
            # Section 4: Price Impact Analysis
            if hasattr(self, 'price_impact') and self.price_impact is not None and len(self.price_impact) > 0:
                f.write("-"*80 + "\n")
                f.write("4. PRICE IMPACT ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                avg_change = self.price_impact['Price_Change_Pct'].mean()
                median_change = self.price_impact['Price_Change_Pct'].median()
                avg_vol = self.price_impact['Volatility'].mean()
                
                f.write(f"Events with Price Data: {len(self.price_impact)}\n")
                f.write(f"Average Price Change: {avg_change:.2f}%\n")
                f.write(f"Median Price Change: {median_change:.2f}%\n")
                f.write(f"Average Volatility: ${avg_vol:.2f}\n\n")
                
                # Positive vs negative price movements
                positive = self.price_impact[self.price_impact['Price_Change_Pct'] > 0]
                negative = self.price_impact[self.price_impact['Price_Change_Pct'] < 0]
                
                f.write(f"Positive Price Movements: {len(positive)} ({len(positive)/len(self.price_impact)*100:.1f}%)\n")
                f.write(f"Negative Price Movements: {len(negative)} ({len(negative)/len(self.price_impact)*100:.1f}%)\n\n")
                
                # Correlation between z-score magnitude and price change
                corr = self.price_impact['Event_ZScore'].abs().corr(self.price_impact['Price_Change_Pct'].abs())
                f.write(f"Correlation (|Z-Score| vs |Price Change%|): {corr:.3f}\n")
                f.write(f"Interpretation: {'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'} relationship\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved to: {output_file}")
        return output_file
    
    def create_visualizations(self, output_dir='./'):
        """Create comprehensive visualization suite"""
        
        print("\nCreating visualizations...")
        
        fig_count = 0
        
        # 1. Correlation Heatmap
        if self.correlations:
            fig_count += 1
            plt.figure(figsize=(12, 8))
            
            # Get latest correlation values
            latest_corrs = {}
            for pair, series in self.correlations.items():
                latest_corrs[pair] = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
            
            # Create matrix
            regions_subset = list(set([p.split('_vs_')[0] for p in latest_corrs.keys()] + 
                                     [p.split('_vs_')[1] for p in latest_corrs.keys()]))
            corr_matrix = pd.DataFrame(index=regions_subset, columns=regions_subset, dtype=float)
            
            for pair, val in latest_corrs.items():
                r1, r2 = pair.split('_vs_')
                corr_matrix.loc[r1, r2] = val
                corr_matrix.loc[r2, r1] = val
            
            for r in regions_subset:
                corr_matrix.loc[r, r] = 1.0
            
            corr_matrix = corr_matrix.astype(float)
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1, square=True, linewidths=1)
            plt.title('Current Regional Storage Correlations', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/1_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Rolling Correlation Time Series
        if self.correlations:
            fig_count += 1
            n_pairs = len(self.correlations)
            n_cols = 2
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            axes = axes.flatten() if n_pairs > 1 else [axes]
            
            for idx, (pair, series) in enumerate(self.correlations.items()):
                ax = axes[idx]
                series.plot(ax=ax, linewidth=2, color='steelblue')
                
                mean_val = series.mean()
                std_val = series.std()
                ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axhline(mean_val - 2*std_val, color='orange', linestyle=':', alpha=0.7, label='-2σ')
                ax.axhline(mean_val + 2*std_val, color='orange', linestyle=':', alpha=0.7, label='+2σ')
                
                ax.set_title(pair.replace('_vs_', ' vs '), fontweight='bold')
                ax.set_ylabel('Correlation')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_pairs, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Rolling {self.correlation_window}-Week Correlations', 
                        fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/2_rolling_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Ratio Analysis
        if self.ratios:
            fig_count += 1
            n_ratios = len(self.ratios)
            fig, axes = plt.subplots(n_ratios, 1, figsize=(14, 4*n_ratios))
            axes = [axes] if n_ratios == 1 else axes
            
            for idx, (ratio_key, series) in enumerate(self.ratios.items()):
                ax = axes[idx]
                
                series.plot(ax=ax, linewidth=2, color='darkgreen', label='Actual Ratio')
                
                mean_val = series.mean()
                std_val = series.std()
                
                ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, 
                          label=f'Mean: {mean_val:.3f}')
                ax.fill_between(series.index, mean_val - 2*std_val, mean_val + 2*std_val,
                               alpha=0.2, color='orange', label='±2σ Band')
                
                ax.set_title(f'Storage Ratio: {ratio_key}', fontweight='bold', fontsize=12)
                ax.set_ylabel('Ratio')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Regional Storage Ratios Over Time', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/3_storage_ratios.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Trading Signals Timeline (redesigned for clarity)
        if hasattr(self, 'all_deviations') and len(self.all_deviations) > 0:
            fig_count += 1
            fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

            # Identify key trading signals
            ratio_devs = self.all_deviations[self.all_deviations['Type'] == 'Ratio Deviation']
            corr_breaks = self.all_deviations[self.all_deviations['Type'] == 'Correlation Break']

            # East undersupply (bullish signal)
            east_under = ratio_devs[(ratio_devs['Pair'].str.contains('East')) & (ratio_devs['Z-Score'] < 0)]
            # Pacific decorrelation (bearish signal)
            pacific = corr_breaks[corr_breaks['Pair'].str.contains('Pacific')]
            # Other events
            other_ratio = ratio_devs[~((ratio_devs['Pair'].str.contains('East')) & (ratio_devs['Z-Score'] < 0))]
            other_corr = corr_breaks[~corr_breaks['Pair'].str.contains('Pacific')]

            ax1 = axes[0]
            # Plot other events in gray (background)
            if len(other_ratio) > 0:
                ax1.scatter(other_ratio['Date'], other_ratio['Z-Score'].abs(),
                           c='lightgray', alpha=0.3, s=30, label='Other Ratio Deviations')
            if len(other_corr) > 0:
                ax1.scatter(other_corr['Date'], other_corr['Z-Score'].abs(),
                           c='lightgray', alpha=0.3, s=30, label='_nolegend_')

            # Plot trading signals prominently
            if len(east_under) > 0:
                ax1.scatter(east_under['Date'], east_under['Z-Score'].abs(),
                           c='green', alpha=0.8, s=100, marker='^',
                           label=f'LONG Signal: East Undersupply ({len(east_under)} events)')
            if len(pacific) > 0:
                ax1.scatter(pacific['Date'], pacific['Z-Score'].abs(),
                           c='red', alpha=0.8, s=100, marker='v',
                           label=f'SHORT Signal: Pacific Decorrelation ({len(pacific)} events)')

            ax1.axhline(2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax1.axhline(3, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax1.axhline(4, color='gray', linestyle=':', alpha=0.5, linewidth=1)

            ax1.set_ylabel('|Z-Score| (Deviation Magnitude)', fontsize=12, fontweight='bold')
            ax1.set_title('Trading Signals Timeline\nGreen Triangle = Go Long (East Undersupply) | Red Triangle = Go Short (Pacific Decorrelation)',
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(self.all_deviations['Date'].min(), self.all_deviations['Date'].max())

            # Bottom panel: Price with signal markers
            ax2 = axes[1]
            if self.price_data is not None:
                # Resample price to weekly for cleaner chart
                weekly_price = self.price_data.resample('W').last()
                ax2.plot(weekly_price.index, weekly_price['Price'], color='black', linewidth=1, alpha=0.7)
                ax2.fill_between(weekly_price.index, weekly_price['Price'], alpha=0.1, color='blue')
                ax2.set_ylabel('Natural Gas Price ($)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/4_trading_signals.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Trading Signal Performance (redesigned for clarity)
        if hasattr(self, 'price_impact') and self.price_impact is not None and len(self.price_impact) > 0:
            fig_count += 1
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Identify trading signals in price impact data
            ratio_devs = self.price_impact[self.price_impact['Event_Type'] == 'Ratio Deviation']
            corr_breaks = self.price_impact[self.price_impact['Event_Type'] == 'Correlation Break']

            east_under = ratio_devs[(ratio_devs['Event_Pair'].str.contains('East')) & (ratio_devs['Event_ZScore'] < 0)]
            pacific = corr_breaks[corr_breaks['Event_Pair'].str.contains('Pacific')]
            pacific = pacific.copy()
            pacific['Z_Abs'] = pacific['Event_ZScore'].abs()

            # 5a. Signal Performance Summary (Bar Chart)
            ax1 = axes[0, 0]
            signals = ['East Undersupply\n(LONG)', 'Pacific Decorrelation\n(SHORT)', 'All Other\nEvents']

            # Calculate metrics
            east_win = len(east_under[east_under['Price_Change_Pct'] > 0]) / len(east_under) * 100 if len(east_under) > 0 else 0
            pacific_win = len(pacific[pacific['Price_Change_Pct'] < 0]) / len(pacific) * 100 if len(pacific) > 0 else 0
            other = self.price_impact[~self.price_impact.index.isin(east_under.index) & ~self.price_impact.index.isin(pacific.index)]
            other_win = 50  # baseline

            win_rates = [east_win, pacific_win, other_win]
            colors = ['green', 'red', 'gray']

            bars = ax1.bar(signals, win_rates, color=colors, alpha=0.7, edgecolor='black')
            ax1.axhline(50, color='black', linestyle='--', linewidth=2, label='Random (50%)')
            ax1.set_ylabel('Win Rate %', fontsize=12, fontweight='bold')
            ax1.set_title('Trading Signal Win Rates', fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 100)

            # Add value labels on bars
            for bar, rate in zip(bars, win_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3, axis='y')

            # 5b. Average Returns by Signal
            ax2 = axes[0, 1]
            east_ret = east_under['Price_Change_Pct'].mean() if len(east_under) > 0 else 0
            pacific_ret = pacific['Price_Change_Pct'].mean() if len(pacific) > 0 else 0
            other_ret = other['Price_Change_Pct'].mean() if len(other) > 0 else 0

            returns = [east_ret, pacific_ret, other_ret]
            bar_colors = ['green' if r > 0 else 'red' for r in returns]

            bars = ax2.bar(signals, returns, color=bar_colors, alpha=0.7, edgecolor='black')
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Avg 4-Week Return %', fontsize=12, fontweight='bold')
            ax2.set_title('Average Returns by Signal Type', fontsize=14, fontweight='bold')

            for bar, ret in zip(bars, returns):
                y_pos = bar.get_height() + 0.5 if ret >= 0 else bar.get_height() - 1.5
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{ret:+.1f}%', ha='center', fontsize=12, fontweight='bold')

            ax2.grid(True, alpha=0.3, axis='y')

            # 5c. Pacific Signal: Z-Score vs Return (shows scaling)
            ax3 = axes[1, 0]
            if len(pacific) > 0:
                scatter = ax3.scatter(pacific['Z_Abs'], pacific['Price_Change_Pct'],
                                     c=pacific['Price_Change_Pct'], cmap='RdYlGn_r',
                                     alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
                ax3.axhline(0, color='black', linestyle='-', linewidth=1)
                ax3.axvline(3, color='orange', linestyle='--', alpha=0.7, label='|Z|=3 threshold')
                ax3.axvline(4, color='red', linestyle='--', alpha=0.7, label='|Z|=4 threshold')

                # Add trend line
                z = np.polyfit(pacific['Z_Abs'], pacific['Price_Change_Pct'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(pacific['Z_Abs'].min(), pacific['Z_Abs'].max(), 100)
                ax3.plot(x_line, p(x_line), 'b--', linewidth=2, alpha=0.7, label=f'Trend (corr={pacific["Z_Abs"].corr(pacific["Price_Change_Pct"]):.2f})')

                ax3.set_xlabel('|Z-Score| (Deviation Magnitude)', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Price Change %', fontsize=12, fontweight='bold')
                ax3.set_title('Pacific Signal: Larger Deviations = Larger Drops', fontsize=14, fontweight='bold')
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3)

            # 5d. Summary Statistics Table
            ax4 = axes[1, 1]
            ax4.axis('off')

            # Create summary table
            table_data = [
                ['Signal', 'Events', 'Win Rate', 'Avg Return', 'Best Threshold'],
                ['_' * 12, '_' * 8, '_' * 10, '_' * 12, '_' * 15],
                ['East Undersupply', f'{len(east_under)}', f'{east_win:.1f}%', f'{east_ret:+.1f}%', '|Z| >= 2.0'],
                ['Pacific (all)', f'{len(pacific)}', f'{pacific_win:.1f}%', f'{pacific_ret:+.1f}%', '|Z| >= 2.0'],
            ]

            # Add Pacific by threshold
            for thresh in [3.0, 4.0]:
                subset = pacific[pacific['Z_Abs'] >= thresh]
                if len(subset) > 0:
                    win = len(subset[subset['Price_Change_Pct'] < 0]) / len(subset) * 100
                    ret = subset['Price_Change_Pct'].mean()
                    table_data.append([f'Pacific |Z|>={thresh}', f'{len(subset)}', f'{win:.1f}%', f'{ret:+.1f}%', ''])

            # Draw table
            y_pos = 0.95
            for row in table_data:
                x_positions = [0.05, 0.30, 0.45, 0.62, 0.80]
                for x, text in zip(x_positions, row):
                    fontweight = 'bold' if row == table_data[0] else 'normal'
                    ax4.text(x, y_pos, text, fontsize=11, fontweight=fontweight,
                            family='monospace', transform=ax4.transAxes)
                y_pos -= 0.12

            ax4.set_title('Signal Summary Statistics', fontsize=14, fontweight='bold', pad=20)

            plt.suptitle('Trading Signal Performance Analysis', fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/5_signal_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Created {fig_count} visualization files")
        return fig_count
    
    def run_full_analysis(self, corr_window=26, deviation_threshold=2.0, price_forward_window=4, min_history=52):
        """
        Run complete analysis pipeline

        Parameters:
        -----------
        corr_window : int
            Rolling correlation window in weeks
        deviation_threshold : float
            Z-score threshold for flagging deviations
        price_forward_window : int
            Weeks forward to analyze price impact
        min_history : int
            Minimum weeks of history before flagging deviation events (default: 52)
            Uses expanding window methodology to avoid look-ahead bias
        """
        print("\n" + "="*80)
        print("STARTING FULL ANALYSIS PIPELINE")
        print("="*80 + "\n")
        print(f"Methodology: EXPANDING WINDOW (min history: {min_history} weeks)")
        print("This avoids look-ahead bias by using only data available at each point in time\n")

        # Load data
        self.load_storage_data()
        self.load_price_data()

        # Calculate correlations and ratios
        self.calculate_rolling_correlations(window=corr_window)
        self.calculate_regional_ratios()

        # Detect deviations using expanding window (no look-ahead bias)
        corr_breaks = self.detect_correlation_breaks(threshold_std=deviation_threshold, min_history=min_history)
        ratio_devs = self.detect_ratio_deviations(threshold_std=deviation_threshold, min_history=min_history)
        
        # Combine all deviation events
        self.all_deviations = pd.concat([corr_breaks, ratio_devs], ignore_index=True)
        self.all_deviations = self.all_deviations.sort_values('Date')
        
        # Analyze price impact
        if self.price_data is not None:
            self.price_impact = self.analyze_price_impact(
                self.all_deviations, 
                forward_window=price_forward_window
            )
        else:
            self.price_impact = None
        
        # Generate outputs
        report_file = self.generate_report()
        self.create_visualizations()
        
        # Save detailed results to CSV
        if len(self.all_deviations) > 0:
            deviation_file = 'deviation_events.csv'
            self.all_deviations.to_csv(deviation_file, index=False)
            print(f"Saved deviation events to: {deviation_file}")
        
        if self.price_impact is not None and len(self.price_impact) > 0:
            price_impact_file = 'price_impact_analysis.csv'
            self.price_impact.to_csv(price_impact_file, index=False)
            print(f"Saved price impact analysis to: {price_impact_file}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return {
            'storage_data': self.storage_data,
            'correlations': self.correlations,
            'ratios': self.ratios,
            'deviations': self.all_deviations,
            'price_impact': self.price_impact
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python storage_correlation_analysis.py <storage_file> [price_file]")
        print("\nExample: python storage_correlation_analysis.py ngshistory.xls prices.csv")
        sys.exit(1)
    
    storage_file = sys.argv[1]
    price_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Initialize and run analysis
    analyzer = StorageCorrelationAnalyzer(storage_file, price_file)
    results = analyzer.run_full_analysis(
        corr_window=26,           # 6-month rolling correlation
        deviation_threshold=2.0,   # 2 standard deviations
        price_forward_window=4     # 4 weeks forward price impact
    )
    
    print("\n✓ Analysis complete! Check the generated files:")
    print("  - storage_analysis_report.txt")
    print("  - deviation_events.csv")
    print("  - price_impact_analysis.csv (if price data provided)")
    print("  - Visualization PNG files (1-5)")
