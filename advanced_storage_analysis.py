#!/usr/bin/env python3
"""
Advanced Statistical Analysis Module
Provides cointegration testing, Granger causality, and regime detection
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import coint, grangercausalitytests, adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Advanced tests will be limited.")
    print("Install with: pip install statsmodels")


class AdvancedStorageAnalysis:
    """
    Advanced statistical methods for storage relationship analysis
    """
    
    def __init__(self, storage_data, price_data=None):
        """
        Initialize with storage and optional price data
        
        Parameters:
        -----------
        storage_data : pd.DataFrame
            DataFrame with regions as columns, dates as index
        price_data : pd.DataFrame, optional
            DataFrame with price data
        """
        self.storage_data = storage_data
        self.price_data = price_data
        self.results = {}
    
    def test_cointegration(self, region1, region2, significance=0.05):
        """
        Test if two regions are cointegrated (share long-run equilibrium)
        
        Parameters:
        -----------
        region1, region2 : str
            Region names to test
        significance : float
            Significance level for test
            
        Returns:
        --------
        dict : Test results and interpretation
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        print(f"\nTesting cointegration: {region1} vs {region2}")
        
        # Get data
        data1 = self.storage_data[region1].dropna()
        data2 = self.storage_data[region2].dropna()
        
        # Align data
        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx]
        data2 = data2.loc[common_idx]
        
        # Engle-Granger test
        score, pvalue, _ = coint(data1, data2)
        
        # Interpretation
        is_cointegrated = pvalue < significance
        
        result = {
            'region1': region1,
            'region2': region2,
            'test_statistic': score,
            'p_value': pvalue,
            'is_cointegrated': is_cointegrated,
            'interpretation': (
                f"Regions ARE cointegrated (p={pvalue:.4f} < {significance})" if is_cointegrated
                else f"Regions NOT cointegrated (p={pvalue:.4f} >= {significance})"
            )
        }
        
        print(f"  p-value: {pvalue:.4f}")
        print(f"  Result: {result['interpretation']}")
        
        return result
    
    def test_all_cointegration_pairs(self, regions=None, significance=0.05):
        """
        Test cointegration for all region pairs
        
        Parameters:
        -----------
        regions : list, optional
            Specific regions to test (default: all)
        significance : float
            Significance level
            
        Returns:
        --------
        pd.DataFrame : Results for all pairs
        """
        if not STATSMODELS_AVAILABLE:
            return pd.DataFrame()
        
        if regions is None:
            regions = self.storage_data.columns.tolist()
        
        print("\nTesting cointegration for all region pairs...")
        
        results = []
        for i, r1 in enumerate(regions):
            for r2 in regions[i+1:]:
                result = self.test_cointegration(r1, r2, significance)
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Summary
        n_cointegrated = df['is_cointegrated'].sum()
        print(f"\nSummary: {n_cointegrated}/{len(df)} pairs are cointegrated")
        
        return df
    
    def test_granger_causality(self, cause_region, effect_region, max_lag=4):
        """
        Test if one region's storage changes "Granger-cause" another's
        
        Parameters:
        -----------
        cause_region : str
            Potential cause region
        effect_region : str
            Potential effect region
        max_lag : int
            Maximum lag to test
            
        Returns:
        --------
        dict : Test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        print(f"\nGranger Causality Test: {cause_region} → {effect_region}")
        
        # Prepare data (need changes/differences for stationarity)
        data = pd.DataFrame({
            'effect': self.storage_data[effect_region].diff().dropna(),
            'cause': self.storage_data[cause_region].diff().dropna()
        }).dropna()
        
        # Run test
        try:
            test_result = grangercausalitytests(data[['effect', 'cause']], max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, max_lag + 1):
                # Use F-test p-value
                p_values[lag] = test_result[lag][0]['ssr_ftest'][1]
            
            # Check if any lag is significant
            min_pval = min(p_values.values())
            is_causal = min_pval < 0.05
            
            result = {
                'cause': cause_region,
                'effect': effect_region,
                'p_values_by_lag': p_values,
                'min_p_value': min_pval,
                'is_granger_causal': is_causal,
                'interpretation': (
                    f"{cause_region} DOES Granger-cause {effect_region} (p={min_pval:.4f})"
                    if is_causal else
                    f"{cause_region} does NOT Granger-cause {effect_region} (p={min_pval:.4f})"
                )
            }
            
            print(f"  Min p-value: {min_pval:.4f}")
            print(f"  Result: {result['interpretation']}")
            
            return result
            
        except Exception as e:
            print(f"  Error: {e}")
            return {"error": str(e)}
    
    def detect_structural_breaks(self, region, window=52):
        """
        Detect structural breaks in storage patterns using rolling statistics
        
        Parameters:
        -----------
        region : str
            Region to analyze
        window : int
            Window size for detecting changes
            
        Returns:
        --------
        pd.DataFrame : Detected break points
        """
        print(f"\nDetecting structural breaks for {region}...")
        
        data = self.storage_data[region].dropna()
        
        # Calculate rolling mean and std
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Detect significant shifts
        mean_changes = rolling_mean.diff().abs()
        std_changes = rolling_std.diff().abs()
        
        # Normalize changes
        mean_z = (mean_changes - mean_changes.mean()) / mean_changes.std()
        std_z = (std_changes - std_changes.mean()) / std_changes.std()
        
        # Flag breaks (mean or volatility shifts)
        breaks = data.index[(mean_z > 3) | (std_z > 3)]
        
        results = pd.DataFrame({
            'Date': breaks,
            'Region': region,
            'Type': 'Structural Break'
        })
        
        print(f"  Found {len(results)} potential structural breaks")
        
        return results
    
    def analyze_spread_behavior(self, region1, region2, threshold_pct=10):
        """
        Analyze the spread (difference) between two regions
        
        Parameters:
        -----------
        region1, region2 : str
            Regions to compare
        threshold_pct : float
            Percentage threshold for flagging wide spreads
            
        Returns:
        --------
        dict : Spread statistics and events
        """
        print(f"\nAnalyzing spread: {region1} - {region2}")
        
        data1 = self.storage_data[region1]
        data2 = self.storage_data[region2]
        
        # Calculate spread
        spread = data1 - data2
        spread_pct = (spread / data2) * 100
        
        # Statistics
        stats_dict = {
            'mean_spread': spread.mean(),
            'std_spread': spread.std(),
            'mean_spread_pct': spread_pct.mean(),
            'std_spread_pct': spread_pct.std(),
            'max_spread': spread.max(),
            'min_spread': spread.min(),
            'max_spread_date': spread.idxmax(),
            'min_spread_date': spread.idxmin()
        }
        
        # Detect wide spread events
        wide_spreads = spread_pct[abs(spread_pct - spread_pct.mean()) > threshold_pct]
        
        events = pd.DataFrame({
            'Date': wide_spreads.index,
            'Spread_Pct': wide_spreads.values,
            'Pair': f"{region1} - {region2}"
        })
        
        print(f"  Mean spread: {stats_dict['mean_spread']:.1f} BCF ({stats_dict['mean_spread_pct']:.1f}%)")
        print(f"  Wide spread events: {len(events)}")
        
        return {
            'statistics': stats_dict,
            'wide_spread_events': events,
            'spread_series': spread,
            'spread_pct_series': spread_pct
        }
    
    def test_mean_reversion(self, region1, region2, half_life_window=52):
        """
        Test if the spread between two regions is mean-reverting
        
        Parameters:
        -----------
        region1, region2 : str
            Regions to test
        half_life_window : int
            Window for calculating half-life
            
        Returns:
        --------
        dict : Mean reversion test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
        
        print(f"\nTesting mean reversion: {region1} vs {region2}")
        
        # Calculate spread
        spread = self.storage_data[region1] - self.storage_data[region2]
        spread = spread.dropna()
        
        # Test for stationarity (ADF test)
        adf_result = adfuller(spread, maxlag=12)
        adf_stat, adf_pval = adf_result[0], adf_result[1]
        
        is_stationary = adf_pval < 0.05
        
        # Calculate half-life (if mean-reverting)
        if is_stationary:
            # Fit AR(1): spread[t] = a + b*spread[t-1] + error
            spread_lag = spread.shift(1).dropna()
            spread_current = spread[1:]
            
            # Align
            common_idx = spread_lag.index.intersection(spread_current.index)
            spread_lag = spread_lag.loc[common_idx]
            spread_current = spread_current.loc[common_idx]
            
            # OLS regression
            slope, intercept = np.polyfit(spread_lag, spread_current, 1)
            
            # Half-life = -ln(2) / ln(b)
            if slope > 0 and slope < 1:
                half_life = -np.log(2) / np.log(slope)
            else:
                half_life = np.inf
        else:
            half_life = np.inf
        
        result = {
            'region1': region1,
            'region2': region2,
            'adf_statistic': adf_stat,
            'adf_pvalue': adf_pval,
            'is_mean_reverting': is_stationary,
            'half_life_weeks': half_life,
            'interpretation': (
                f"Spread IS mean-reverting (ADF p={adf_pval:.4f}, half-life={half_life:.1f} weeks)"
                if is_stationary and half_life < np.inf else
                f"Spread NOT mean-reverting (ADF p={adf_pval:.4f})"
            )
        }
        
        print(f"  ADF p-value: {adf_pval:.4f}")
        print(f"  Result: {result['interpretation']}")
        
        return result
    
    def generate_advanced_report(self, output_file='advanced_analysis_report.txt'):
        """Generate report of advanced statistical findings"""
        
        print(f"\nGenerating advanced analysis report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ADVANCED STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            if not STATSMODELS_AVAILABLE:
                f.write("WARNING: Advanced statistical tests require statsmodels package\n")
                f.write("Install with: pip install statsmodels\n\n")
            
            # Add any stored results
            if hasattr(self, 'cointegration_results'):
                f.write("-"*80 + "\n")
                f.write("COINTEGRATION ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                for _, row in self.cointegration_results.iterrows():
                    f.write(f"{row['region1']} vs {row['region2']}: ")
                    f.write(f"{'COINTEGRATED' if row['is_cointegrated'] else 'Not cointegrated'} ")
                    f.write(f"(p={row['p_value']:.4f})\n")
                
                f.write("\n")
            
            if hasattr(self, 'mean_reversion_results'):
                f.write("-"*80 + "\n")
                f.write("MEAN REVERSION ANALYSIS\n")
                f.write("-"*80 + "\n\n")
                
                for result in self.mean_reversion_results:
                    f.write(f"{result['region1']} - {result['region2']}:\n")
                    f.write(f"  {result['interpretation']}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"Advanced report saved to: {output_file}")
        return output_file


def run_advanced_analysis(storage_data, price_data=None):
    """
    Convenience function to run full advanced analysis suite
    
    Parameters:
    -----------
    storage_data : pd.DataFrame
        Storage data with regions as columns
    price_data : pd.DataFrame, optional
        Price data
        
    Returns:
    --------
    AdvancedStorageAnalysis : Analysis object with results
    """
    analyzer = AdvancedStorageAnalysis(storage_data, price_data)
    
    regions = storage_data.columns.tolist()
    
    # Run cointegration tests
    if STATSMODELS_AVAILABLE and len(regions) > 1:
        analyzer.cointegration_results = analyzer.test_all_cointegration_pairs(regions)
        
        # Test mean reversion for cointegrated pairs
        cointegrated_pairs = analyzer.cointegration_results[
            analyzer.cointegration_results['is_cointegrated']
        ]
        
        analyzer.mean_reversion_results = []
        for _, row in cointegrated_pairs.iterrows():
            result = analyzer.test_mean_reversion(row['region1'], row['region2'])
            analyzer.mean_reversion_results.append(result)
        
        # Test Granger causality for key pairs
        print("\n" + "="*80)
        print("Testing Granger Causality (Selected Pairs)")
        print("="*80)
        
        # Test a few key relationships
        if 'Total Lower 48' in regions and 'East' in regions:
            analyzer.test_granger_causality('Total Lower 48', 'East')
        if 'South Central' in regions and 'Total Lower 48' in regions:
            analyzer.test_granger_causality('South Central', 'Total Lower 48')
    
    # Detect structural breaks
    print("\n" + "="*80)
    print("Detecting Structural Breaks")
    print("="*80)
    
    all_breaks = []
    for region in regions[:3]:  # Test first 3 regions to avoid too much output
        breaks = analyzer.detect_structural_breaks(region)
        all_breaks.append(breaks)
    
    if all_breaks:
        analyzer.structural_breaks = pd.concat(all_breaks, ignore_index=True)
    
    # Generate report
    analyzer.generate_advanced_report()
    
    return analyzer


# Example usage
if __name__ == "__main__":
    print("This module provides advanced statistical analysis functions.")
    print("Import it into your main analysis script:")
    print("\n  from advanced_storage_analysis import AdvancedStorageAnalysis, run_advanced_analysis")
    print("\n  # After loading your storage data:")
    print("  advanced = run_advanced_analysis(storage_data, price_data)")
