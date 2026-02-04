#!/usr/bin/env python3
"""
Example: Complete Natural Gas Storage Analysis
Demonstrates full workflow from data loading to insights
"""

from storage_correlation_analysis import StorageCorrelationAnalyzer
from advanced_storage_analysis import run_advanced_analysis
import pandas as pd

def main():
    print("="*80)
    print("NATURAL GAS STORAGE ANALYSIS - COMPLETE EXAMPLE")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Initialize and Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    # Replace with your actual file paths
    STORAGE_FILE = 'ngshistory.xls'  # Your EIA storage data
    PRICE_FILE = 'prices.csv'         # Your price data (optional)
    
    # Initialize analyzer
    analyzer = StorageCorrelationAnalyzer(STORAGE_FILE, PRICE_FILE)
    
    # Load data
    storage_data = analyzer.load_storage_data()
    price_data = analyzer.load_price_data()
    
    print(f"\n✓ Loaded {len(storage_data)} weeks of storage data")
    print(f"✓ Regions: {', '.join(analyzer.regions)}")
    
    # ========================================================================
    # STEP 2: Basic Correlation & Ratio Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Calculating Correlations and Ratios")
    print("="*80)
    
    # Calculate rolling correlations (26-week window = ~6 months)
    correlations = analyzer.calculate_rolling_correlations(window=26)
    
    # Calculate key regional ratios
    ratios = analyzer.calculate_regional_ratios()
    
    print(f"\n✓ Calculated {len(correlations)} correlation pairs")
    print(f"✓ Calculated {len(ratios)} regional ratios")
    
    # Show current correlation status
    print("\nCurrent Correlation Snapshot:")
    for pair, series in list(correlations.items())[:3]:  # Show first 3
        current_corr = series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
        if current_corr is not None:
            print(f"  {pair}: {current_corr:.3f}")
    
    # ========================================================================
    # STEP 3: Detect Deviation Events
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Detecting Deviation Events")
    print("="*80)
    
    # Detect correlation breaks (when correlations drop significantly)
    # Using expanding window with min 52 weeks history to avoid look-ahead bias
    corr_breaks = analyzer.detect_correlation_breaks(threshold_std=2.0, min_history=52)

    # Detect ratio deviations (when ratios exceed historical ranges)
    # Using expanding window with min 52 weeks history to avoid look-ahead bias
    ratio_devs = analyzer.detect_ratio_deviations(threshold_std=2.0, min_history=52)
    
    # Combine all events
    all_deviations = pd.concat([corr_breaks, ratio_devs], ignore_index=True)
    all_deviations = all_deviations.sort_values('Date')
    analyzer.all_deviations = all_deviations
    
    print(f"\n✓ Found {len(corr_breaks)} correlation break events")
    print(f"✓ Found {len(ratio_devs)} ratio deviation events")
    print(f"✓ Total: {len(all_deviations)} deviation events")
    
    # Show most recent events
    if len(all_deviations) > 0:
        print("\nMost Recent Deviation Events:")
        recent = all_deviations.tail(5)
        for idx, event in recent.iterrows():
            print(f"  {event['Date'].strftime('%Y-%m-%d')}: {event['Type']} - {event['Pair']}")
            print(f"    Z-Score: {event['Z-Score']:.2f}")
    
    # ========================================================================
    # STEP 4: Analyze Price Impact
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Analyzing Price Impact")
    print("="*80)
    
    if price_data is not None and len(all_deviations) > 0:
        # Analyze what happened to prices after deviations
        price_impact = analyzer.analyze_price_impact(
            all_deviations, 
            forward_window=4  # Look 4 weeks forward
        )
        analyzer.price_impact = price_impact
        
        if price_impact is not None and len(price_impact) > 0:
            print(f"\n✓ Analyzed {len(price_impact)} events with price data")
            
            # Summary statistics
            avg_change = price_impact['Price_Change_Pct'].mean()
            median_change = price_impact['Price_Change_Pct'].median()
            
            print(f"\nPrice Impact Summary:")
            print(f"  Average price change: {avg_change:.2f}%")
            print(f"  Median price change: {median_change:.2f}%")
            
            # Breakdown by event type
            print("\nBy Event Type:")
            for event_type in price_impact['Event_Type'].unique():
                subset = price_impact[price_impact['Event_Type'] == event_type]
                avg = subset['Price_Change_Pct'].mean()
                print(f"  {event_type}: {avg:.2f}% avg change ({len(subset)} events)")
            
            # Check correlation between deviation magnitude and price change
            corr = price_impact['Event_ZScore'].abs().corr(
                price_impact['Price_Change_Pct'].abs()
            )
            print(f"\nDeviation Magnitude vs Price Change Correlation: {corr:.3f}")
            
            if corr > 0.3:
                print("  → Larger deviations tend to cause larger price moves")
            elif corr < -0.3:
                print("  → Inverse relationship detected")
            else:
                print("  → Weak relationship between deviation size and price impact")
    else:
        print("\n! No price data available or no deviation events to analyze")
    
    # ========================================================================
    # STEP 5: Run Advanced Statistical Tests
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Advanced Statistical Analysis")
    print("="*80)
    
    try:
        # Run comprehensive advanced analysis
        advanced = run_advanced_analysis(storage_data, price_data)
        
        # Display cointegration results
        if hasattr(advanced, 'cointegration_results'):
            print("\nCointegration Summary:")
            coint_df = advanced.cointegration_results
            cointegrated = coint_df[coint_df['is_cointegrated']]
            print(f"  {len(cointegrated)}/{len(coint_df)} region pairs are cointegrated")
            
            if len(cointegrated) > 0:
                print("\n  Cointegrated Pairs:")
                for _, row in cointegrated.iterrows():
                    print(f"    {row['region1']} ↔ {row['region2']} (p={row['p_value']:.4f})")
        
        # Display mean reversion results
        if hasattr(advanced, 'mean_reversion_results'):
            print("\nMean Reversion Analysis:")
            for result in advanced.mean_reversion_results[:3]:  # Show first 3
                print(f"  {result['region1']} - {result['region2']}:")
                if result['is_mean_reverting']:
                    print(f"    Mean-reverting (half-life: {result['half_life_weeks']:.1f} weeks)")
                else:
                    print(f"    Not mean-reverting")
    
    except Exception as e:
        print(f"\n! Advanced analysis error: {e}")
        print("  (Advanced tests require 'statsmodels' package)")
    
    # ========================================================================
    # STEP 6: Generate Reports and Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Generating Outputs")
    print("="*80)
    
    # Generate comprehensive report
    report_file = analyzer.generate_report()
    print(f"\n✓ Generated text report: {report_file}")
    
    # Create visualizations
    num_charts = analyzer.create_visualizations()
    print(f"✓ Created {num_charts} visualization files")
    
    # Save data files
    if len(all_deviations) > 0:
        all_deviations.to_csv('deviation_events.csv', index=False)
        print("✓ Saved deviation events: deviation_events.csv")
    
    if analyzer.price_impact is not None and len(analyzer.price_impact) > 0:
        analyzer.price_impact.to_csv('price_impact_analysis.csv', index=False)
        print("✓ Saved price impact analysis: price_impact_analysis.csv")
    
    # ========================================================================
    # STEP 7: Key Insights Summary
    # ========================================================================
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find most correlated regions
    if correlations:
        avg_corrs = {pair: series.mean() for pair, series in correlations.items()}
        top_pair = max(avg_corrs, key=avg_corrs.get)
        print(f"\n1. Strongest Regional Correlation:")
        print(f"   {top_pair}: {avg_corrs[top_pair]:.3f} average")
    
    # Most volatile ratio
    if ratios:
        cvs = {pair: (series.std() / series.mean() * 100) 
               for pair, series in ratios.items()}
        most_volatile = max(cvs, key=cvs.get)
        print(f"\n2. Most Variable Regional Ratio:")
        print(f"   {most_volatile}: CV = {cvs[most_volatile]:.1f}%")
    
    # Price impact insight
    if analyzer.price_impact is not None and len(analyzer.price_impact) > 0:
        significant = analyzer.price_impact[
            abs(analyzer.price_impact['Price_Change_Pct']) > 5
        ]
        print(f"\n3. Significant Price Movements:")
        print(f"   {len(significant)}/{len(analyzer.price_impact)} events caused >5% price moves")
    
    # Cointegration insight
    if hasattr(advanced, 'cointegration_results'):
        coint_pairs = advanced.cointegration_results[
            advanced.cointegration_results['is_cointegrated']
        ]
        print(f"\n4. Long-term Relationships:")
        print(f"   {len(coint_pairs)} region pairs show cointegration")
        print(f"   (These pairs tend to revert when they diverge)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  📄 storage_analysis_report.txt")
    print("  📊 1_correlation_heatmap.png")
    print("  📊 2_rolling_correlations.png")
    print("  📊 3_storage_ratios.png")
    print("  📊 4_deviation_events.png")
    print("  📊 5_price_impact_analysis.png")
    print("  📈 deviation_events.csv")
    print("  📈 price_impact_analysis.csv")
    print("  📄 advanced_analysis_report.txt")
    print("\n")


if __name__ == "__main__":
    # Check if files exist
    import os
    import sys
    
    if len(sys.argv) >= 2:
        STORAGE_FILE = sys.argv[1]
        PRICE_FILE = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        STORAGE_FILE = 'ngshistory.xls'
        PRICE_FILE = None
    
    if not os.path.exists(STORAGE_FILE):
        print(f"Error: Storage file '{STORAGE_FILE}' not found!")
        print("\nUsage: python example_usage.py <storage_file> [price_file]")
        print("\nExample: python example_usage.py ngshistory.xls prices.csv")
        sys.exit(1)
    
    # Update the file paths in main
    import storage_correlation_analysis
    
    # Run the analysis
    main()
