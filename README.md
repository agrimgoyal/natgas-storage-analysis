# Natural Gas Storage Regional Correlation Analysis

A comprehensive Python-based analytical toolkit for detecting and analyzing correlations, deviations, and relationships between natural gas storage regions, with integrated price impact analysis.

## 🎯 Purpose

This tool helps answer critical questions:
1. **Do storage levels across regions move together?**
2. **At what ratios do they typically move?**
3. **When do these relationships break down?**
4. **What do these breakdowns tell us about price movements?**

## 📋 Features

### Core Analysis (`storage_correlation_analysis.py`)
- **Rolling Correlation Analysis**: Track how regional correlations evolve over time
- **Ratio Analysis**: Monitor storage level ratios between regions
- **Deviation Detection**: Identify when correlations/ratios break historical patterns
- **Price Impact Analysis**: Correlate deviation events with price movements
- **Automated Reporting**: Generate comprehensive text reports
- **Rich Visualizations**: Create 5+ publication-ready charts

### Advanced Analysis (`advanced_storage_analysis.py`)
- **Cointegration Testing**: Determine if regions share long-term equilibrium relationships
- **Granger Causality**: Test if one region's changes predict another's
- **Mean Reversion Testing**: Check if regional spreads revert to historical means
- **Structural Break Detection**: Identify regime changes in storage patterns
- **Spread Behavior Analysis**: Deep dive into regional differentials

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install pandas numpy scipy matplotlib seaborn openpyxl xlrd

# For advanced statistical tests (optional but recommended)
pip install statsmodels
```

### Basic Usage

```bash
# With storage data only
python storage_correlation_analysis.py ngshistory.xls

# With storage and price data
python storage_correlation_analysis.py ngshistory.xls henry_hub_prices.csv
```

### Expected Output Files

1. **storage_analysis_report.txt** - Comprehensive text report
2. **deviation_events.csv** - All detected deviation events with details
3. **price_impact_analysis.csv** - Price movements following deviations (if price data provided)
4. **1_correlation_heatmap.png** - Current correlation matrix visualization
5. **2_rolling_correlations.png** - Time series of correlations for all pairs
6. **3_storage_ratios.png** - Regional ratio trends with deviation bands
7. **4_deviation_events.png** - Timeline of all deviation events
8. **5_price_impact_analysis.png** - 4-panel price impact visualization

## 📊 Data Requirements

### Storage Data (Required)
- **Format**: Excel file (.xls or .xlsx)
- **Source**: EIA Natural Gas Storage data
- **Expected Structure**:
  - Header row containing "Week Ending" and region names
  - Columns for: Total Lower 48, East, Midwest, Mountain, Pacific, South Central, Salt, Non-Salt
  - Weekly data with dates and storage levels in BCF

### Price Data (Optional but Recommended)
- **Format**: CSV or Excel
- **Required Columns**: 
  - Date column (any common date format)
  - Price column (numeric values)
- **Suggested Sources**:
  - Henry Hub Natural Gas prices
  - Regional basis differentials
  - Futures settlement prices

## 🔧 Customization

### Adjusting Analysis Parameters

```python
from storage_correlation_analysis import StorageCorrelationAnalyzer

# Initialize
analyzer = StorageCorrelationAnalyzer('ngshistory.xls', 'prices.csv')

# Run with custom parameters
results = analyzer.run_full_analysis(
    corr_window=52,           # 1-year rolling correlation window (default: 26)
    deviation_threshold=3.0,   # 3 std dev threshold (default: 2.0)
    price_forward_window=8,    # 8 weeks forward price impact (default: 4)
    min_history=52             # Min weeks before flagging events (default: 52)
)
```

### Custom Ratio Pairs

```python
# Calculate specific ratios
analyzer.load_storage_data()
analyzer.calculate_regional_ratios(ratio_pairs=[
    ('East', 'Midwest'),
    ('Mountain', 'Pacific'),
    ('Salt', 'Total Lower 48')
])
```

## 📈 Understanding the Output

### Correlation Analysis
- **High correlation (>0.8)**: Regions move together consistently
- **Medium correlation (0.5-0.8)**: Moderate relationship
- **Low correlation (<0.5)**: Independent movement patterns
- **Correlation breaks**: When correlation drops >2 std dev below mean

### Ratio Analysis
- **Stable ratios**: Low coefficient of variation (<10%)
- **Variable ratios**: High CV (>20%) suggests changing relationships
- **Deviation events**: When ratios exceed ±2 standard deviations

### Price Impact Interpretation
- **Positive correlation**: Larger deviations → larger price moves
- **Event clustering**: Multiple deviations before major price movements
- **Directional bias**: Check if deviations predict price increases/decreases

## 🔬 Advanced Analysis

### Running Advanced Tests

```python
from storage_correlation_analysis import StorageCorrelationAnalyzer
from advanced_storage_analysis import run_advanced_analysis

# Load data
analyzer = StorageCorrelationAnalyzer('ngshistory.xls')
analyzer.load_storage_data()

# Run advanced statistical tests
advanced = run_advanced_analysis(
    analyzer.storage_data,
    analyzer.price_data
)

# Access specific results
print(advanced.cointegration_results)
print(advanced.mean_reversion_results)
```

### Interpreting Advanced Results

**Cointegration**
- **Cointegrated regions**: Share long-term equilibrium (deviations temporary)
- **Not cointegrated**: Can drift apart indefinitely
- **Trading implication**: Cointegrated spreads may be mean-reverting

**Granger Causality**
- **A Granger-causes B**: A's past values help predict B
- **Bidirectional**: Both regions predict each other (feedback loop)
- **No causality**: Independent movements

**Mean Reversion**
- **Half-life < 10 weeks**: Fast mean reversion (tactical opportunity)
- **Half-life 10-30 weeks**: Moderate reversion
- **Half-life > 30 weeks**: Slow/no reversion

## 💡 Analysis Workflow Example

### Step 1: Identify Stable Relationships
```bash
python storage_correlation_analysis.py ngshistory.xls
```
Review `storage_analysis_report.txt` for correlation stability

### Step 2: Detect Historical Deviations
Review `deviation_events.csv` for:
- Frequency of deviations
- Which region pairs deviate most
- Seasonal patterns in deviations

### Step 3: Analyze Price Impact
Review `price_impact_analysis.csv` for:
- Average price change after deviations
- Correlation between deviation magnitude and price movement
- Event types with strongest price impact

### Step 4: Test Statistical Relationships
```python
from advanced_storage_analysis import run_advanced_analysis
advanced = run_advanced_analysis(storage_data, price_data)
```

### Step 5: Generate Trading Insights
Look for:
- **Pairs trading opportunities**: Cointegrated regions with wide spreads
- **Leading indicators**: Regions that Granger-cause others
- **Regime changes**: Structural breaks preceding price volatility
- **Reversal signals**: Mean-reverting spreads at extremes

## 📊 Visualization Guide

### Chart 1: Correlation Heatmap
- **Purpose**: See current state of all regional relationships
- **Use**: Identify unusual correlation patterns
- **Red**: Strong positive correlation
- **Blue**: Negative/weak correlation

### Chart 2: Rolling Correlations
- **Purpose**: Track how relationships change over time
- **Use**: Identify when correlations strengthened/weakened
- **Orange bands**: ±2 standard deviation thresholds

### Chart 3: Storage Ratios
- **Purpose**: Monitor regional balance over time
- **Use**: Spot ratio deviations from historical norms
- **Shaded area**: Normal range (±2 std dev)

### Chart 4: Deviation Events
- **Purpose**: Timeline of all significant events
- **Use**: Check if deviations cluster before major moves
- **Y-axis**: Magnitude of deviation (|Z-score|)

### Chart 5: Price Impact Analysis
- **Panel A**: Deviation size vs price change
- **Panel B**: Distribution of price changes after events
- **Panel C**: Which event types cause biggest moves
- **Panel D**: Deviation size vs price volatility

## 🎓 Methodological Notes

### Correlation Calculation
- Uses Pearson correlation on rolling windows
- Window size: 26 weeks (6 months) default
- Robust to missing data via pairwise deletion

### Deviation Detection
- **Expanding Window Methodology**: Avoids look-ahead bias by using only data available up to each observation
- Z-score methodology: (value - expanding_mean) / expanding_std
- Minimum 52 weeks of history required before flagging events
- Threshold: 2 standard deviations default
- Suitable for live trading applications

### Price Impact Measurement
- Forward-looking: Price change in N weeks after event
- Volatility: Standard deviation during window
- Controls for general market moves via comparison

### Statistical Tests (Advanced Module)
- **Cointegration**: Engle-Granger two-step method
- **Granger Causality**: F-test on vector autoregression
- **Stationarity**: Augmented Dickey-Fuller test
- **Mean Reversion**: Half-life from AR(1) model

## 🛠️ Troubleshooting

### "Could not find header row"
- Ensure Excel file has "Week Ending" in header
- Check that data starts in first few rows

### "No correlation data"
- Need at least `corr_window` weeks of data
- Check for gaps in time series

### "Price data not loading"
- Verify date column format
- Ensure price column is numeric
- Check file path

### "statsmodels not available"
- Advanced tests require: `pip install statsmodels`
- Basic analysis works without it

## 📚 Further Reading

- EIA Natural Gas Storage: https://www.eia.gov/naturalgas/storage/
- Cointegration: Engle & Granger (1987)
- Granger Causality: Granger (1969)
- Trading Strategies: Pairs Trading literature

## 📝 Citation

If you use this tool for research or trading:
```
Natural Gas Storage Regional Correlation Analyzer (2026)
Advanced statistical analysis of EIA natural gas storage data
```

## 🤝 Contributing

Suggestions for improvement:
1. Additional statistical tests
2. Real-time data integration
3. Machine learning models
4. Interactive dashboards

## ⚠️ Disclaimer

This tool is for informational and analytical purposes only. Not financial advice. 
Past correlations and patterns do not guarantee future relationships. 
Always verify findings and consult professionals before making trading decisions.

## 📧 Support

For questions about:
- **Data formats**: Check EIA documentation
- **Statistical methods**: Review methodology notes above
- **Custom analysis**: Modify source code or contact developer

---

**Version**: 2.0
**Last Updated**: February 2026
**Python Version**: 3.8+

### Changelog
- **v2.0**: Implemented expanding window methodology to eliminate look-ahead bias
  - Z-scores now calculated using only historical data available at each point
  - Minimum 52 weeks of history required before flagging events
  - See `methodology_comparison.txt` for detailed analysis of changes
