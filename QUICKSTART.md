# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Prepare Your Data

You need the EIA natural gas storage Excel file. You can download it from:
https://www.eia.gov/naturalgas/storage/

**Option A: If you have price data**
```bash
python prepare_price_data.py your_prices.csv
```

**Option B: Create sample price data for testing**
```bash
python prepare_price_data.py sample
```

### Step 2: Run the Analysis

**Basic Analysis (storage data only):**
```bash
python storage_correlation_analysis.py ngshistory.xls
```

**Full Analysis (with price data):**
```bash
python storage_correlation_analysis.py ngshistory.xls prepared_prices.csv
```

**Or use the example script:**
```bash
python example_usage.py ngshistory.xls prepared_prices.csv
```

### Step 3: Review Your Results

The analysis generates several files:

📄 **Reports:**
- `storage_analysis_report.txt` - Comprehensive text summary
- `advanced_analysis_report.txt` - Statistical test results

📊 **Visualizations:**
- `1_correlation_heatmap.png` - Current regional correlations
- `2_rolling_correlations.png` - How correlations changed over time
- `3_storage_ratios.png` - Regional balance trends
- `4_deviation_events.png` - Timeline of unusual events
- `5_price_impact_analysis.png` - Price impact of deviations

📈 **Data Files:**
- `deviation_events.csv` - All detected anomalies
- `price_impact_analysis.csv` - Price movements after deviations

## 🎯 What To Look For

### In the Correlation Heatmap:
- **Dark red** = Strong positive correlation (regions move together)
- **Dark blue** = Negative/weak correlation
- **Unusual patterns** = Potential trading opportunities

### In Rolling Correlations:
- **Stable lines** = Consistent relationship
- **Sudden drops** = Relationship breakdown
- **Below orange bands** = Deviation events

### In Storage Ratios:
- **Inside shaded area** = Normal range
- **Outside shaded area** = Deviation event
- **Trending away** = Structural change

### In Price Impact Analysis:
- **Top-left panel**: Does deviation size predict price change?
- **Top-right panel**: What's the typical price response?
- **Bottom-left panel**: Which event types matter most?
- **Bottom-right panel**: Do deviations increase volatility?

## 💡 Common Use Cases

### Finding Trading Opportunities
Look for:
1. Cointegrated pairs with wide spreads (mean reversion trades)
2. Regions that Granger-cause others (leading indicators)
3. Deviation events followed by large price moves

### Risk Management
Monitor:
1. When correlations break down (diversification failure)
2. Structural breaks in regional relationships
3. High z-score deviations (potential volatility ahead)

### Market Analysis
Analyze:
1. Which regions drive the Total Lower 48
2. Salt vs Non-Salt divergence patterns
3. Seasonal correlation changes

## 🔧 Customization Examples

### Adjust Sensitivity
```python
# In storage_correlation_analysis.py, modify:
analyzer.run_full_analysis(
    corr_window=52,          # Longer window = smoother
    deviation_threshold=3.0,  # Higher = fewer but stronger signals
    price_forward_window=8    # Longer = delayed price impact
)
```

### Focus on Specific Regions
```python
# Calculate only specific ratios:
analyzer.calculate_regional_ratios(ratio_pairs=[
    ('Salt', 'Non Salt'),
    ('East', 'Total Lower 48')
])
```

### Different Time Periods
```python
# Filter data before analysis:
recent_data = storage_data[storage_data.index >= '2020-01-01']
analyzer.storage_data = recent_data
```

## ❓ Troubleshooting

**"Could not find header row"**
→ Make sure your Excel file has standard EIA format with "Week Ending" in the header

**"No price data"**
→ Price data is optional. The analysis will still run without it.

**"statsmodels not available"**
→ For advanced tests: `pip install statsmodels`

**Charts not displaying**
→ Make sure matplotlib is installed: `pip install matplotlib seaborn`

## 📚 Learn More

- **Full documentation**: See README.md
- **Methodology details**: Check comments in storage_correlation_analysis.py
- **Advanced features**: Review advanced_storage_analysis.py

## 🆘 Need Help?

1. Read the error message carefully
2. Check the README.md for detailed explanations
3. Verify your input file formats
4. Make sure all required packages are installed

---

**Ready to go deeper?** Check out the full README.md for advanced analysis techniques!
