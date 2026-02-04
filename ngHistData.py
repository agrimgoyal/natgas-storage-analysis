import requests
import pandas as pd
import urllib3
from datetime import datetime, timedelta

# Disable SSL warnings (since verify=False is used)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

month_codes = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}

# Generate qhcode for next 32 months starting from next month
qhcodes = []
today = datetime.today()
for i in range(1, 31):  # 1 to 32 months ahead
    future = today.replace(day=1) + pd.DateOffset(months=i)
    code = f"NG{month_codes[future.month]}{str(future.year)[-2:]}"
    qhcodes.append(code)
qhcode_str = ",".join(qhcodes)

# Define the API endpoint
url = "https://qh-api.corp.hertshtengroup.com/api/v2/ohlc/"

# Set the parameters
params = {"instruments": f"NG{month_codes}24", "interval": "1H"}

headers = {
    "Authorization": "Bearer your_token",
}

# Make the request
response = requests.get(url, headers=headers, params=params, verify=False)
print(qhcode_str)

if response.status_code == 200:
    data = response.json()
    results = data.get("results", [])
    if results:
        df = pd.DataFrame(results)
        # Convert datetime to date only
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        # Pivot the DataFrame
        pivot_df = df.pivot(index="instruments", columns="date", values="close")
        # Forward fill then backward fill across columns (dates)
        pivot_df = pivot_df.ffill(axis=1).bfill(axis=1)
        # Reverse the columns so latest date comes first
        pivot_df = pivot_df[pivot_df.columns[::-1]]

        print(pivot_df)
    else:
        print("No results found in response.")
else:
    print(f"Request failed with status code {response.status_code}")
