import pandas as pd

# Load dataset
df = pd.read_csv("vaayu_final_ml_ready.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))

print("\nMissing Percentage:")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0]
print(missing_pct.sort_values(ascending=False))

print("\nYear Distribution:")
print(df["year"].value_counts().sort_index())

print("\nNumeric Statistics:")
print(df.describe())

print("\nNO2 Statistics:")
print(df["no2"].describe())

print("\nO3 Statistics:")
print(df["o3"].describe())

print("\nUnique Stations:")
print(df["station_id"].nunique())

print("\nTop 10 Stations:")
print(df["station_name"].value_counts().head(10))

print("\nDate Range:")
if "datetime" in df.columns:
    print(pd.to_datetime(df["datetime"]).min())
    print(pd.to_datetime(df["datetime"]).max())

print("\nDataset Memory Usage:")
print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
