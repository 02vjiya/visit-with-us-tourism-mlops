
from pathlib import Path
import pandas as pd

DATA_PATH = Path("tourism_project/data/tourism.csv")
EXPECTED = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
df = pd.read_csv(DATA_PATH)
missing = sorted(set(EXPECTED) - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if "ProdTaken" not in df or df["ProdTaken"].isna().any():
    raise ValueError("Target ProdTaken is missing or contains null values.")
if not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain only 0 and 1.")
print(f"Dataset registered: {DATA_PATH}")
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print("Target distribution:")
print(df["ProdTaken"].value_counts(dropna=False).sort_index())
print("Missing values (top):")
print(df.isna().sum().sort_values(ascending=False).head(10))
