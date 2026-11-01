
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]
df = pd.read_csv(DATA_PATH)

# Remove non-feature columns and exact duplicates.
df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])]
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_frame(TARGET).to_csv("ytrain.csv", index=False)
y_test.to_frame(TARGET).to_csv("ytest.csv", index=False)
print(f"Prepared rows: {len(df)}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train target ratio:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Test target ratio:", y_test.value_counts(normalize=True).round(3).to_dict())
