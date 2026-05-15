# =====================================
# Farmer Mistake Detector - Training
# =====================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Loading datasets...")

# -------------------------------
# 1. Load Data
# -------------------------------
crop_df = pd.read_csv("data/crop_production.csv")
rain_df = pd.read_csv("data/rainfall in india 1901-2015.csv")

crop_df.columns = crop_df.columns.str.strip()
rain_df.columns = rain_df.columns.str.strip()

# -------------------------------
# 2. Feature Engineering
# -------------------------------
crop_df["Yield"] = crop_df["Production"] / crop_df["Area"]
crop_df.replace([np.inf, -np.inf], np.nan, inplace=True)
crop_df.dropna(inplace=True)

crop_df["State_Name"]    = crop_df["State_Name"].str.upper().str.strip()
crop_df["District_Name"] = crop_df["District_Name"].str.upper().str.strip()

# -------------------------------
# 3. Aggregate Crop Data
# -------------------------------
crop_grouped = crop_df.groupby(
    ["State_Name", "District_Name", "Crop_Year", "Crop"]
).agg({
    "Area":       "sum",
    "Production": "sum",
    "Yield":      "mean"
}).reset_index()

# -------------------------------
# 4. Rainfall Preparation
# -------------------------------
rain_df = rain_df[["SUBDIVISION", "YEAR", "ANNUAL"]]
rain_df["SUBDIVISION"] = rain_df["SUBDIVISION"].str.upper().str.strip()

subdivision_to_states = {
    "ANDAMAN & NICOBAR ISLANDS":           ["ANDAMAN AND NICOBAR ISLANDS"],
    "ARUNACHAL PRADESH":                   ["ARUNACHAL PRADESH"],
    "ASSAM & MEGHALAYA":                   ["ASSAM", "MEGHALAYA"],
    "NAGA MANI MIZO TRIPURA":              ["NAGALAND", "MANIPUR", "MIZORAM", "TRIPURA"],
    "SUB HIMALAYAN WEST BENGAL & SIKKIM":  ["WEST BENGAL", "SIKKIM"],
    "GANGETIC WEST BENGAL":                ["WEST BENGAL"],
    "ORISSA":                              ["ODISHA"],
    "JHARKHAND":                           ["JHARKHAND"],
    "BIHAR":                               ["BIHAR"],
    "EAST UTTAR PRADESH":                  ["UTTAR PRADESH"],
    "WEST UTTAR PRADESH":                  ["UTTAR PRADESH"],
    "UTTARAKHAND":                         ["UTTARAKHAND"],
    "HARYANA DELHI & CHANDIGARH":          ["HARYANA", "CHANDIGARH"],
    "PUNJAB":                              ["PUNJAB"],
    "HIMACHAL PRADESH":                    ["HIMACHAL PRADESH"],
    "JAMMU & KASHMIR":                     ["JAMMU AND KASHMIR"],
    "WEST RAJASTHAN":                      ["RAJASTHAN"],
    "EAST RAJASTHAN":                      ["RAJASTHAN"],
    "WEST MADHYA PRADESH":                 ["MADHYA PRADESH"],
    "EAST MADHYA PRADESH":                 ["MADHYA PRADESH"],
    "GUJARAT REGION":                      ["GUJARAT"],
    "SAURASHTRA & KUTCH":                  ["GUJARAT"],
    "KONKAN & GOA":                        ["GOA", "MAHARASHTRA"],
    "MADHYA MAHARASHTRA":                  ["MAHARASHTRA"],
    "MARATHWADA":                          ["MAHARASHTRA"],
    "VIDARBHA":                            ["MAHARASHTRA"],
    "CHHATTISGARH":                        ["CHHATTISGARH"],
    "COASTAL ANDHRA PRADESH":              ["ANDHRA PRADESH"],
    "RAYALSEEMA":                          ["ANDHRA PRADESH"],
    "TELANGANA":                           ["TELANGANA"],
    "TAMIL NADU":                          ["TAMIL NADU", "PUDUCHERRY"],
    "COASTAL KARNATAKA":                   ["KARNATAKA"],
    "NORTH INTERIOR KARNATAKA":            ["KARNATAKA"],
    "SOUTH INTERIOR KARNATAKA":            ["KARNATAKA"],
    "KERALA":                              ["KERALA"]
}

expanded_rows = []
for _, row in rain_df.iterrows():
    if row["SUBDIVISION"] in subdivision_to_states:
        for state in subdivision_to_states[row["SUBDIVISION"]]:
            expanded_rows.append({
                "State_Name":      state,
                "YEAR":            row["YEAR"],
                "ANNUAL_RAINFALL": row["ANNUAL"]
            })

rain_state_df = pd.DataFrame(expanded_rows)

# -------------------------------
# 5. Merge
# -------------------------------
merged_df = pd.merge(
    crop_grouped,
    rain_state_df,
    left_on=["State_Name", "Crop_Year"],
    right_on=["State_Name", "YEAR"],
    how="inner"
)

# -------------------------------
# 6. Statistical Risk (Z-Score)
# -------------------------------
state_crop_stats = merged_df.groupby(
    ["State_Name", "Crop"]
)["Yield"].agg(["mean", "std"]).reset_index()

state_crop_stats.rename(columns={"mean": "Mean_Yield", "std": "Std_Yield"}, inplace=True)

merged_df = pd.merge(merged_df, state_crop_stats, on=["State_Name", "Crop"], how="left")

merged_df["Std_Yield"].replace(0, 0.0001, inplace=True)

merged_df["Z_Score"] = (
    (merged_df["Yield"] - merged_df["Mean_Yield"]) / merged_df["Std_Yield"]
)

Z_THRESHOLD = -1.0
merged_df["Risk_Label"] = np.where(merged_df["Z_Score"] < Z_THRESHOLD, 1, 0)

print("Risk Distribution:")
print(merged_df["Risk_Label"].value_counts())

# -------------------------------
# 7. Rainfall Bucket
# -------------------------------
merged_df["Rainfall_Bucket"] = pd.cut(
    merged_df["ANNUAL_RAINFALL"],
    bins=[0, 400, 700, 1000, 1300, 9999],
    labels=["<400 mm (Very Dry)", "400–700 mm (Dry)", "700–1000 mm (Moderate)",
            "1000–1300 mm (Good)", ">1300 mm (Heavy)"]
).astype(str)

# -------------------------------
# 8. KMeans Clustering (3 clusters)
# -------------------------------
cluster_features = merged_df[["ANNUAL_RAINFALL", "Yield"]].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged_df.loc[cluster_features.index, "Cluster"] = kmeans.fit_predict(cluster_features)
merged_df["Cluster"] = merged_df["Cluster"].fillna(-1).astype(int)

joblib.dump(kmeans, "models/kmeans_model.pkl")
print("KMeans clustering done.")

# -------------------------------
# 9. Encoding
# -------------------------------
le_state    = LabelEncoder()
le_crop     = LabelEncoder()
le_district = LabelEncoder()

merged_df["State_Encoded"]    = le_state.fit_transform(merged_df["State_Name"])
merged_df["Crop_Encoded"]     = le_crop.fit_transform(merged_df["Crop"])
merged_df["District_Encoded"] = le_district.fit_transform(merged_df["District_Name"])

# -------------------------------
# 10. Model Training
# -------------------------------
X = merged_df[["ANNUAL_RAINFALL", "Area", "State_Encoded", "Crop_Encoded"]]
y = merged_df["Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight={0: 1, 1: 2}
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------
# 11. Save Everything
# -------------------------------
joblib.dump(model,       "models/risk_model.pkl")
joblib.dump(le_state,    "models/state_encoder.pkl")
joblib.dump(le_crop,     "models/crop_encoder.pkl")
joblib.dump(le_district, "models/district_encoder.pkl")

merged_df.to_csv("data/final_merged_dataset.csv", index=False)

print("\nAll models, encoders & dataset saved successfully.")
print(f"Total records: {len(merged_df)}")
print(f"States: {merged_df['State_Name'].nunique()}")
print(f"Crops: {merged_df['Crop'].nunique()}")
print(f"Districts: {merged_df['District_Name'].nunique()}")