# 🌾 Farmer Mistake Detector

> AI-powered agricultural risk analysis system built on 100+ years of Indian farming data (1901–2015)

## 📌 What is this?

Farmers in India often make crop decisions based on intuition or tradition — without knowing the historical risk associated with growing a specific crop in their region under given rainfall conditions.

**Farmer Mistake Detector** solves this by analyzing 100+ years of agricultural data and predicting whether a farming decision (crop + state + rainfall + area) is **risky or safe** — and explaining *why*.

---

## 🚀 Live Demo

👉 **[Open the App](https://farmer-mistake-detector-2w3u37aplwf2bke9cjfu4m.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **AI Risk Prediction** | Random Forest model predicts crop failure risk (0–100%) |
| 🔬 **SHAP Explainability** | Shows which factors (rainfall, crop, state, area) drove the prediction |
| 🌱 **Alternative Crop Suggestions** | Recommends safer, higher-yield crops for your region |
| 📋 **Crop Comparison Table** | Compare all crops in your state by yield, max yield, and risk % |
| 📈 **Historical Yield Trend** | Visualize yield history with risk years highlighted |
| 🌧️ **Rainfall Trend Chart** | See how historical rainfall compares to your input |
| 📊 **EDA Dashboard** | Distribution plots, correlation heatmap, boxplot, scatter analysis |
| 🧠 **AI Insights** | Rule-based automated analysis with actionable recommendations |
| 🔵 **KMeans Clustering** | Farms grouped into 3 clusters by rainfall + yield pattern |

---

## 🖼️ App Preview

### AI Prediction Tab
- Enter State, Crop, Rainfall, Area → Get instant risk verdict
- SHAP bar chart explains each feature's contribution
- Risk probability shown with color-coded metric cards

### EDA Tab
- Yield & Rainfall distribution histograms
- Safe vs At-Risk farm comparison
- Correlation heatmap between Yield, Rainfall, Area
- Outlier detection via boxplot

### Trends Tab
- Year-wise yield trend with slope analysis
- National crop risk % over time
- Top 10 crops by average yield
- State-wise productivity comparison
- Optimal rainfall range analysis

### AI Insights Tab
- Automated district-level analysis
- 8+ rule-based insights generated per region
- Best crop recommendation with risk + yield score

---

## 🛠️ Tech Stack

```
Language     : Python 3.11
ML Model     : Random Forest Classifier (scikit-learn)
Explainability: SHAP (TreeExplainer)
Clustering   : KMeans (scikit-learn)
Frontend     : Streamlit
Data         : Pandas, NumPy
Visualization: Matplotlib, Seaborn
Deployment   : Streamlit Cloud
```

---

## 📂 Project Structure

```
Farmer_Mistake_Detector/
│
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version (3.11)
│
├── models/
│   ├── risk_model.pkl          # Trained Random Forest model
│   ├── state_encoder.pkl       # LabelEncoder for states
│   ├── crop_encoder.pkl        # LabelEncoder for crops
│   ├── district_encoder.pkl    # LabelEncoder for districts
│   └── kmeans_model.pkl        # KMeans clustering model
│
└── data/
    └── final_merged_dataset.csv  # Merged crop + rainfall dataset
```

---

## ⚙️ How It Works

### 1. Data Sources
- **crop_production.csv** — State, district, crop, year, area, production data
- **rainfall in india 1901-2015.csv** — Annual rainfall by subdivision

### 2. Feature Engineering
```
Yield = Production / Area
```
Subdivisions are mapped to states manually (e.g. "ASSAM & MEGHALAYA" → ["ASSAM", "MEGHALAYA"])

### 3. Risk Labeling (Z-Score Method)
```
Z = (Yield - Mean_Yield_for_StateCrop) / Std_Yield_for_StateCrop

Z < -1.0  →  Risk_Label = 1  (risky year)
Z ≥ -1.0  →  Risk_Label = 0  (safe year)
```
Z-score is calculated *within* each State+Crop combination — so Rice in Kerala is compared to Rice in Kerala's history, not to Wheat in Rajasthan.

### 4. Model Training
- **Algorithm:** Random Forest Classifier (200 trees, max_depth=12)
- **Features:** Annual Rainfall, Area, State (encoded), Crop (encoded)
- **Class Weight:** `{0:1, 1:2}` — Risk cases get 2x importance
- **Split:** 80% train / 20% test

### 5. Prediction
```
User Input → LabelEncode → predict_proba() → Risk %
If probability > 0.35 → HIGH RISK
```

### 6. SHAP Explainability
TreeExplainer computes feature-level SHAP values — showing exactly how much each input pushed the prediction toward risk or safety.

---

## 🔧 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/chiragkapoor1711/Farmer-Mistake-Detector.git
cd Farmer-Mistake-Detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(only needed if models/ folder is missing)*
```bash
python train_model.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📊 Dataset Info

| Property | Details |
|---|---|
| Source | Government of India agricultural records |
| Time Period | 1901 – 2015 |
| Coverage | 29 States, 50+ Crops |
| Records | ~90,000+ rows after merging |
| Features | State, District, Crop, Year, Area, Production, Rainfall |

---

## 🎯 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest (200 trees) |
| Risk Threshold | 0.35 probability |
| Class Weighting | Risk class 2x boosted |
| Labeling Method | Z-score (threshold = -1.0) |

---

## 👨‍💻 Author

**Chirag Kapoor**  
Final Year Project — Data Science & AI  

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  <sub>Built with ❤️ using Python · Streamlit · Scikit-learn · SHAP</sub>
</div>
