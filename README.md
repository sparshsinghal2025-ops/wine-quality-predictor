# 🍷 Wine Quality Predictor

An end-to-end machine learning application that predicts whether a wine sample is of **Premium Quality** or **Standard Quality** based on its laboratory chemical composition. 

🔗 **Live Web Application:** [Predict Wine Quality Live](https://streamlit.app)

---

## 📌 Project Overview
This project uses chemical profiles to automate quality control. It processes raw physicochemical features, trains multiple optimized classifiers, and serves predictions via an interactive web interface. 

### 🚀 Key Features
* **Production Pipeline**: Prevents data leakage using Scikit-Learn transformers.
* **Hyperparameter Tuning**: Optimizes algorithms using `RandomizedSearchCV`.
* **Robust Frontend**: Employs an ultra-stable, cross-browser numeric input dashboard.
* **Automated Serialization**: Packages pipelines into secure `.pkl` artifacts.

---

## 📊 Dataset Profile
The project utilizes the **WineQT Dataset**, consisting of 1,143 wine composition records. 

### Predictive Features (X):
1. **Fixed Acidity** (g/dm³) - Primary structural wine acids.
2. **Volatile Acidity** (g/dm³) - Acetic acid levels (vinegar taste).
3. **Citric Acid** (g/dm³) - Freshness and crisp flavor notes.
4. **Residual Sugar** (g/dm³) - Post-fermentation natural sugars.
5. **Chlorides** (g/dm³) - Dissolved salt levels.
6. **Free Sulfur Dioxide** (mg/dm³) - Gas preventing microbial spoilage.
7. **Total Sulfur Dioxide** (mg/dm³) - Cumulative preservative concentration.
8. **Density** (g/cm³) - Fluid thickness matrix score.
9. **pH Level** - Measures global alkalinity vs acidity.
10. **Sulphates** (g/dm³) - Antimicrobial preservation additives.
11. **Alcohol** (% by volume) - Percentage of alcohol content.

### Target Classification (y):
* **`1` (Premium Quality)**: Quality score \(\geq\) 6
* **`0` (Standard Quality)**: Quality score \(<\) 6

---

## ⚡ Machine Learning Benchmarks
Three classifier models were trained and tuned using 3-fold cross-validation. **XGBoost** emerged as the winning model architecture.

| Model Algorithm | Evaluation Accuracy | F1-Score Baseline | Status |
| :--- | :---: | :---: | :---: |
| **XGBoost Classifier** | **81.22%** | **0.8352** | 🏆 **Winner** |
| Random Forest Classifier | 80.35% | 0.8249 | Candidate |
| Logistic Regression | 77.73% | 0.7918 | Baseline |

*Key Insight: Feature evaluation reveals that **Alcohol Content** and **Sulphate Concentration** provide the strongest statistical signals for quality classification.*

---

## 📂 Repository File Structure
```text
├── app.py                      # Main Streamlit web app dashboard code
├── WineQualityPredictor.py     # Data cleaning, model tuning, and training pipeline
├── plot_importance.py          # Feature extraction and matplotlib chart generator
├── WineQualityPrediction.pkl   # Serialized pipeline (model, scaler, preprocessor)
├── wine_feature_importance.png # Saved horizontal bar chart asset
├── requirements.txt            # Cloud engine dependency version locks
└── README.md                   # Production project documentation
```

---

## 🛠️ Local Installation & Setup

1. **Clone the project repository:**
   ```bash
   git clone https://github.com
   cd wine-quality-predictor
   ```

2. **Install all required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models and generate pipeline artifacts:**
   ```bash
   python WineQualityPredictor.py
   ```

4. **Launch the local interactive Streamlit server:**
   ```bash
   streamlit run app.py
   ```

---

## 🎛️ Technologies Used
* **Languages**: Python
* **Data Processing**: Pandas, NumPy
* **Modeling Engine**: Scikit-Learn, XGBoost
* **Data Visualizations**: Matplotlib, Seaborn
* **Deployment Interface**: Streamlit Cloud
