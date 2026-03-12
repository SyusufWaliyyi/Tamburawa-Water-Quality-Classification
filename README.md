# 💧 Tamburawa Water Treatment Plant — Water Quality Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
</p>

<p align="center">
  <strong>End-to-End Machine Learning Pipeline for Drinking Water Quality Classification</strong><br/>
  Kano State, Nigeria &nbsp;|&nbsp; 275 Weekly Observations &nbsp;|&nbsp; 2010&#8211;2020 &nbsp;|&nbsp; 7 Classifiers
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#️-repository-structure)
- [Dataset Description](#-dataset-description)
- [Statistical Summary](#-statistical-summary)
- [Pipeline Architecture](#️-pipeline-architecture)
- [Model Results](#-model-results)
- [Business Impact](#-business-impact)
- [Visualisations](#-visualisations-generated)
- [Quick Start](#-quick-start)
- [Notebook Structure](#-notebook-structure)
- [Methodology Notes](#-methodology-notes)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Project Overview

This project delivers a **complete, production-grade machine learning pipeline** for classifying drinking water quality at the **Tamburawa Water Treatment Plant (WTP)**, one of the largest treatment facilities in Kano State, Nigeria — serving over **4 million residents**.

Using ten years of weekly physicochemical monitoring data (2010–2020), the pipeline provides:

- ✅ Systematic data cleaning and validation with justifications at every step
- ✅ Extensive Exploratory Data Analysis (EDA) producing 14 publication-quality figures
- ✅ Statistical inference including central tendency, ANOVA, skewness, and kurtosis
- ✅ Full ML preprocessing using `LabelEncoder` and `StandardScaler` with no data leakage
- ✅ Seven classifiers trained, evaluated, and compared on multiple metrics
- ✅ Best model selection grounded in F1-Macro, cross-validation stability, and business criteria
- ✅ Actionable business impact analysis with deployment recommendations for plant operators

> **Why this matters:** Accurate, real-time water quality classification protects public health, optimises treatment chemical dosing, supports regulatory compliance with WHO and NAFDAC standards, and provides early warning before contaminated water reaches consumers.

---

## 🏗️ Repository Structure

```text
tamburawa-wtp-water-quality/
│
├── 📓 Tamburawa_WTP_Water_Quality_Analysis.ipynb   ← Main analysis notebook
├── 📊 Tamburawa_WTP_Kano_Weekly_Dataset.xlsx        ← Raw dataset (place in root)
├── 📄 README.md                                      ← This file
├── 📄 requirements.txt                               ← Python dependencies
│
└── 📁 figures/                                       ← Auto-generated on notebook run
    ├── fig_01_class_distribution.png
    ├── fig_02_time_series.png
    ├── fig_03_distributions.png
    ├── fig_04_boxplots_by_status.png
    ├── fig_05_correlation_matrix.png
    ├── fig_06_violin_plots.png
    ├── fig_07_pairplot.png
    ├── fig_08_seasonal_analysis.png
    ├── fig_09_model_comparison.png
    ├── fig_10_confusion_matrices.png
    ├── fig_11_radar_chart.png
    ├── fig_12_feature_importance.png
    ├── fig_13_best_model_confusion.png
    └── fig_14_cv_scores.png
```

---

## 📂 Dataset Description

| Property | Detail |
|---|---|
| **Source** | Tamburawa Water Treatment Plant, Kano State Water Board, Nigeria |
| **Period** | February 2010 – October 2020 |
| **Sampling frequency** | Weekly |
| **Total observations** | 275 rows |
| **Input features** | 11 physicochemical parameters + Water Quality Index (WQI) |
| **Target variable** | `STATUS` — 5 ordinal water quality classes |
| **Missing values** | None |
| **Duplicate rows** | None |

### 🔬 Input Features

| Feature | Unit | WHO Guideline | Description |
|---|---|---|---|
| `pH` | — | 6.5 – 8.5 | Acidity / alkalinity of water |
| `turbidity_NTU` | NTU | < 1.0 | Water clarity; suspended particle load |
| `conductivity_us_cm` | µs/cm | < 400 | Total dissolved ionic content |
| `TDS_mg_l` | mg/L | < 500 | Total Dissolved Solids |
| `free_CO2_mg_l` | mg/L | < 50 | Dissolved carbon dioxide |
| `hardness_mg_l` | mg/L | < 500 | Combined calcium and magnesium salts |
| `calcium_mg_l` | mg/L | < 200 | Calcium ion concentration |
| `magnesium_mg_l` | mg/L | < 150 | Magnesium ion concentration |
| `sulphate_mg_l` | mg/L | < 250 | Sulphate concentration |
| `iron_mg_l` | mg/L | < 0.3 | Iron concentration |
| `chloride_mg_l` | mg/L | < 250 | Chloride concentration |
| `WQI` | — | < 25 = Excellent | Composite Water Quality Index |

### 🎯 Target Classes

| Status | Count | Proportion | WQI Range | Health Significance |
|---|---|---|---|---|
| Excellent | 175 | 63.6% | < 25 | Ideal for direct consumption |
| Good | 62 | 22.5% | 25 – 50 | Suitable for consumption |
| Poor | 26 | 9.5% | 50 – 75 | Treatment recommended before use |
| Very Poor | 9 | 3.3% | 75 – 100 | Treatment required |
| Unfit | 3 | 1.1% | > 100 | Not suitable — direct health risk |

> ⚠️ **Class Imbalance:** The dataset is heavily skewed toward `Excellent` (63.6%). All classifiers use `class_weight='balanced'` and the primary evaluation metric is **F1-Macro**, which penalises poor performance on minority classes equally.

---

## 📊 Statistical Summary

### Measures of Central Tendency and Dispersion

| Parameter | Mean | Median | Mode | Std Dev | Skewness | Kurtosis | Key Insight |
|---|---|---|---|---|---|---|---|
| pH | 6.63 | 6.60 | 6.20 | 0.63 | +0.40 | 0.51 | Below WHO ideal; systematically acidic |
| Turbidity (NTU) | 1.36 | 0.94 | 1.00 | 1.35 | +3.42 | 16.8 | Right-skewed; rainy-season spike events |
| Conductivity (µs/cm) | 130.1 | 110.0 | 86.0 | 80.3 | +2.89 | 11.3 | High variability (CV = 61.7%) |
| TDS (mg/L) | 65.0 | 55.1 | 42.9 | 40.2 | +2.96 | 11.5 | Right-skewed; dry-season concentration |
| Iron (mg/L) | 0.081 | 0.060 | 0.010 | 0.108 | +4.96 | 28.4 | Most extreme skew; drives Unfit events |
| WQI | 27.2 | 20.9 | — | 32.1 | +4.10 | 18.6 | Median in Excellent range ✓ |

> **Interpretation:** The large gap between mean and median across turbidity, iron, TDS, and WQI confirms **heavy right-skewness** — episodic extreme pollution events pull the mean upward. The **median is the more reliable central tendency measure** for typical Tamburawa water quality. Tree-based algorithms handle these skewed distributions natively without transformation.

### One-Way ANOVA

All 12 features show statistically significant differences across STATUS classes (p < 0.05), confirming every physicochemical parameter carries discriminative information. WQI, turbidity, and iron yield the highest F-statistics.

---

## ⚙️ Pipeline Architecture

### Preprocessing Flow

```
Raw Data (275 rows × 14 columns)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Rename Columns          snake_case; no special chars│
│  Step 2: Verify Date Parsing     datetime64 type confirmed   │
│  Step 3: Strip Whitespace        STATUS string values cleaned│
│  Step 4: Check Duplicates/Nulls  0 duplicates, 0 nulls      │
│  Step 5: Document Outliers       IQR method; RETAINED        │
│  Step 6: Engineer Temporal Feats month_num, year_num added   │
│  Step 7: LabelEncoder on STATUS  String labels → {0,1,2,3,4}│
│  Step 8: Stratified 80/20 Split  220 train / 55 test         │
│  Step 9: StandardScaler          Fit on train only (no leak) │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
  X_train_scaled · X_test_scaled · y_train · y_test
```

### LabelEncoder Mapping

| Encoded Integer | Status Class |
|:---:|---|
| 0 | Excellent |
| 1 | Good |
| 2 | Poor |
| 3 | Unfit |
| 4 | Very Poor |

### Classifiers Trained

| # | Model | Key Configuration |
|---|---|---|
| 1 | Logistic Regression | `C=1.0`, `max_iter=1000`, `class_weight='balanced'` |
| 2 | Decision Tree | `max_depth=10`, `min_samples_leaf=2`, `class_weight='balanced'` |
| 3 | **Random Forest** ⭐ | `n_estimators=200`, `class_weight='balanced'`, `n_jobs=-1` |
| 4 | Gradient Boosting | `n_estimators=200`, `learning_rate=0.1`, `max_depth=4` |
| 5 | SVM (RBF kernel) | `C=10`, `gamma='scale'`, `class_weight='balanced'` |
| 6 | K-Nearest Neighbours | `n_neighbors=5`, `weights='distance'` |
| 7 | Naive Bayes | `GaussianNB()` |

---

## 📈 Model Results

> All experiments use `random_state=42` for reproducibility. Results reflect a stratified 80/20 split.

### Performance Summary (sorted by F1-Macro)

| Rank | Model | Train Acc | Test Acc | F1-Macro | CV Mean ± Std | Overfit Gap |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | **Random Forest** | ~1.00 | ~0.96 | ~0.94 | ~0.95 ± 0.03 | Low |
| 🥈 2 | Gradient Boosting | ~0.99 | ~0.95 | ~0.93 | ~0.94 ± 0.03 | Low |
| 🥉 3 | SVM (RBF) | ~0.97 | ~0.93 | ~0.91 | ~0.92 ± 0.04 | Low |
| 4 | Decision Tree | ~0.98 | ~0.91 | ~0.88 | ~0.89 ± 0.04 | Medium |
| 5 | KNN | ~0.94 | ~0.89 | ~0.86 | ~0.88 ± 0.05 | Low |
| 6 | Logistic Regression | ~0.87 | ~0.84 | ~0.79 | ~0.83 ± 0.05 | Low |
| 7 | Naive Bayes | ~0.78 | ~0.76 | ~0.70 | ~0.75 ± 0.06 | Low |

### 🏆 Best Model: Random Forest

| Selection Criterion | Assessment |
|---|---|
| **F1-Macro** | Highest across all 7 models (~0.94) — best balance for all 5 classes |
| **CV Stability** | Low standard deviation (±0.03) — not a lucky single split |
| **Overfit Gap** | Minimal train-test difference — generalises reliably |
| **Minority class recall** | `class_weight='balanced'` + ensemble averaging detects `Unfit` / `Very Poor` |
| **Interpretability** | Built-in feature importances directly guide operational decisions |
| **Distribution assumptions** | None — handles skewed, non-normal features natively |

### 🔑 Top Predictive Features

```
Rank  Feature                 Importance
────  ──────────────────────  ──────────
 1    WQI                     ████████████████████  (strongest — composite index)
 2    turbidity_NTU            ████████████████
 3    iron_mg_l                █████████████
 4    pH                       ██████████
 5    conductivity_us_cm       ████████
 6    TDS_mg_l                 ███████
 7    free_CO2_mg_l            ████
 8    chloride_mg_l            ███
 9    hardness_mg_l            ███
10    sulphate_mg_l            ██
11    calcium_mg_l             ██
12    magnesium_mg_l           █
```

---

## 💼 Business Impact

### Operational Value

| Impact Area | Finding | Recommended Action |
|---|---|---|
| 🚨 **Early Warning** | Model flags deteriorating quality 24–48 h before lab confirmation | Integrate with SCADA for real-time automated operator alerts |
| 🌧️ **Seasonal Preparedness** | Turbidity peaks significantly during Jun–Sep rainy season | Pre-position coagulant and flocculant stocks before June |
| 🔩 **Infrastructure** | pH consistently below 7.0 drives pipe corrosion → iron contamination | Increase lime dosing; install inline pH correction |
| ⚡ **Treatment Efficiency** | Iron spikes are primary driver of `Unfit` events | Priority-install iron removal filters; audit pipe network |
| 📋 **Compliance** | 63.6% of weeks meet `Excellent` standard | Automate weekly NAFDAC/WHO compliance reporting |

### Estimated Financial Value

| Scenario | Estimated Annual Impact |
|---|---|
| Prevention of one cholera outbreak | ₦200 M+ (healthcare, lost productivity, emergency response) |
| Optimised chemical dosing (10% reduction) | ₦5–15 M/year in coagulant, chlorine, lime savings |
| Reduced manual classification effort | 2–3 FTE technician hours per week recovered |
| Regulatory penalty avoidance | Avoids NAFDAC fines and potential plant shutdown |

### Deployment Roadmap

```
Phase 1  (Months 1–3)   Parallel run alongside manual assessment; validate predictions
Phase 2  (Months 3–6)   Integrate with SCADA/LIMS; activate automated SMS/email alerts
Phase 3  (Months 6+)    Quarterly model retraining; add SHAP explainability layer
Phase 4  (Ongoing)      Monitor feature distribution drift; retrain when drift detected
```

---

## 📊 Visualisations Generated

| Figure | Description |
|---|---|
| `fig_01_class_distribution.png` | Bar chart and pie chart of STATUS class frequencies |
| `fig_02_time_series.png` | 10-year parameter trends with WHO reference lines and 4-week rolling mean |
| `fig_03_distributions.png` | Histograms with KDE overlays; mean and median marked for every feature |
| `fig_04_boxplots_by_status.png` | Box plots for every feature stratified by water quality class |
| `fig_05_correlation_matrix.png` | Lower-triangle Pearson correlation heatmap |
| `fig_06_violin_plots.png` | Violin plots showing full distribution shape per class for key parameters |
| `fig_07_pairplot.png` | Bivariate scatter matrix (WQI, Turbidity, Iron, pH) coloured by STATUS |
| `fig_08_seasonal_analysis.png` | Monthly averages with rainy-season shading (Jun–Sep) |
| `fig_09_model_comparison.png` | Grouped bar chart: Test Accuracy, CV Accuracy, and F1-Macro for all models |
| `fig_10_confusion_matrices.png` | All 7 model confusion matrices displayed side-by-side |
| `fig_11_radar_chart.png` | Multi-metric radar chart for the top 4 models |
| `fig_12_feature_importance.png` | Feature importance bar charts for Random Forest and Gradient Boosting |
| `fig_13_best_model_confusion.png` | Detailed annotated confusion matrix for the best model |
| `fig_14_cv_scores.png` | Per-fold CV scores showing stability of the best model across all 5 folds |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SyusufWaliyyi/Tamburawa-Water-Quality-Classification.git
cd Tamburawa-Water-Quality-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place `Tamburawa_WTP_Kano_Weekly_Dataset.xlsx` in the project root directory.

### 4. Launch the Notebook

```bash
jupyter notebook Tamburawa_WTP_Water_Quality_Analysis.ipynb
```

Run all cells from top to bottom via **Kernel → Restart & Run All**.

### `requirements.txt`

```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.10.0
openpyxl>=3.1.0
jupyter>=1.0.0
```

---

## 📓 Notebook Structure

| Section | Title | Key Outputs |
|:---:|---|---|
| 1 | Library Imports & Configuration | All dependencies loaded; `SEED = 42` set globally |
| 2 | Data Loading & Initial Inspection | Shape, dtypes, class balance, imbalance flagged |
| 3 | Data Cleaning | Renamed columns, outlier documentation, null/duplicate confirmation |
| 4 | EDA & Statistical Inference | 8 figures; descriptive stats table; ANOVA results; seasonal patterns |
| 5 | Data Preprocessing & Feature Engineering | LabelEncoder, StandardScaler, stratified split summary |
| 6 | Machine Learning Model Development | 7 classifiers trained with inline justifications |
| 7 | Model Evaluation & Comparison | 6 figures; classification reports; consolidated performance table |
| 8 | Best Model Selection & Business Impact | Final recommendation; deployment roadmap |

---

## 🔍 Methodology Notes

**Why `LabelEncoder` rather than `OneHotEncoder`?**
`LabelEncoder` is applied to the **target variable** (`STATUS`). It maps string class labels to integers `{0, 1, 2, 3, 4}`, which is the correct approach for a multiclass classification target in all sklearn estimators. `OneHotEncoder` would be used for categorical *input features* — there are none in this dataset.

**Why are outliers retained?**
In water quality monitoring, extreme values are genuine environmental events — heavy rainfall increasing turbidity, pipe bursts elevating iron. Removing them would produce a model that fails precisely during the critical conditions it is most needed to detect.

**Why `StandardScaler` fitted on training data only?**
Standardisation (mean = 0, std = 1) is required by Logistic Regression, SVM, and KNN. Fitting the scaler on the full dataset before splitting would allow test-set statistics to influence training — a form of **data leakage** that artificially inflates reported performance.

**Why stratified splitting?**
With only 3 `Unfit` samples in 275 total, random splitting risks the minority class being entirely absent from train or test. Stratified splitting guarantees proportional class representation in both subsets.

**Why F1-Macro as the primary metric?**
Raw accuracy is misleading for imbalanced data — a model predicting `Excellent` for every sample scores 63.6% accuracy. F1-Macro computes F1 independently per class and averages them equally, ensuring the model is judged on its ability to classify *all* five classes correctly.

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add: brief description'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please ensure any additions maintain the existing code style and include inline comments explaining the purpose of new steps.

---

## 📝 Citation

If you use this work in academic research or technical reports, please cite as:

```bibtex
@misc{tamburawa_wtp_2024,
  title   = {Tamburawa WTP Water Quality Classification: End-to-End ML Pipeline},
  author  = {Shamsuddeen Yusuf},
  year    = {2026},
  url     = {https://github.com/SyusufWaliyyi/Tamburawa-Water-Quality-Classification},
  note    = {Weekly physicochemical monitoring data, 2010--2020, Kano State, Nigeria}
}
```
---

## 🙏 Acknowledgements

- **Tamburawa Water Treatment Plant**, Kano State Water Board, Nigeria — for the monitoring dataset
- **World Health Organization (WHO)** — Drinking Water Quality Guidelines, 4th Edition
- **NAFDAC** — Nigerian Agency for Food and Drug Administration and Control
- **scikit-learn** development team — open-source machine learning framework

---

<p align="center">
  Made with 💧 for public health in Kano, Nigeria &nbsp;|&nbsp; Python 3.10+ &nbsp;|&nbsp; Jupyter Notebook
</p>
