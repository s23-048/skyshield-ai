# 🛸 SkyShield AI: Unified Aviation Safety Platform

SkyShield AI is a comprehensive, intelligent aviation safety platform that merges two key analytical capabilities into a single dashboard and CLI interface:

1. **Bird Strike Damage Predictor (ML)**: A predictive model using a Random Forest Classifier that estimates the probability and severity of aircraft damage resulting from wildlife strikes.
2. **Aviation Incident Search Engine (IR)**: An Information Retrieval search engine built using TF-IDF weighting and the Vector Space Model to retrieve and rank historical NTSB aviation accident summaries.

Developed by **Sharanabasava S** (USN: 1SI23CI048), B.Tech CSE (AI & ML) at Siddaganga Institute of Technology, Tumakuru.

---

## 🏗️ Project Architecture & Structure

```
SkyShield AI/
│
├── app.py                    # Unified Streamlit Web Dashboard
├── airsafe_search.py         # IR Search Engine Core (CLI & API module)
├── train.py                  # ML Model Training Pipeline
├── preprocess.py             # Data loading and preprocessing library
├── get_dataset.py            # Automated FAA-style dataset mirror downloader
├── requirements.txt          # Shared python dependencies
│
├── data/
│   └── faa_bird_strike_sample_10k.csv  # 10k sample bird strike dataset
│
├── model/
│   └── bird_strike_pipeline.pkl        # Saved Random Forest Pipeline model
│
├── docs/
│   └── report_1.txt ... report_20.txt  # Document corpus (NTSB Accident Reports)
│
└── venv/                     # Python virtual environment
```

---

## ⚡ Setup & Installation

### 1. Initialize Virtual Environment
Create and activate a Python virtual environment to manage dependencies locally:
```bash
# Create environment
python3 -m venv venv

# Activate on macOS / Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies
Install all required libraries, including Streamlit, scikit-learn, pandas, numpy, joblib, and matplotlib:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the Applications

### 🖥️ Launch the Unified Web Dashboard (Streamlit)
Start the high-end Streamlit web dashboard to access both modules interactively:
```bash
streamlit run app.py
```
*The web page will automatically open at `http://localhost:8501`.*

### 🐚 Launch the Accident Search CLI
If you prefer a lightweight terminal experience for search, you can run the CLI engine directly:
```bash
python3 airsafe_search.py
```
*Type in queries like `engine failure` or `bird strike` to see ranked reports and scores.*

### 🤖 Retrain the ML Model
To refresh the dataset and retrain the Random Forest model:
```bash
# Optional: Download/clean fresh FAA wildlife strike data
python3 get_dataset.py

# Retrain model and evaluate performance
python3 train.py
```

---

## 🧠 Technical Details

### 1. Bird Strike Predictor (Machine Learning)
* **Model**: `Random Forest Classifier` (150 estimators).
* **Pipeline**: Integrates automated `OneHotEncoder` for categorical categories and passes through numerical features.
* **Accuracy**: ~93.3% accuracy on a stratified test split.
* **Input Features**:
  * `AIRCRAFT`: Aircraft class (e.g., Narrowbody, Widebody, Business Jet)
  * `PHASE_OF_FLIGHT`: Takeoff, Landing, Climb, Cruise, etc.
  * `SPECIES`: Bird species (e.g., Gull, Waterfowl, Raptor)
  * `TIME_OF_DAY`: Day, Night, Dawn, Dusk
  * `HEIGHT`: Altitude in feet above ground level

### 2. Accident Search Engine (Information Retrieval)
* **Pre-processing**: Lowercasing, Alphanumeric tokenization, and customizable Stopword removal.
* **Model**: TF-IDF Term Frequency-Inverse Document Frequency weight representation.
* **Similarity Metric**: Vector Space Cosine Similarity normalization.
* **Visual Highlights**: Search results displayed on the Streamlit dashboard highlight matching terms in real-time.
