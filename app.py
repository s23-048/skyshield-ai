# app.py
import os
import re
import math
import joblib
import pandas as pd
import streamlit as st

# Import information retrieval modules from airsafe_search.py
from airsafe_search import (
    load_documents,
    build_index,
    search,
    tokenize,
    make_snippet
)

# Configuration
MODEL_PATH = "model/bird_strike_pipeline.pkl"
DATA_PATH = "data/faa_bird_strike_sample_10k.csv"

SEVERITY_LABELS = {
    0: "No Damage",
    1: "Minor Damage",
    2: "Severe Damage",
}

# --- CSS Custom Styling for a Premium Dark Look ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom Cards */
.skyshield-hero {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    color: #ffffff;
}

.skyshield-card {
    background: rgba(30, 34, 42, 0.65);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease-in-out;
}

.skyshield-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.1);
}

.skyshield-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #4facfe;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Result cards */
.result-card-none {
    background: rgba(46, 117, 89, 0.2);
    border-left: 6px solid #2e7559;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
}

.result-card-minor {
    background: rgba(223, 163, 62, 0.2);
    border-left: 6px solid #dfa33e;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
}

.result-card-severe {
    background: rgba(219, 68, 85, 0.2);
    border-left: 6px solid #db4455;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
}

/* Badges */
.skyshield-badge {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    color: #0d0f12;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    display: inline-block;
}

.highlight {
    background-color: rgba(0, 242, 254, 0.25);
    color: #00f2fe;
    font-weight: 600;
    padding: 0 4px;
    border-radius: 4px;
    border-bottom: 1px solid rgba(0, 242, 254, 0.4);
}

/* Metrics and layout */
.stat-box {
    text-align: center;
    padding: 15px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #00f2fe;
}

.stat-label {
    font-size: 0.9rem;
    color: #8a99ad;
}

/* Progress bars custom styling */
.prog-container {
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
    width: 100%;
    margin-bottom: 10px;
    height: 10px;
    overflow: hidden;
}

.prog-bar {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease-in-out;
}
</style>
"""

# Load dataset with caching
@st.cache_data
def get_cached_dataset(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        return df
    return None

# Load model with caching
@st.cache_resource
def get_cached_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# Load search documents with caching
@st.cache_resource
def get_cached_search_index():
    docs = load_documents()
    if not docs:
        return {}, {}, {}, {}
    inverted_index, idf, doc_lengths = build_index(docs)
    return docs, inverted_index, idf, doc_lengths

# Highlight search queries in text
def highlight_keywords(text: str, query_tokens: list) -> str:
    highlighted = text
    # Sort tokens by length in descending order to avoid nested replacements
    unique_tokens = sorted(list(set(query_tokens)), key=len, reverse=True)
    for token in unique_tokens:
        if len(token) < 2:  # Skip single characters/punctuation
            continue
        pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"<span class='highlight'>{m.group(0)}</span>", highlighted)
    return highlighted

def main():
    st.set_page_config(
        page_title="SkyShield AI",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("<h2 style='text-align: center;'>🛸 SkyShield AI</h2>", unsafe_allow_html=True)
    st.sidebar.write("Unified Intelligent Aviation Safety Suite")
    st.sidebar.markdown("---")

    # Load resources
    pipeline = get_cached_model(MODEL_PATH)
    df = get_cached_dataset(DATA_PATH)
    docs, inverted_index, idf, doc_lengths = get_cached_search_index()

    # Tabs
    tab_overview, tab_prediction, tab_search = st.tabs([
        "📊 System Hub",
        "🔮 Damage Predictor",
        "🔍 Accident Search Engine"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab_overview:
        st.markdown(
            """
            <div class="skyshield-hero">
                <h1 style='margin:0; font-size: 2.5rem; font-weight: 700;'>SkyShield AI</h1>
                <p style='font-size: 1.2rem; margin-top: 10px; opacity: 0.9;'>
                    Combining Predictive Machine Learning and Vector-Based Information Retrieval to Enhance Aviation Incident Assessment.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### ✈️ Platform Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                '<div class="stat-box"><div class="stat-number">10k</div><div class="stat-label">FAA Incidents Logged</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                '<div class="stat-box"><div class="stat-number">~86%</div><div class="stat-label">RF Predictor Accuracy</div></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                '<div class="stat-box"><div class="stat-number">20</div><div class="stat-label">NTSB Reports Indexed</div></div>',
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                '<div class="stat-box"><div class="stat-number">TF-IDF</div><div class="stat-label">Vector Space Search</div></div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("---")

        st.markdown("### 🛠️ Core Capabilities")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                """
                <div class="skyshield-card">
                    <div class="skyshield-title">🧠 Damage Predictor (ML Module)</div>
                    <p>Uses a <strong>Random Forest Classifier</strong> pipeline with automated One-Hot Encoding to estimate the risk and severity of damage to aircraft resulting from wildlife strikes.</p>
                    <ul>
                        <li><strong>Features:</strong> Aircraft class, flight phase, bird species, time of day, altitude.</li>
                        <li><strong>Output:</strong> Multi-class probability distributions (No Damage, Minor, Severe).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_right:
            st.markdown(
                """
                <div class="skyshield-card">
                    <div class="skyshield-title">🔍 Incident Search (IR Module)</div>
                    <p>Implements a <strong>Vector Space Retrieval Model</strong> using <strong>TF-IDF weighting</strong> and cosine similarity to quickly extract historical NTSB aviation accident summaries.</p>
                    <ul>
                        <li><strong>Features:</strong> Case-insensitive natural language querying, text snippets generation.</li>
                        <li><strong>Output:</strong> Ranked documents matching keywords with similarity score overlays.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #8a99ad; padding: 15px;'>
                <strong>SkyShield AI Platform</strong> • Built by Sharanabasava S (USN: 1SI23CI048)<br>
                B.Tech CSE (AI & ML) • Siddaganga Institute of Technology, Tumakuru
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- TAB 2: DAMAGE PREDICTOR ---
    with tab_prediction:
        st.markdown("### 🔮 Predict Bird Strike Damage Severity")
        st.write("Provide the flight and wildlife conditions to estimate the potential damage severity class.")

        if pipeline is None:
            st.error("Model file not found. Please verify `model/bird_strike_pipeline.pkl` or run `python train.py` first.")
        else:
            # Dropdowns populated from dataset
            if df is not None:
                aircraft_options = sorted(df["AIRCRAFT"].astype(str).unique().tolist())
                phase_options = sorted(df["PHASE_OF_FLIGHT"].astype(str).unique().tolist())
                species_options = sorted(df["SPECIES"].astype(str).unique().tolist())
                time_options = sorted(df["TIME_OF_DAY"].astype(str).unique().tolist())
            else:
                aircraft_options = ["NARROWBODY", "WIDEBODY", "BUSINESS JET", "GA SINGLE", "UNKNOWN"]
                phase_options = ["TAKEOFF", "LANDING", "CLIMB", "APPROACH", "CRUISE", "UNKNOWN"]
                species_options = ["GULL", "WATERFOWL", "RAPTOR", "SONGBIRD", "UNKNOWN"]
                time_options = ["DAY", "NIGHT", "DAWN", "DUSK", "UNKNOWN"]

            # Input form layout
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                aircraft = st.selectbox("Aircraft Type", aircraft_options, index=0)
                phase = st.selectbox("Phase of Flight", phase_options, index=0)
                species = st.selectbox("Bird Species", species_options, index=0)
            with col_in2:
                time_of_day = st.selectbox("Time of Day", time_options, index=0)
                altitude = st.number_input(
                    "Altitude / Height (feet AGL)",
                    min_value=-1,
                    max_value=50000,
                    value=1000,
                    step=100,
                    help="Height above ground level when strike occurred."
                )

            st.write("")
            if st.button("🔮 Forecast Strike Impact", type="primary"):
                input_df = pd.DataFrame([
                    {
                        "AIRCRAFT": aircraft,
                        "PHASE_OF_FLIGHT": phase,
                        "SPECIES": species,
                        "TIME_OF_DAY": time_of_day,
                        "HEIGHT": altitude,
                    }
                ])

                # Run prediction
                pred = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0]

                # Style the prediction output banner
                if pred == 0:
                    css_class = "result-card-none"
                    emoji = "🟢"
                    rec = "No structural damage is expected based on historical incidents under these conditions."
                elif pred == 1:
                    css_class = "result-card-minor"
                    emoji = "🟡"
                    rec = "Minor damage expected. Recommend inspecting nose cone, wings, or engine cowl before next flight."
                else:
                    css_class = "result-card-severe"
                    emoji = "🔴"
                    rec = "High risk of severe damage. Immediate inspection, turbine check, or landing abort is strongly recommended!"

                severity_text = SEVERITY_LABELS.get(int(pred), f"Class {pred}")

                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <h3 style='margin:0; color:#fff;'>{emoji} Predicted Damage: {severity_text}</h3>
                        <p style='margin: 8px 0 0 0; font-size:0.95rem; opacity:0.9;'>{rec}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Probabilities visual progress bars
                st.write("")
                st.markdown("#### Class Probabilities Breakdown:")
                
                colors = ["#2e7559", "#dfa33e", "#db4455"] # green, yellow, red
                for i, prob in enumerate(proba):
                    label = SEVERITY_LABELS[i]
                    pct = prob * 100
                    st.write(f"**{label}**: {pct:.2f}%")
                    st.markdown(
                        f"""
                        <div class="prog-container">
                            <div class="prog-bar" style="width: {pct}%; background-color: {colors[i]};"></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # --- TAB 3: SEARCH ENGINE ---
    with tab_search:
        st.markdown("### 🔍 NTSB Aviation Accident Search Engine")
        st.write("Search historical accident reports using natural language. Ranking is done using TF-IDF & Vector Space cosine similarity.")

        if not docs:
            st.error("No documents found in the `docs` folder. Please populate the folder with txt reports.")
        else:
            # Query entry
            query = st.text_input(
                "Enter search terms (e.g. 'bird strike engine failure', 'landing gear', 'communication loss'):",
                placeholder="Type here..."
            )

            if query.strip():
                # Process search
                results = search(query, docs, inverted_index, idf, doc_lengths, k=10)
                query_tokens = tokenize(query)

                if not results:
                    st.warning("No reports matched your search criteria.")
                else:
                    st.markdown(f"Found **{len(results)}** matching reports:")

                    for i, (name, score, snippet) in enumerate(results, start=1):
                        highlighted_snippet = highlight_keywords(snippet, query_tokens)
                        full_text = docs[next(k for k, v in docs.items() if v["name"] == name)]["text"]
                        
                        # Style matching percentage
                        score_pct = score * 100

                        st.markdown(
                            f"""
                            <div class="skyshield-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div class="skyshield-title">📄 {name}</div>
                                    <div class="skyshield-badge">Similarity Score: {score_pct:.2f}%</div>
                                </div>
                                <p style="font-size:0.95rem; font-style:italic; margin-top:10px; color:#c5d1e0;">
                                    Snippet: {highlighted_snippet}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Full Report Expander
                        with st.expander(f"View Full Report for {name}"):
                            st.text_area(
                                label="Report Content",
                                value=full_text,
                                height=250,
                                key=f"text_area_{name}_{i}"
                            )

if __name__ == "__main__":
    main()
