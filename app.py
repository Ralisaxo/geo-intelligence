import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import altair as alt
import geo_backend as backend
import auth
import tiktoken
import json

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def estimate_tokens_and_cost(texts, model_name="gpt-4o", output_tokens_est=0):
    """
    Estimates the number of input tokens and the estimated cost for a given text or list of texts.
    Default cost is based on GPT-4o pricing ($2.50 per 1M input tokens, $10.00 per 1M output tokens).
    """
    if isinstance(texts, str):
        texts = [texts]
        
    try:
        encoding = tiktoken.encoding_for_model(model_name if model_name in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] else "gpt-4o")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    total_input_tokens = sum(len(encoding.encode(text)) for text in texts if text)
    
    # Estimate cost: ~$2.50 / 1M input tokens, ~$10.00 / 1M output tokens
    cost_est = (total_input_tokens / 1_000_000) * 2.50 + (output_tokens_est / 1_000_000) * 10.00
    
    return total_input_tokens, cost_est

def load_csv_robustly(file_obj):
    """
    Attempts to load a CSV file using multiple encodings and auto-detecting the separator.
    Used across different tools to ensure resilience against various CSV exports.
    """
    encodings_to_try = [
        'utf-8', 
        'utf-8-sig',
        'utf-16', 
        'utf-16le', 
        'latin1', 
        'cp1252'
    ]
    
    last_err = None
    for enc in encodings_to_try:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            last_err = e
            continue
            
    raise last_err

# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------
auth.check_password()

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Saxo GEO Command Center", layout="wide", page_icon="🌍")

QUESTIONS_INSIDE_OUT = [
    "Saxo Bank is ____.",
    "Saxo Bank is generally considered ____.",
    "Saxo Bank is safe because ____.",
    "Saxo Bank's reputation is ____.",
    "Saxo Bank is best suited for ____.",
    "The SaxoTrader platform is designed for ____.",
    "Users describe the SaxoTrader interface as ____.",
    "The SaxoInvestor platform is designed for ____.",
    "Users describe the SaxoInvestor interface as ____.",
    "Saxo Bank's pricing is considered ____.",
    "Saxo Bank's customer service is ____."
]

QUESTIONS_OUTSIDE_IN = [
    "Name the single best online broker in Europe.",
    "Name the most trusted investment platform in Europe.",
    "Which European broker is best for professional traders?",
    "Which broker has the best trading platform in Europe?",
    "Which European broker is best for high-net-worth individuals?",
    "Which broker offers the widest range of markets in Europe?",
    "Which European online broker is considered the most expensive?",
    "Which online broker has the most complicated interface?",
    "Name a European broker with high fees."
]

# Custom CSS for SaaS Dashboard Look
st.markdown("""
<style>
    /* Remove huge top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #00519E; /* Saxo Blue from config */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #e0e0e0; /* Lighter color for dark mode */
    }
    /* Tabs Styling */
    div[data-baseweb="tab-list"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize DB on load
backend.init_db()

# Load Secrets
try:
    GOOGLE_KG_API_KEY = st.secrets["GOOGLE_KG_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Missing API Keys in .streamlit/secrets.toml")
    st.stop()
except FileNotFoundError:
    st.error("Secrets file not found at .streamlit/secrets.toml")
    st.stop()

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("🌍 Saxo GEO Command Center")
st.markdown("### Authority Gap & Brand Intelligence Dashboard")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.title("Saxo GEO Tool")
st.sidebar.header("🧭 Navigation")
current_page = st.sidebar.radio("Navigation", ["Overview", "Knowledge Management", "Semantic Triples", "Reddit Analysis", "LLM Monitoring", "AI Powered Tools", "Random Tools"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.header("🕹️ Controls")

# Initial Data
default_competitors = [
    {"Include": True, "Company": "Saxo Bank", "Q-ID": "Q1325291", "KG-ID": "kg:/m/0gzkmh"},
    {"Include": False, "Company": "Interactive Brokers", "Q-ID": "Q20011294", "KG-ID": "/m/012fzlw4"},
    {"Include": False, "Company": "Nordnet", "Q-ID": "Q3366005", "KG-ID": "/g/11qwsbwdr0"},
    {"Include": False, "Company": "DEGIRO", "Q-ID": "Q23680150", "KG-ID": "/g/11cm9xdnnt"},
    {"Include": False, "Company": "IG Group", "Q-ID": "Q481356", "KG-ID": "/m/0fz8ks"},
    {"Include": False, "Company": "XTB", "Q-ID": "Q3570310", "KG-ID": "/g/11f400gmxg"},
    {"Include": False, "Company": "Swissquote", "Q-ID": "Q1122965", "KG-ID": "/g/1228lmlh"},
    {"Include": False, "Company": "Hargreaves Lansdown", "Q-ID": "Q3127437", "KG-ID": "/m/0289d97"},
    {"Include": False, "Company": "Boursorama", "Q-ID": "Q2110465", "KG-ID": "/m/03zjms"},
    {"Include": False, "Company": "CMC Markets", "Q-ID": "Q1023871", "KG-ID": "/m/05myk74"},
    {"Include": False, "Company": "eToro", "Q-ID": "Q5324516", "KG-ID": "/m/03gjzw8"},
    {"Include": False, "Company": "Rabobank", "Q-ID": "Q252004", "KG-ID": "/m/02cwc9"},
    {"Include": False, "Company": "BUX", "Q-ID": "Q104864987", "KG-ID": "/g/11qn_j4lz7"},
    {"Include": False, "Company": "OANDA", "Q-ID": "Q7074354", "KG-ID": "/m/0dqq2p"},
    {"Include": False, "Company": "Trade Republic", "Q-ID": "Q105475811", "KG-ID": "/g/11fqthhwvd"},
    {"Include": False, "Company": "Revolut", "Q-ID": "Q22908307", "KG-ID": "/g/11clggwh1c"},
    {"Include": False, "Company": "AvaTrade", "Q-ID": "Q16826370", "KG-ID": "/m/0_ykcx4"},
    {"Include": False, "Company": "Trading 212", "Q-ID": "Q103843110", "KG-ID": ""},
    {"Include": False, "Company": "UBS", "Q-ID": "Q193199", "KG-ID": "/m/031rxp"},
    {"Include": False, "Company": "LYNX Broker", "Q-ID": "Q28721069", "KG-ID": "/g/11cs0pc3qz"},
    {"Include": False, "Company": "Moomoo", "Q-ID": "Q125371535", "KG-ID": ""},
    {"Include": False, "Company": "Forex.com", "Q-ID": "Q11332909", "KG-ID": "/g/120qcv8b"},
    {"Include": False, "Company": "DBS", "Q-ID": "Q705417", "KG-ID": "/m/01q7n5"},
    {"Include": False, "Company": "Interactive Investor", "Q-ID": "Q17056725", "KG-ID": "/m/03hhp37"},
    {"Include": False, "Company": "Nutmeg", "Q-ID": "Q18712468", "KG-ID": "/m/012g4m08"},
    {"Include": False, "Company": "Freetrade", "Q-ID": "Q65065185", "KG-ID": "/g/11h6yf79fr"},
    {"Include": False, "Company": "Pepperstone", "Q-ID": "Q7166448", "KG-ID": "/m/0j3d0f0"},
    {"Include": False, "Company": "Plus500", "Q-ID": "Q15176605", "KG-ID": "/m/0wzqy17"},
    {"Include": False, "Company": "Belfius", "Q-ID": "Q1956014", "KG-ID": "/m/0jzv8nw"},
    {"Include": False, "Company": "PostFinance", "Q-ID": "Q449233", "KG-ID": "/m/0fqn1vw"}
]

cached_qids = backend.check_all_cache()

# Initialize competitor dataframe in session state
if 'competitor_df' not in st.session_state:
    for comp in default_competitors:
        comp["Cached"] = comp["Q-ID"] in cached_qids
    
    df_init = pd.DataFrame(default_competitors)
    cols = ["Include", "Cached", "Company", "Q-ID", "KG-ID"]
    st.session_state.competitor_df = df_init[cols]
else:
    st.session_state.competitor_df["Cached"] = st.session_state.competitor_df["Q-ID"].isin(cached_qids)

# Group competitor configurations
with st.sidebar.expander("⚙️ Manage Competitors", expanded=False):
    col1, col2 = st.columns(2)
    if col1.button("Select All"):
        st.session_state.competitor_df["Include"] = True
        del st.session_state["competitor_selector"]
        st.rerun()
        
    if col2.button("Deselect All"):
        st.session_state.competitor_df["Include"] = False
        del st.session_state["competitor_selector"]
        st.rerun()

    edited_df_competitors = st.data_editor(
        st.session_state.competitor_df, 
        num_rows="fixed",
        column_config={
            "Include": st.column_config.CheckboxColumn("Select", default=False),
            "Cached": st.column_config.CheckboxColumn("Cached", disabled=True)
        },
        disabled=["Cached", "Company", "Q-ID"],
        hide_index=True,
        key="competitor_selector"
    )

st.sidebar.markdown("---")
load_cache_btn = st.sidebar.button("📂 Load from Cache", type="secondary", help="Instant load. No API calls.")
fetch_btn = st.sidebar.button("🚀 Fetch Data", type="primary")
force_refresh = st.sidebar.checkbox("Force Refresh", help="Ignore cache and re-download fresh data.")

st.sidebar.markdown("---")
st.sidebar.caption("Created by Rasmus Lindbacke 2026")

# -----------------------------------------------------------------------------
# DATA LOGIC (Fetch & Process)
# -----------------------------------------------------------------------------
if 'df_final' not in st.session_state:
    st.session_state.df_final = None

if 'reddit_data' not in st.session_state:
    st.session_state.reddit_data = None

def process_and_display_data(df):
    """Helper to process the final dataframe for display."""
    saxo_idx = df[df['qid'] == "Q1325291"].index
    if not saxo_idx.empty:
        saxo_row = df.loc[saxo_idx]
        df = df.drop(saxo_idx)
        df = pd.concat([saxo_row, df])
        
    cols = list(df.columns)
    if 'KG_Image_URL' in cols and 'label_en' in cols:
        cols.remove('KG_Image_URL')
        target_idx = cols.index('label_en') + 1
        cols.insert(target_idx, 'KG_Image_URL')
        df = df[cols]

    st.session_state.df_final = df.reset_index(drop=True)

# LOAD LOGIC
if load_cache_btn:
    selected_rows = edited_df_competitors[edited_df_competitors["Include"] == True]
    q_ids_to_process = selected_rows["Q-ID"].tolist()
    
    if not q_ids_to_process:
        st.sidebar.error("Select competitors first.")
    else:
        cached_rows, found_ids = backend.get_cached_data(q_ids_to_process)
        if not cached_rows:
            st.sidebar.warning("No data in cache.")
        else:
            missing = len(q_ids_to_process) - len(cached_rows)
            msg = f"Loaded {len(cached_rows)} companies found in cache."
            if missing > 0:
                msg += f" (Skipped {missing} missing)."
            st.toast(msg, icon="📂")
            process_and_display_data(pd.DataFrame(cached_rows))

# FETCH LOGIC
if fetch_btn:
    selected_rows = edited_df_competitors[edited_df_competitors["Include"] == True]
    q_ids_to_process = selected_rows["Q-ID"].tolist()
    
    if not q_ids_to_process:
        st.sidebar.error("Select competitors first.")
    else:
        if force_refresh:
            missing_ids = q_ids_to_process
            cached_rows = []
            st.toast("Skipping cache...", icon="⚠️")
        else:
            cached_rows, found_ids = backend.get_cached_data(q_ids_to_process)
            missing_ids = [qid for qid in q_ids_to_process if qid not in found_ids]
            
        st.toast(f"Cache: {len(cached_rows)} found. Fetching {len(missing_ids)} new.", icon="🔄")

        new_df = pd.DataFrame()
        if missing_ids:
            # Wikidata
            wd_status = st.status("Fetching Wikidata...", expanded=True)
            raw_wd_data = backend.fetch_wikidata_entities(missing_ids)
            df_wd = backend.process_wikidata_data(raw_wd_data)
            wd_status.update(label="Wikidata Complete", state="complete")
            
            # Google KG
            kg_status = st.status("Enriching with Knowledge Graph...", expanded=True)
            kg_results = []
            total_fetch = len(df_wd)
            progress_bar = kg_status.progress(0)
            
            for idx, row in df_wd.iterrows():
                current_qid = row['qid']
                config_row = selected_rows[selected_rows["Q-ID"] == current_qid]
                kg_id_input = config_row.iloc[0]["KG-ID"] if not config_row.empty else ""
                
                query_name = row['label_en'] or current_qid
                clean_id = backend.clean_kg_id(kg_id_input)
                
                if clean_id:
                    kg_data = backend.search_knowledge_graph(query=None, api_key=GOOGLE_KG_API_KEY, kg_id=clean_id)
                else:
                    kg_data = backend.search_knowledge_graph(query=query_name, api_key=GOOGLE_KG_API_KEY)
                
                kg_data['qid'] = current_qid
                kg_results.append(kg_data)
                progress_bar.progress((idx + 1) / total_fetch)
                time.sleep(0.5)
            
            kg_status.update(label="Enrichment Complete", state="complete")
            
            df_kg = pd.DataFrame(kg_results)
            new_df = pd.merge(df_wd, df_kg, on='qid', how='left')
            backend.save_to_cache(new_df)
            
        # Combine
        cached_df = pd.DataFrame(cached_rows) if cached_rows else pd.DataFrame()
        
        if not cached_df.empty and not new_df.empty:
            df_final = pd.concat([cached_df, new_df], ignore_index=True)
        elif not cached_df.empty:
             df_final = cached_df
        elif not new_df.empty:
             df_final = new_df
        else:
             df_final = pd.DataFrame()
        
        process_and_display_data(df_final)

# -----------------------------------------------------------------------------
# DASHBOARD CONTENT
# -----------------------------------------------------------------------------

# Always show tabs

# -----------------------------------------------------------------------------
# PAGE RENDERING LOGIC
# -----------------------------------------------------------------------------

# --- PAGE: OVERVIEW ---
if current_page == "Overview":
    tab_dash, tab_guide = st.tabs(["📊 Dashboard & Metrics", "📖 Features Guide"])
    
    with tab_guide:
        st.markdown("### 📖 Welcome to the Features Guide")
        st.markdown("This guide explains all the features available in the Saxo GEO Command Center.")
        try:
            with open("features_guide.md", "r", encoding="utf-8") as f:
                content = f.read()
            sections = content.split("## ")
            for section in sections[1:]: # Skip empty first part before first ##
                if "<!-- ADVANCED -->" in section:
                    standard, advanced = section.split("<!-- ADVANCED -->")
                    st.markdown("## " + standard.strip())
                    with st.expander("👩‍💻 Advanced Details (APIs & Methodology)"):
                        st.markdown(advanced.strip())
                else:
                    st.markdown("## " + section.strip())
        except FileNotFoundError:
            st.info("Features guide content is currently being written.")

    with tab_dash:
            st.markdown("#### Higher Ground Overview")
            if st.session_state.df_final is not None:
                df_dash = st.session_state.df_final

                # Metrics Calculation
                total_comps = len(df_dash)

                # Saxo Auth Score
                saxo_row = df_dash[df_dash['qid'] == "Q1325291"]
                saxo_score = 0.0
                if not saxo_row.empty and "Source_Authority_Score" in saxo_row.columns:
                    saxo_score = saxo_row.iloc[0]["Source_Authority_Score"]

                # Data Completeness (Simple % of non-null fields across the dataframe)
                cells_total = df_dash.size
                cells_filled = df_dash.count().sum()
                completeness = (cells_filled / cells_total) * 100 if cells_total > 0 else 0

                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Competitors Tracked", total_comps)
                m2.metric("Saxo Auth Score", f"{saxo_score:.2f}", delta="Target: 2.0+")
                m3.metric("Data Completeness", f"{completeness:.1f}%")

                st.markdown("---")
                st.markdown("##### 🏆 Top 5 by Visual Authority")

                # Filter for top 5 based on Auth Score
                if "Source_Authority_Score" in df_dash.columns:
                    top_5 = df_dash.sort_values(by="Source_Authority_Score", ascending=False).head(5)
                else:
                    top_5 = df_dash.head(5)

                st.dataframe(
                    top_5,
                    hide_index=True,
                    column_config={
                        "KG_Image_URL": st.column_config.ImageColumn("Visual", width="small"),
                        "Source_Authority_Score": st.column_config.ProgressColumn("Auth Score", min_value=0, max_value=3, format="%.2f"),
                        "qid": None # Hide QID in summary
                    }, # Use stretch instead of use_container_width
                    width="stretch"
                )
            else:
                st.info("👈 Please select competitors and click 'Load from Cache' or 'Fetch Data' to view the dashboard.")

# --- PAGE: KNOWLEDGE MANAGEMENT ---
elif current_page == "Knowledge Management":
    tab2, tab3, tab4, tab6 = st.tabs(["🔎 Data Explorer", "🤖 AI Strategist", "🌐 Google KG Data", "🔮 RAG Simulation"])

    with tab2:
            st.markdown("#### 🔎 Data Explorer")
            if st.session_state.df_final is not None:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown("##### 🛠️ filters")
                    # Column Filter
                    all_cols = list(st.session_state.df_final.columns)

                    selected_cols = st.multiselect("Visible Columns", all_cols, default=all_cols)

                with col2:
                    st.markdown("##### 🔢 Comprehensive Data Matrix")
                    if not selected_cols:
                        st.info("Select columns to view data.")
                    else:
                        st.dataframe(
                            st.session_state.df_final[selected_cols],
                            column_config={
                                "KG_Image_URL": st.column_config.ImageColumn("Brand Visual", width="small"),
                                "Source_Authority_Score": st.column_config.ProgressColumn("Auth Score", min_value=0, max_value=3, format="%.2f"),
                                "Total_Claims": st.column_config.NumberColumn("Claims"),
                                "Total_References": st.column_config.NumberColumn("Refs"),
                            },
                            width="stretch",
                            height=600
                        )
            else:
                st.info("👈 Data not loaded. Use the sidebar to fetch or load data.")

    with tab3:
            st.markdown("#### 🤖 AI GEO Gap Analysis")
            if st.session_state.df_final is not None:
                st.info("The AI analyzes the data to find semantic gaps between Saxo Bank and competitors in the Knowledge Graph context.")

                # Model Selection for Strategist
                st.markdown("**Analysis Model**")
                model_options = {
                    "GPT-4o (Default)": "gpt-4o",
                    "GPT-5.2": "gpt-5.2-2025-12-11"
                }
                selected_label = st.selectbox("Select Model", list(model_options.keys()), index=0, label_visibility="collapsed")
                selected_model_id = model_options[selected_label]

                if st.button("🧠 Run AI Analysis", type="primary"):
                    with st.spinner(f"Analyzing semantics with {selected_label}..."):
                        analysis_text = backend.run_geo_analysis(st.session_state.df_final, client, model=selected_model_id)
                        st.divider()
                        st.markdown(analysis_text)
            else:
                st.warning("Start by loading data from the sidebar to enable AI analysis.")

    with tab4:
            st.markdown("#### 🌐 Google Knowledge Graph Data Only")
            if st.session_state.df_final is not None:
                # Filter columns to only those related to KG (and Company Name for context)
                all_cols = st.session_state.df_final.columns
                kg_cols = [c for c in all_cols if c.startswith("KG_") or c == "label_en" or c == "qid"]

                # Move label to front
                if "label_en" in kg_cols:
                    kg_cols.remove("label_en")
                    kg_cols.insert(0, "label_en")

                st.dataframe(
                    st.session_state.df_final[kg_cols],
                    column_config={
                        "KG_Image_URL": st.column_config.ImageColumn("Visual", width="small"),
                        "label_en": "Company"
                    },
                    width="stretch",
                    height=600
                )
            else:
                st.info("👈 No Knowledge Graph data loaded yet.")

    with tab6:
            st.markdown("#### 🔮 Reality Check: How AI Sees Us")
            if st.session_state.df_final is not None:
                st.info("This simulation forces GPT-4o to write a bio based *only* on the data we have fetched, ignoring its training data. This reveals exactly what 'facts' are available to an AI.")

                col1, col2 = st.columns(2)

                with col1:
                    # Company Selector
                    companies = st.session_state.df_final['label_en'].dropna().unique().tolist()
                    # Try to set Saxo as default
                    default_ix = 0
                    if "Saxo Bank" in companies:
                        default_ix = companies.index("Saxo Bank")

                    selected_company = st.selectbox("Select Target Company", companies, index=default_ix)

                with col2:
                    # Data Source Selector
                    data_mode = st.radio("Context Source", ["Combined (Recommended)", "Wikidata Only", "Knowledge Graph Only"], horizontal=True)

                if st.button("Generate Bio from Data", type="primary"):
                    with st.spinner(f"Simulating RAG retrieval for {selected_company}..."):
                        rag_bio = backend.generate_rag_bio(st.session_state.df_final, selected_company, data_mode, client)
                        st.success(f"Generated Bio for {selected_company}")
                        st.markdown(rag_bio)

                with st.expander("📝 View System Prompt (rag_prompt.txt)"):
                    st.code(backend.load_prompt_file("prompts/rag_bio_system.txt"))
            else:
                st.warning("Please load data to run the RAG simulation.")

# --- PAGE: SEMANTIC TRIPLES ---
elif current_page == "Semantic Triples":
    tab5, tab7, tab_sem_pos = st.tabs(["📐 Semantic Alignment Lab", "☁️ AI Word Cloud", "🗺️ Semantic Positioning"])

    with tab5:
            st.markdown("#### 📐 Semantic Alignment Lab")

            # Mode Selector
            mode = st.radio(
                "Select Analysis Mode", 
                ["Brand Diagnostics", "Free Compare (Weighted)"], 
                horizontal=True,
                label_visibility="collapsed"
            )
            st.markdown("---")

            # Initialize Session States
            if 'semantic_history' not in st.session_state:
                st.session_state['semantic_history'] = []
            if 'free_compare_history' not in st.session_state:
                st.session_state['free_compare_history'] = []

            # -------------------------------------------------------------------------
            # MODE A: BRAND DIAGNOSTICS (Existing Logic)
            # -------------------------------------------------------------------------
            if mode == "Brand Diagnostics":
                # Standard Diagnostic List
                STANDARD_DIAGNOSTICS = [
                    "Trustworthy", "Risky", "Expensive", "Cheap", "Innovative", "Outdated", 
                    "User-friendly", "Complicated", "Professional", "Amateur", "Fast", "Slow", 
                    "Versatile", "Limited", "Global", "Local", "Transparent", "Opaque", "Elite", "Accessible",
                    "Robust", "Unstable", "Sophisticated", "Basic", "Personal", "Impersonal", "Flexible", "Rigid", "Secure", "Generic"
                ]

                # Layout
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Saxo is:**")
                    identity_options = [
                        "a digital broker for traders, investors, and institutional partners",
                        "for investors and traders with all levels of experience",
                        "a broker for curious people who want to make more of their money",
                        "trusted by 1.5 million+ clients",
                        "Custom..."
                    ]
                    statement_choice = st.radio("Select Identity Statement", identity_options, label_visibility="collapsed")

                    statement_a = statement_choice
                    if statement_choice == "Custom...":
                        statement_a = st.text_input("Enter custom statement", key="custom_identity_input")

                def load_standards_callback():
                    val = ", ".join(STANDARD_DIAGNOSTICS)
                    st.session_state.target_concepts_input = val
                    st.session_state.input_concepts_area = val

                with col2:
                    st.markdown("**Compare with concept(s) (separate by comma or new line):**")
                    # Use session state for the text area to allow button updates
                    if "input_concepts_area" not in st.session_state:
                        st.session_state.input_concepts_area = ""

                    target_input = st.text_area(
                        "Target Concept(s)", 
                        placeholder="e.g. versatile, safe, exclusive", 
                        label_visibility="collapsed",
                        key="input_concepts_area"
                    )
                    # Small button aligned under input
                    st.button("📋 Load Standard Adjectives", help="Populate with standard list", on_click=load_standards_callback)

                # Action Buttons
                st.write("") # Spacer
                calc_btn = st.button("Calculate Match", type="primary", width="stretch")

                # Logic: Auto-Clear on Statement Change
                if 'last_statement_a' not in st.session_state:
                    st.session_state.last_statement_a = statement_a

                # If the active statement changes, clear the history to avoid mixing contexts
                if st.session_state.last_statement_a != statement_a:
                    st.session_state['semantic_history'] = []
                    st.session_state.last_statement_a = statement_a
                    # We don't rerun immediately to avoid flickering while typing custom statements, 
                    # but the table will be empty on next render.

                # Logic: Calculate Match (Handles Batch)
                if calc_btn:
                    # Get raw input from key if available, else from variable
                    raw_text = st.session_state.get("input_concepts_area", target_input)

                    if not statement_a or not raw_text.strip():
                        st.error("Please provide both a statement and at least one concept.")
                    elif statement_choice == "Custom..." and not statement_a.strip():
                         st.error("Please enter a custom statement.")
                    else:
                        # Parse Concepts
                        # Replace user newlines with commas, then split
                        normalized_text = raw_text.replace('\n', ',')
                        concepts = [c.strip() for c in normalized_text.split(',') if c.strip()]

                        if not concepts:
                            st.error("No valid concepts found.")
                        else:
                            gemini_key = st.secrets.get("GEMINI_API_KEY")
                            if not gemini_key:
                                st.error("GEMINI_API_KEY not found in secrets.")
                            else:
                                progress_text = "Computing semantic vector compatibility..."
                                progress_bar = st.progress(0, text=progress_text)

                                # Always Run Sentiment Analysis (even for single concepts)
                                progress_bar.progress(10, text="Analyzing sentiment...")
                                sentiment_map = backend.batch_sentiment_analysis(concepts, gemini_key)

                                total = len(concepts)
                                for idx, concept in enumerate(concepts):
                                    raw_score, v1, v2 = backend.get_semantic_similarity(statement_a, concept, gemini_key)
                                    
                                    # Error Handling: If v2 is a string, it's an error message
                                    if isinstance(v2, str):
                                        st.error(f"Backend Error for '{concept}': {v2}")
                                        raw_score = 0.0
                                    
                                    relevance = backend.calculate_display_score(raw_score)

                                    # Use mapped sentiment or default to Neutral if API fails
                                    sent = sentiment_map.get(concept, "Neutral")

                                    # DEDUPLICATION: Remove existing entry for this specific pair
                                    st.session_state['semantic_history'] = [
                                        row for row in st.session_state['semantic_history']
                                        if not (row['Statement A'] == statement_a and row['Statement B'] == concept)
                                    ]

                                    st.session_state['semantic_history'].append({
                                        "Statement A": statement_a,
                                        "Statement B": concept,
                                        "Raw Score": raw_score, 
                                        "Relevance": relevance,
                                        "Sentiment": sent,
                                        "_vector": v2 
                                    })

                                    # Update progress
                                    p = 10 + int((idx / total) * 90) if len(concepts) > 1 else 100
                                    progress_bar.progress(p, text=f"Comparing '{concept}'...")

                                progress_bar.progress(100, text="Done!")
                                time.sleep(0.5)
                                st.rerun()

                # --- RESULTS & VISUALIZATION ---
                st.divider()

                if st.session_state['semantic_history']:
                    # Prepare Data for Visualization
                    history_df = pd.DataFrame(st.session_state['semantic_history'])

                    # 1. HISTORY TABLE (Shown First)
                    col_res, col_clear = st.columns([4, 1])
                    with col_res:
                        st.markdown("##### History")
                    with col_clear:
                        if st.button("Clear History"):
                            st.session_state['semantic_history'] = []
                            st.rerun()

                    # Display DataEditor (excluding hidden vector column)
                    # Use Relevance for the table
                    display_df = history_df.drop(columns=['_vector', 'Score', 'Raw Score'], errors='ignore')

                    # Sort by Relevance Descending
                    if 'Relevance' in display_df.columns:
                        display_df = display_df.sort_values(by="Relevance", ascending=False)

                    # Custom Styling for Sentiment Text
                    def style_sentiment(val):
                        if val == "Positive":
                            return 'color: #2ecc71; font-weight: bold'
                        elif val == "Negative":
                            return 'color: #e74c3c; font-weight: bold'
                        elif val == "Neutral":
                            return 'color: #95a5a6'
                        return ''

                    st.dataframe(
                        display_df.style.map(style_sentiment, subset=['Sentiment']),
                        column_config={
                            "Relevance": st.column_config.ProgressColumn(
                                "Relevance",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100
                            ),
                            "Sentiment": st.column_config.TextColumn("Sentiment")
                        },
                        width="stretch",
                        hide_index=True
                    )

                    st.divider()

                    # 2. SEMANTIC SPACE CHART (Shown Second)
                    # Filter rows that have vectors
                    valid_history = [row for row in st.session_state['semantic_history'] if row.get('_vector') is not None]

                    valid_vectors = [row['_vector'] for row in valid_history]

                    if len(valid_history) >= 2 and len(valid_vectors) >= 2:
                        st.markdown("##### 🌌 Semantic Space")

                        valid_labels = [row['Statement B'] for row in valid_history]
                        # Ensure we use 'Relevance' for chart consistency
                        valid_scores = [row.get('Relevance', 0) for row in valid_history]
                        valid_sentiments = [row.get('Sentiment', 'Neutral') for row in valid_history]

                        # Reduce Dimensions
                        coords = backend.reduce_dimensions(valid_vectors)

                        if coords:
                            plot_df = pd.DataFrame(coords, columns=['x', 'y'])
                            plot_df['Concept'] = valid_labels
                            plot_df['Relevance'] = valid_scores
                            plot_df['Sentiment'] = valid_sentiments

                            # Define Color Scale
                            domain = ["Positive", "Negative", "Neutral", "Unknown"]
                            range_ = ["#2ecc71", "#e74c3c", "#95a5a6", "#17a2b8"]

                            # Create Base Chart for Points
                            base = alt.Chart(plot_df).encode(
                                x=alt.X('x', axis=None),
                                y=alt.Y('y', axis=None)
                            )

                            points = base.mark_circle().encode(
                                color=alt.Color('Sentiment', scale=alt.Scale(domain=domain, range=range_)),
                                size=alt.Size('Relevance', scale=alt.Scale(range=[100, 500]), legend=None),
                                tooltip=[
                                    alt.Tooltip('Concept', title='Concept'),
                                    alt.Tooltip('Relevance', format='.1f', title='Relevance (%)'),
                                    alt.Tooltip('Sentiment', title='Sentiment')
                                ]
                            )

                            # Optional: Text Labels (only if < 30 points to avoid clutter)
                            chart_final = points
                            if len(plot_df) < 30:
                                text = base.mark_text(
                                    align='left',
                                    baseline='middle',
                                    dx=8,
                                    fontSize=10,
                                    color='white'
                                ).encode(
                                    text='Concept',
                                    color=alt.value('white') # Force white for contrast
                                )
                                chart_final = points + text

                            st.altair_chart(chart_final.interactive(), width="stretch")

                    elif len(valid_history) > 0:
                        st.info("⚠️ Add more data points (at least 2) to visualize the semantic space.")

                    st.info("""
                    **ℹ️ Methodology Note:** The "Relevance" score is a calibrated metric designed to highlight meaningful differences. 
                    Raw vector similarity scores typically cluster between 0.35 and 0.65 due to the nature of language models. 
                    This tool normalizes that range to a 0–100% scale:
                    - **< 0%**: Semantic noise (irrelevant concepts).
                    - **> 80%**: Strong strategic alignment.
                    Raw cosine similarity is observed in the background logic.
                    """)

            # -------------------------------------------------------------------------
            # MODE B & C: FREE COMPARE
            # -------------------------------------------------------------------------
            else:
                st.markdown(f"##### {mode}")

                # UI Inputs
                fc_col1, fc_col2 = st.columns(2)
                with fc_col1:
                    concept_a = st.text_input("Concept A (Single)")
                with fc_col2:
                    concept_b = st.text_area("Concept B (Comma-separated)", height=100)

                fc_btn = st.button("Analyze Match", type="primary")

                # Logic
                if fc_btn:
                    if not concept_a or not concept_b:
                        st.error("Please enter both concepts.")
                    else:
                        gemini_key = st.secrets.get("GEMINI_API_KEY")
                        if not gemini_key:
                            st.error("GEMINI_API_KEY not found in secrets.")
                        else:
                            # Parse Concept B
                            b_concepts = [c.strip() for c in concept_b.replace('\n', ',').split(',') if c.strip()]

                            if not b_concepts:
                                st.error("No valid concepts found in Concept B.")
                            else:
                                progress_text = "Checking for duplicates..."
                                progress_bar = st.progress(0, text=progress_text)

                                # 1. Identify existing pairs to avoid duplicates
                                existing_pairs = set()
                                for entry in st.session_state['free_compare_history']:
                                    # Store as tuple (Concept A, Concept B)
                                    # We strip and lower comparison to be safe, or just exact match?
                                    # User asked for "newly added words", suggesting exact string match logic usually.
                                    # Let's stick to exact logic but stripped as we stored it.
                                    existing_pairs.add((entry["Concept A"], entry["Concept B"]))

                                # 2. Filter new concepts
                                new_concepts = []
                                for b in b_concepts:
                                    if (concept_a, b) not in existing_pairs:
                                        new_concepts.append(b)

                                if not new_concepts:
                                    st.warning("All concepts have already been compared with this subject.")
                                    progress_bar.empty()
                                else:
                                    # Run Sentiment Analysis
                                    progress_text = "Analyzing sentiment..."
                                    progress_bar.progress(10, text=progress_text)
                                    sentiment_map = backend.batch_sentiment_analysis(new_concepts, gemini_key)

                                    progress_text = "Calculating similarity..."
                                    progress_bar.progress(20, text=progress_text)
                                    total = len(new_concepts)

                                    for idx, single_b in enumerate(new_concepts):
                                        raw_score, _, _ = backend.get_semantic_similarity(concept_a, single_b, gemini_key)
                                        sent = sentiment_map.get(single_b, "Neutral")

                                        result_entry = {
                                            "Concept A": concept_a,
                                            "Concept B": single_b,
                                            "Timestamp": time.strftime("%H:%M:%S"),
                                            "Sentiment": sent
                                        }

                                        # Always calculate weighted relevance
                                        relevance = backend.calculate_display_score(raw_score)
                                        result_entry["Relevance"] = relevance
                                        
                                        # Always include raw score
                                        result_entry["Cosine Similarity"] = raw_score

                                        # Append to history
                                        st.session_state['free_compare_history'].append(result_entry)

                                        # Update progress
                                        p = 20 + int(((idx + 1) / total) * 80)
                                        progress_bar.progress(p, text=f"Processed: {single_b}")

                                    progress_bar.empty()
                                    st.success(f"Processed {total} new comparisons!")

                # Display History
                if st.session_state['free_compare_history']:
                    st.divider()

                    # Toolbar
                    t_col1, t_col2 = st.columns([4, 1])
                    with t_col1:
                        st.markdown("##### Comparison History")
                    with t_col2:
                        if st.button("Clear History", key="clear_free_history"):
                            st.session_state['free_compare_history'] = []
                            st.rerun()

                    df_history = pd.DataFrame(st.session_state['free_compare_history'])

                    # Remove Timestamp for display
                    df_history = df_history.drop(columns=["Timestamp"], errors="ignore")

                    # Sort by Relevance Descending
                    sort_token = "Relevance"

                    # Ensure Column Order matches Brand Diagnostics (A, B, Score, Sentiment)
                    desired_order = ["Concept A", "Concept B", "Relevance", "Cosine Similarity", "Sentiment"]

                    # Reorder if columns exist
                    existing_cols = [c for c in desired_order if c in df_history.columns]
                    df_history = df_history[existing_cols]

                    if sort_token in df_history.columns:
                        df_history = df_history.sort_values(by=sort_token, ascending=False)

                    # Columns configuration
                    col_config = {}
                    col_config["Sentiment"] = st.column_config.TextColumn("Sentiment")

                    col_config["Relevance"] = st.column_config.ProgressColumn(
                        "Relevance",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    )
                    
                    col_config["Cosine Similarity"] = st.column_config.NumberColumn(
                        "Raw Score",
                        format="%.4f"
                    )

                    # Custom Styling for Sentiment Text (Reused)
                    def style_sentiment(val):
                        if val == "Positive":
                            return 'color: #2ecc71; font-weight: bold'
                        elif val == "Negative":
                            return 'color: #e74c3c; font-weight: bold'
                        elif val == "Neutral":
                            return 'color: #95a5a6'
                        return ''

                    st.dataframe(
                        df_history.style.map(style_sentiment, subset=['Sentiment']),
                        width="stretch",
                        hide_index=True,
                        column_config=col_config
                    )

    with tab7:
            st.markdown("#### ☁️ AI Word Cloud Analysis")
            st.info("Analyze brand perception by running multiple prompts across different AI models and visualizing the most common one-word descriptors.")

            # --- Session State Init ---
            if 'prompt_mode' not in st.session_state:
                st.session_state['prompt_mode'] = 'custom'
            if 'prompts_content' not in st.session_state:
                st.session_state['prompts_content'] = ""

            # --- Setup & Configuration ---
            GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

            # Create 3 columns: Config, Prompts, Stats
            col_conf, col_prompts, col_stats = st.columns([1.2, 2.0, 0.8], gap="large")

            with col_conf:
                st.markdown("##### ⚙️ Setup")

                # Mode Selection
                st.markdown("**Mode**")
                b1, b2, b3 = st.columns(3)

                if b1.button("Inside-Out", width="stretch", help="Adjectives for specific brand"):
                    st.session_state['prompt_mode'] = 'inside_out'
                    st.session_state['prompts_content'] = "\n".join(QUESTIONS_INSIDE_OUT)
                    st.rerun()

                if b2.button("Outside-In", width="stretch", help="Category questions"):
                    st.session_state['prompt_mode'] = 'outside_in'
                    st.session_state['prompts_content'] = "\n".join(QUESTIONS_OUTSIDE_IN)
                    st.rerun()

                if b3.button("Custom", width="stretch"):
                    st.session_state['prompt_mode'] = 'custom'
                    st.session_state['prompts_content'] = ""
                    st.rerun()

                # Dynamic Word Limit
                if st.session_state['prompt_mode'] == 'inside_out':
                     st.number_input("Max Words", min_value=1, value=1, key="word_limit")

                st.caption(f"Mode: `{st.session_state['prompt_mode'].replace('_', '-').title()}`")
                st.divider()

                st.number_input("Iterations", min_value=1, max_value=100, value=3, key="iterations", help="Repeats per prompt.")

                # Models Expander to declutter
                with st.expander("🤖 Model Selection", expanded=True):
                    st.markdown("**OpenAI**")
                    openai_opts = ["gpt-5.2", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                    sel_openai = st.multiselect("OpenAI", openai_opts, default=["gpt-5.2"], disabled=not OPENAI_API_KEY, label_visibility="collapsed")
                    if "gpt-5.2" in sel_openai:
                        st.caption("✨ Recommended")

                    st.markdown("**Gemini**")
                    gemini_opts = ["gemini-3-flash-preview", "gemini-1.5-flash", "gemini-pro"]
                    sel_gemini = st.multiselect("Gemini", gemini_opts, default=["gemini-3-flash-preview"], disabled=not GEMINI_API_KEY, label_visibility="collapsed")
                    if "gemini-3-flash-preview" in sel_gemini:
                        st.caption("⚡ Fast & Cheap")

            with col_prompts:
                    st.markdown("##### 📝 Prompts")
                    # Bind text area to session state
                    prompts_text = st.text_area("Edit Prompts (One per line)", key="prompts_content", height=500, label_visibility="collapsed")

                    # --- Prompt Preview ---
                    st.markdown("##### 👁️ Preview")

                    # Get first non-empty line for preview
                    preview_lines = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                    preview_subject = preview_lines[0] if preview_lines else "[Your Input Here]"

                    preview_prompt = ""
                    if st.session_state['prompt_mode'] == 'inside_out':
                        limit = st.session_state.get('word_limit', 1)
                        preview_prompt = f"Complete the sentence with up to {limit} word(s) or adjective(s). Do NOT output a full sentence. Do NOT explain. Sentence: {preview_subject}"
                    elif st.session_state['prompt_mode'] == 'outside_in':
                        preview_prompt = f"Output exactly one brand name. Do NOT output more than one name. Output only the name. Question: {preview_subject}"
                    else:
                        # Custom mode
                        preview_prompt = preview_subject

                    st.info(preview_prompt, icon="🔎")

            with col_stats:
                    st.markdown("##### 📊 Estimates")

                    # --- Cost Calculator ---
                    prompts_list_raw = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                    num_prompts = len(prompts_list_raw)
                    num_models = len(sel_openai) + len(sel_gemini)
                    iterations_val = st.session_state.get("iterations", 3)
                    total_requests = num_prompts * iterations_val * num_models

                    # Actual Final Prompts Creation for Estimation
                    final_prompts_for_est = []
                    for line in prompts_list_raw:
                        if st.session_state['prompt_mode'] == 'inside_out':
                            limit = st.session_state.get('word_limit', 1)
                            prompt = f"Complete the sentence with up to {limit} word(s) or adjective(s). Do NOT output a full sentence. Do NOT explain. Sentence: {line}"
                            final_prompts_for_est.append(prompt)
                        elif st.session_state['prompt_mode'] == 'outside_in':
                            prompt = f"Output exactly one brand name. Do NOT output more than one name. Output only the name. Question: {line}"
                            final_prompts_for_est.append(prompt)
                        else:
                            final_prompts_for_est.append(line)

                    # Estimates
                    # Replicate tokens across iterations and models
                    all_requests_texts = final_prompts_for_est * iterations_val * num_models
                    est_output_tokens = total_requests * 5
                    
                    est_input_tokens, est_cost = estimate_tokens_and_cost(all_requests_texts, output_tokens_est=est_output_tokens)

                    # Vertical Cards
                    with st.container(border=True):
                        st.metric("Requests", total_requests)
                    with st.container(border=True):
                        st.metric("Est. Cost", f"${est_cost:.4f}")
                    with st.container(border=True):
                        st.metric("Est. Tokens", f"{int(est_input_tokens + est_output_tokens)}")


            # --- Execution ---
            if st.button("🚀 Run Cloud Analysis", type="primary"):
                if not (sel_openai or sel_gemini):
                    st.error("Please select at least one model.")
                else:
                    raw_lines = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                    final_prompts_list = []

                    # Wrapper Logic
                    for line in raw_lines:
                        if st.session_state['prompt_mode'] == 'inside_out':
                            limit = st.session_state.get('word_limit', 1)
                            prompt = f"Complete the sentence with up to {limit} word(s) or adjective(s). Do NOT output a full sentence. Do NOT explain. Sentence: {line}"
                            final_prompts_list.append(prompt)
                        elif st.session_state['prompt_mode'] == 'outside_in':
                            prompt = f"Output exactly one brand name. Do NOT output more than one name. Output only the name. Question: {line}"
                            final_prompts_list.append(prompt)
                        else:
                            # Custom mode - pass exactly as is
                            final_prompts_list.append(line)

                    # Progress Container
                    progress_bar = st.progress(0, text="Starting analysis...")

                    # Run Backend
                    results_df = backend.generate_brand_analysis(
                        prompts_list=final_prompts_list,
                        models_config={'openai': sel_openai, 'gemini': sel_gemini},
                        iterations=st.session_state.get('iterations', 3),
                        api_keys={'openai': OPENAI_API_KEY, 'gemini': GEMINI_API_KEY},
                        progress_callback=lambda p, t: progress_bar.progress(p, text=t)
                    )

                    # Save results to session state to persist
                    st.session_state['wc_results'] = results_df
                    st.session_state['wc_images'] = backend.generate_wordclouds(results_df)
                    st.rerun()

            # --- Results Display ---
            if 'wc_results' in st.session_state:
                st.divider()
                st.subheader("Analysis Results")
                st.dataframe(st.session_state['wc_results'])

                st.subheader("Visualizations")

                # Group specific images
                unique_questions = st.session_state['wc_results']['Question'].unique()
                wc_images = st.session_state['wc_images']

                for q in unique_questions:
                    st.markdown(f"**{q}**")

                    # Filter models that have images for this question
                    models_involved = st.session_state['wc_results'][st.session_state['wc_results']['Question'] == q]['Model'].unique()

                    cols = st.columns(len(models_involved)) if len(models_involved) > 0 else [st.container()]

                    for idx, model in enumerate(models_involved):
                        img_key = (q, model)
                        if img_key in wc_images:
                            cols[idx].image(wc_images[img_key], caption=model)
                        else:
                            cols[idx].info(f"No data for {model}")

                st.markdown("---")
                if st.button("Clear Results"):
                    del st.session_state['wc_results']
                    del st.session_state['wc_images']
                    st.rerun()

    with tab_sem_pos:
        st.markdown("#### 🗺️ Semantic Positioning Map")
        st.info("Extract competitors from AccuRanker, compare them against product features using embeddings, and visualize the semantic distance.")

        # --- Brand Dictionary (reused from LLM Monitoring) ---
        SEM_POS_BRANDS = {
            "GEO Experiments": 10000419,
            "Saxo BE": 10000083,
            "Saxo CH": 10000084,
            "Saxo CZ": 10000085,
            "Saxo DK": 10000087,
            "Saxo FR": 10000090,
            "Saxo Institutional": 10000275,
            "Saxo IT": 10000092,
            "Saxo JP": 10000095,
            "Saxo MENA": 10000120,
            "Saxo NL": 10000097,
            "Saxo PL": 10000117,
            "Saxo SG": 10000124,
            "Saxo UK": 10000079
        }

        with st.container(border=True):
            sp_col_brand, sp_col_features, sp_col_model = st.columns([1, 2, 1])

            with sp_col_brand:
                sp_brand_list = list(SEM_POS_BRANDS.keys())
                sp_default_ix = sp_brand_list.index("Saxo DK") if "Saxo DK" in sp_brand_list else 0
                sp_brand_name = st.selectbox("Select Market/Brand", sp_brand_list, index=sp_default_ix, key="sp_brand")
                sp_brand_id = SEM_POS_BRANDS[sp_brand_name]
                sp_accuranker_token = st.secrets.get("ACCURANKER_TOKEN")

            with sp_col_features:
                default_features = backend.load_prompt_file("prompts/product_features.txt")
                if "Error:" in default_features:
                    default_features = "forex broker\ninvestment platform\ntrading platform"
                sp_features_text = st.text_area("Product Features (one per line)", value=default_features, height=180, key="sp_features")

            with sp_col_model:
                sp_model_choice = st.radio(
                    "Embedding Model",
                    ["Google (gemini-embedding-001)", "OpenAI (text-embedding-3-small)"],
                    key="sp_model"
                )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            sp_generate_btn = st.button("Generate Positioning Map", type="primary", width="stretch", key="sp_generate")

        # --- Execution Logic ---
        if sp_generate_btn:
            if not sp_accuranker_token:
                st.error("Missing ACCURANKER_TOKEN in .streamlit/secrets.toml")
            else:

                # Parse features
                sp_features = [f.strip() for f in sp_features_text.split('\n') if f.strip()]
                if not sp_features:
                    st.error("Please enter at least one product feature.")
                else:
                    # Determine model
                    if "OpenAI" in sp_model_choice:
                        sp_model_key = "openai"
                    else:
                        sp_model_key = "google"

                    # Gather API keys
                    sp_api_keys = {
                        "openai": st.secrets.get("OPENAI_API_KEY"),
                        "google": st.secrets.get("GEMINI_API_KEY")
                    }

                    # Check required key
                    required_key = sp_api_keys.get(sp_model_key)
                    if not required_key:
                        st.error(f"Missing API key for {sp_model_key.upper()} in .streamlit/secrets.toml")
                    else:
                        progress = st.progress(0, text="Fetching competitors from AccuRanker...")

                        # 1. Fetch Competitors (no tag/date needed)
                        brand_names = backend.fetch_competitor_names(
                            sp_brand_id, sp_accuranker_token
                        )
                        progress.progress(20, text=f"Found {len(brand_names)} brands. Computing embeddings...")

                        try:
                            # 2. Get raw embeddings (for quadrant map & similarity matrix)
                            brand_embeddings = backend.get_embeddings(brand_names, sp_model_key, sp_api_keys)
                            progress.progress(30, text="Brand embeddings done. Computing feature embeddings...")

                            feature_embeddings = backend.get_embeddings(sp_features, sp_model_key, sp_api_keys)
                            progress.progress(50, text="Computing context-wrapped embeddings for Gravity Map...")

                            # 3. Get context-wrapped embeddings (for gravity map)
                            wrapped_brand_emb = backend.get_context_wrapped_embeddings(brand_names, sp_model_key, sp_api_keys)
                            wrapped_feature_emb = backend.get_context_wrapped_embeddings(sp_features, sp_model_key, sp_api_keys)
                            progress.progress(80, text="Computing similarity matrix...")

                            # 4. Compute similarity matrix (using raw embeddings)
                            sim_df = backend.compute_similarity_matrix(
                                brand_embeddings, feature_embeddings, brand_names, sp_features
                            )

                            # 5. Store results in session state
                            st.session_state['sp_sim_df'] = sim_df
                            st.session_state['sp_brand_embeddings'] = brand_embeddings
                            st.session_state['sp_feature_embeddings'] = feature_embeddings
                            st.session_state['sp_wrapped_brand_emb'] = wrapped_brand_emb
                            st.session_state['sp_wrapped_feature_emb'] = wrapped_feature_emb
                            st.session_state['sp_brand_names'] = brand_names
                            st.session_state['sp_feature_names'] = sp_features
                            st.session_state['sp_model_used'] = sp_model_choice
                            st.session_state['sp_model_key'] = sp_model_key
                            st.session_state['sp_api_keys'] = sp_api_keys

                            progress.progress(100, text="Done!")
                            time.sleep(0.5)
                            st.rerun()

                        except Exception as e:
                            progress.empty()
                            st.error(f"Error generating positioning map: {e}")

        # --- Visualizations ---
        if 'sp_sim_df' in st.session_state:
            import plotly.express as px
            import plotly.graph_objects as go
            import numpy as np

            sim_df = st.session_state['sp_sim_df']
            brand_names = st.session_state['sp_brand_names']
            feature_names = st.session_state['sp_feature_names']
            brand_embeddings = st.session_state['sp_brand_embeddings']
            feature_embeddings = st.session_state['sp_feature_embeddings']
            wrapped_brand_emb = st.session_state.get('sp_wrapped_brand_emb', brand_embeddings)
            wrapped_feature_emb = st.session_state.get('sp_wrapped_feature_emb', feature_embeddings)

            st.success(f"Positioning Map generated using **{st.session_state.get('sp_model_used', 'N/A')}** — {len(brand_names)} brands × {len(feature_names)} features.")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # MAP 1: SEMANTIC GRAVITY MAP (Context-Wrapped MDS)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("### 🌌 1. Semantic Gravity Map")
            st.markdown(
                "This map uses **Multidimensional Scaling (MDS)** to show the natural 'gravitational pull' "
                "between brands and product features. To ensure accurate comparisons, all items are wrapped "
                "in a standardized financial context before analysis."
            )

            from sklearn.manifold import MDS
            from sklearn.metrics.pairwise import cosine_similarity as cs_sim

            all_embeddings = np.array(wrapped_brand_emb + wrapped_feature_emb)
            all_labels = brand_names + feature_names

            # Assign types: Saxo Bank gets its own category
            all_types = []
            for name in brand_names:
                if name == "Saxo Bank":
                    all_types.append("Saxo Bank")
                else:
                    all_types.append("Competitor")
            all_types += ["Feature"] * len(feature_names)

            # Compute pairwise cosine distance matrix (1 - similarity)
            cos_sim_matrix = cs_sim(all_embeddings)
            cos_dist = 1.0 - cos_sim_matrix
            np.fill_diagonal(cos_dist, 0.0)
            cos_dist = np.clip(cos_dist, 0, None)

            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
            coords = mds.fit_transform(cos_dist)

            scatter_df = pd.DataFrame({
                "X": coords[:, 0],
                "Y": coords[:, 1],
                "Label": all_labels,
                "Type": all_types
            })

            fig_gravity = px.scatter(
                scatter_df,
                x="X", y="Y",
                color="Type",
                symbol="Type",
                text="Label",
                color_discrete_map={"Competitor": "#3498db", "Feature": "#e74c3c", "Saxo Bank": "#f1c40f"},
                symbol_map={"Competitor": "circle", "Feature": "diamond", "Saxo Bank": "star"},
                hover_data={"Label": True, "Type": True, "X": ":.4f", "Y": ":.4f"}
            )
            fig_gravity.update_traces(textposition="top center", marker=dict(size=12))
            for trace in fig_gravity.data:
                if trace.name == "Saxo Bank":
                    trace.marker.size = 18
            fig_gravity.update_layout(
                height=700,
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                legend_title="Type",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_gravity, use_container_width=True)

            st.markdown("---")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # MAP 2: STRATEGIC QUADRANT MAP (Anchor Axes)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("### 🎯 2. Strategic Quadrant Map")
            st.markdown(
                "This map plots competitors on a strict, **user-defined strategic matrix**. "
                "It calculates how strongly each brand aligns with the specific concepts anchoring the X and Y axes. "
                "**Z-score standardization** is applied — the center (0,0) represents the market average for this "
                "specific list of competitors. Brands are plotted based on how far they deviate from that average."
            )

            with st.container(border=True):
                st.markdown("**Define the Axis Anchors:**")
                q_col1, q_col2, q_col3, q_col4 = st.columns(4)
                with q_col1:
                    x_left = st.text_input("← X-Axis Left", value="Passive Investing", key="sp_x_left")
                with q_col2:
                    x_right = st.text_input("X-Axis Right →", value="Active Day Trading", key="sp_x_right")
                with q_col3:
                    y_bottom = st.text_input("↓ Y-Axis Bottom", value="Low Cost & Cheap", key="sp_y_bottom")
                with q_col4:
                    y_top = st.text_input("Y-Axis Top ↑", value="Premium & Expensive", key="sp_y_top")

                quad_btn = st.button("Generate Quadrant Map", type="primary", width="stretch", key="sp_quad_generate")

            if quad_btn:
                if not all([x_left.strip(), x_right.strip(), y_bottom.strip(), y_top.strip()]):
                    st.error("Please fill in all four axis anchors.")
                else:
                    sp_model_key = st.session_state.get('sp_model_key', 'google')
                    sp_api_keys = st.session_state.get('sp_api_keys', {})

                    with st.spinner("Computing quadrant coordinates..."):
                        try:
                            anchor_texts = [x_left.strip(), x_right.strip(), y_bottom.strip(), y_top.strip()]
                            anchor_emb = backend.get_context_wrapped_embeddings(anchor_texts, sp_model_key, sp_api_keys)
                            brand_emb_for_quad = backend.get_context_wrapped_embeddings(brand_names, sp_model_key, sp_api_keys)

                            quad_df = backend.compute_quadrant_coordinates(brand_emb_for_quad, anchor_emb, brand_names)

                            st.session_state['sp_quad_df'] = quad_df
                            st.session_state['sp_quad_anchors'] = {
                                'x_left': x_left.strip(), 'x_right': x_right.strip(),
                                'y_bottom': y_bottom.strip(), 'y_top': y_top.strip()
                            }
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error generating quadrant map: {e}")

            if 'sp_quad_df' in st.session_state:
                quad_df = st.session_state['sp_quad_df']
                anchors = st.session_state['sp_quad_anchors']

                # Assign types for color
                quad_types = []
                for name in quad_df['Brand']:
                    if name == "Saxo Bank":
                        quad_types.append("Saxo Bank")
                    else:
                        quad_types.append("Competitor")
                quad_df['Type'] = quad_types

                fig_quad = px.scatter(
                    quad_df,
                    x="X", y="Y",
                    color="Type",
                    symbol="Type",
                    text="Brand",
                    color_discrete_map={"Competitor": "#3498db", "Saxo Bank": "#f1c40f"},
                    symbol_map={"Competitor": "circle", "Saxo Bank": "star"},
                    hover_data={"Brand": True, "Type": True, "X": ":.4f", "Y": ":.4f"}
                )
                fig_quad.update_traces(textposition="top center", marker=dict(size=12))
                for trace in fig_quad.data:
                    if trace.name == "Saxo Bank":
                        trace.marker.size = 18

                # Force axes to cross at (0,0)
                x_max = max(abs(quad_df['X'].min()), abs(quad_df['X'].max())) * 1.3
                y_max = max(abs(quad_df['Y'].min()), abs(quad_df['Y'].max())) * 1.3

                fig_quad.update_layout(
                    height=700,
                    xaxis=dict(
                        zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.3)',
                        range=[-x_max, x_max],
                        title=None
                    ),
                    yaxis=dict(
                        zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.3)',
                        range=[-y_max, y_max],
                        title=None
                    ),
                    legend_title="Type",
                    margin=dict(l=10, r=10, t=30, b=10),
                    # Axis label annotations at the edges
                    annotations=[
                        dict(x=-x_max * 0.95, y=0, text=f"← {anchors['x_left']}",
                             showarrow=False, font=dict(size=13, color="#e74c3c"), xanchor="left"),
                        dict(x=x_max * 0.95, y=0, text=f"{anchors['x_right']} →",
                             showarrow=False, font=dict(size=13, color="#e74c3c"), xanchor="right"),
                        dict(x=0, y=-y_max * 0.95, text=f"↓ {anchors['y_bottom']}",
                             showarrow=False, font=dict(size=13, color="#2ecc71"), yanchor="bottom"),
                        dict(x=0, y=y_max * 0.95, text=f"{anchors['y_top']} ↑",
                             showarrow=False, font=dict(size=13, color="#2ecc71"), yanchor="top"),
                    ]
                )
                st.plotly_chart(fig_quad, use_container_width=True)

            st.markdown("---")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SECTION 3: HEATMAP + DATA TABLE + CSV
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.markdown("### 🔥 Similarity Heatmap")

            fig_heatmap = px.imshow(
                sim_df.values,
                labels=dict(x="Feature", y="Brand", color="Cosine Similarity"),
                x=list(sim_df.columns),
                y=list(sim_df.index),
                color_continuous_scale="Viridis",
                aspect="auto",
                text_auto=".3f"
            )
            fig_heatmap.update_layout(
                height=max(400, len(brand_names) * 35),
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # --- Data Table & CSV Download ---
            st.markdown("### 📊 Similarity Data")

            display_df = sim_df.reset_index()
            display_df = display_df.rename(columns={"index": "Brand"})

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            sp_csv_col1, sp_csv_col2 = st.columns([2, 1])
            with sp_csv_col1:
                sp_csv_format = st.selectbox(
                    "CSV Export Format",
                    ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"],
                    key="sp_csv_format"
                )
            with sp_csv_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

                if "EU Excel" in sp_csv_format:
                    sp_csv_df = display_df.copy()
                    for col in sp_csv_df.select_dtypes(include=['float64', 'float32']).columns:
                        sp_csv_df[col] = sp_csv_df[col].apply(lambda x: str(x).replace('.', ','))
                    sp_csv_bytes = sp_csv_df.to_csv(index=False, sep=';').encode('utf-8')
                else:
                    sp_csv_bytes = display_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="⬇️ Download CSV",
                    data=sp_csv_bytes,
                    file_name="Semantic_Positioning_Map.csv",
                    mime="text/csv",
                    key="sp_download_csv"
                )

# --- PAGE: REDDIT ANALYSIS ---
elif current_page == "Reddit Analysis":
    tab8 = st.tabs(["📢 Reddit Intel"])[0]
    with tab8:
            st.markdown("#### 📢 Reddit Intelligence")
            st.info("Analyze the sentiment of Reddit threads towards Saxo Bank using AI. Fetches thread content and top comments.")

            # Target Brand
            target_brand = st.text_input("Brand to Analyze", value="Saxo Bank")

            # Input Method Toggle
            input_method = st.radio("Input Method", ["Search Reddit", "Paste URLs", "AccuLLM Sources"], index=0, horizontal=True, label_visibility="collapsed")

            urls_to_analyze = []

            # -------------------------------------------------------------------------
            # MODE A: PASTE URLS
            # -------------------------------------------------------------------------
            if input_method == "Paste URLs":
                reddit_urls = st.text_area("Paste Reddit Thread URLs (one per line)", height=150)

                # Dynamic Cost Estimator for Paste
                if reddit_urls:
                    lines = [l for l in reddit_urls.split('\n') if l.strip()]
                    count = len(lines)
                    cost = count * 0.015
                    st.caption(f"Estimated Analysis Cost: ~${cost:.3f} ({count} threads)")

                if st.button("Analyze Threads", type="primary"):
                    raw_urls = [url.strip() for url in reddit_urls.split('\n') if url.strip()]
                    urls_to_analyze = [{"url": u, "cited_pct": None} for u in raw_urls]

            # -------------------------------------------------------------------------
            # MODE B: SEARCH REDDIT
            # -------------------------------------------------------------------------
            elif input_method == "Search Reddit":
                with st.container(border=True):
                    cols_search = st.columns([3, 1, 1])
                    query = cols_search[0].text_input("Search Keywords", placeholder="e.g. Saxo Bank review", label_visibility="collapsed")
                    sort_by = cols_search[1].selectbox("Sort", ["relevance", "hot", "top", "new"], label_visibility="collapsed")
                    time_filter = cols_search[2].selectbox("Time", ["all", "year", "month", "week"], index=1, label_visibility="collapsed")

                    # Limit Slider
                    limit_val = st.slider("Max Results", 1, 20, 5)

                    # Dynamic Cost Estimator
                    est_cost = limit_val * 0.015
                    st.info(f"💰 Estimated Analysis Cost: ~${est_cost:.2f} (based on GPT-4o)", icon="ℹ️")

                    if st.button("🔎 Find & Analyze", type="primary"):
                         # Init Client First to Search
                        try:
                            reddit_client = backend.get_reddit_client(st.secrets)
                        except FileNotFoundError:
                             reddit_client = None

                        if not reddit_client:
                            st.error("Missing Reddit API credentials.")
                        elif not query:
                            st.warning("Please enter search keywords.")
                        else:
                            with st.spinner("Searching Reddit..."):
                                found_urls = backend.search_reddit(
                                    query, 
                                    sort_by=sort_by, 
                                    time_filter=time_filter, 
                                    limit=limit_val, 
                                    reddit_client=reddit_client
                                )

                                if not found_urls:
                                    st.warning("No threads found matching criteria.")
                                else:
                                    urls_to_analyze = [{"url": u, "cited_pct": None} for u in found_urls]
                                    st.success(f"Found {len(found_urls)} threads. Starting analysis...")
                                    time.sleep(1)

            # Load & Render
            if not st.session_state.reddit_data:
                 if input_method == "AccuLLM Sources":
                     st.info("👈 Select a brand / market to start.")
                 else:
                     st.info("👈 Enter a URL or Search Query to start.")
            else:
                 # Ensure result is a list
                 data_to_show = st.session_state.reddit_data
                 if isinstance(data_to_show, dict): 
                     data_to_show = [data_to_show] # Wrap single result

                 for item in data_to_show:
                     with st.container(border=True):
                         col_up, col_down = st.columns([1, 4])
                         with col_up:
                             st.markdown(f"**r/{item.get('subreddit','')}**")
                             st.caption(f"Score: {item.get('score','N/A')}")
                         with col_down:
                             st.markdown(f"[{item.get('title')}]({item.get('url')})")
                             
                             if 'analysis' in item:
                                 analysis = item['analysis']
                                 sentiment = analysis.get('sentiment', 'Unknown')
                                 color = "grey"
                                 if "Positive" in sentiment: color = "green"
                                 elif "Negative" in sentiment: color = "red"
                                 
                                 st.markdown(f"**Sentiment:** :{color}[{sentiment}]") 
                                 st.write(analysis.get('summary', ''))



            # -------------------------------------------------------------------------
            # MODE C: COMMERCIAL GEO (ACCURANKER)
            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------
            # MODE C: COMMERCIAL GEO (ACCURANKER)
            # -------------------------------------------------------------------------
            if input_method == "AccuLLM Sources":
                ACCURANKER_BRANDS = {
                    "GEO Experiments (10000419)": 10000419,
                    "Saxo BE (10000083)": 10000083,
                    "Saxo CH (10000084)": 10000084,
                    "Saxo CZ (10000085)": 10000085,
                    "Saxo DK (10000087)": 10000087,
                    "Saxo FR (10000090)": 10000090,
                    "Saxo Institutional (10000275)": 10000275,
                    "Saxo IT (10000092)": 10000092,
                    "Saxo JP (10000095)": 10000095,
                    "Saxo MENA (10000120)": 10000120,
                    "Saxo NL (10000097)": 10000097,
                    "Saxo PL (10000117)": 10000117,
                    "Saxo SG (10000124)": 10000124,
                    "Saxo UK (10000079)": 10000079
                }

                st.markdown("##### 🌍 AccuLLM Sources")
                st.caption("Identify Reddit threads acting as 'AI Sources' in commercial queries.")

                selected_brand_key = st.selectbox("Select Market/Brand", list(ACCURANKER_BRANDS.keys()))
                selected_brand_id = ACCURANKER_BRANDS[selected_brand_key]

                # Init Session State for AccuRanker Data if not present
                if 'accuranker_data' not in st.session_state:
                    st.session_state['accuranker_data'] = None

                # Check for API Token
                try:
                    accuranker_token = st.secrets["ACCURANKER_TOKEN"]
                except KeyError:
                    st.error("Missing ACCURANKER_TOKEN in .streamlit/secrets.toml")
                    accuranker_token = None

                if accuranker_token:
                    # 1. Fetch Button
                    if st.button("Fetch AI Sources"):
                        with st.spinner("Fetching data from AccuRanker API..."):
                            # Fetch RAW prompts
                            raw_prompts = backend.fetch_accuranker_prompts_raw(selected_brand_id, accuranker_token)

                            if not raw_prompts:
                                st.warning("No data found for this brand (or API error).")
                                st.session_state['accuranker_prompts_raw'] = None
                                st.session_state['accuranker_data'] = None
                            else:
                                st.session_state['accuranker_prompts_raw'] = raw_prompts
                                st.success(f"Fetched {len(raw_prompts)} prompts.")

                    # 2. Tag Selection & Processing
                    if st.session_state.get('accuranker_prompts_raw'):
                        raw_pro = st.session_state['accuranker_prompts_raw']

                        # Extract unique tags
                        all_tags = set()
                        for p in raw_pro:
                            if p.get('tags'):
                                for t in p['tags']:
                                    all_tags.add(t)

                        sorted_tags = sorted(list(all_tags))

                        # Determine default index (Commercial)
                        # We want "Commercial" to be default if exists, case-insensitive match
                        default_idx = 0
                        for i, t in enumerate(sorted_tags):
                            if t.lower() == "commercial":
                                default_idx = i
                                break

                        selected_tag = st.selectbox("Select Tag", sorted_tags, index=default_idx)

                        # Process Data Based on Tag
                        # We only re-process if tag changes or data is missing
                        # But for simplicity, we can process on every rerun if fast enough (1000 items is fast)
                        # Or we can store 'last_processed_tag' to optimize.

                        # Let's just process it.
                        sources, relevant_prompts = backend.process_accuranker_prompts_for_reddit(raw_pro, tag_filter=selected_tag)

                        # Convert to DataFrame
                        if not sources:
                             st.warning(f"No Reddit threads found in prompts with tag '{selected_tag}'.")
                             st.session_state['accuranker_data'] = None
                        else:
                            df_sources = pd.DataFrame(sources)
                            df_sources.insert(0, "Select", True)
                            df_sources.rename(columns={"url": "URL", "count": "Prompts", "percentage": "Cited %", "title": "Title"}, inplace=True)

                            cols = ["Select", "Title", "URL", "Prompts", "Cited %"]
                            cols = [c for c in cols if c in df_sources.columns]
                            df_sources = df_sources[cols]

                            st.session_state['accuranker_data'] = df_sources

                        # 3. Prompts Viewer (New Feature)
                        with st.expander(f"View Prompts in Tag: {selected_tag} ({len(relevant_prompts)})", expanded=False):
                            if relevant_prompts:
                                # Create a simple dataframe for display
                                prompt_data = []
                                for p in relevant_prompts:
                                    # Get latest rank/url specific to the source? 
                                    # Just showing the keyword is probably main intent, but prompts usually have 'keyword' field?
                                    # Wait, the API response structure involves 'keyword_id' or similar? 
                                    # Actually AccuRanker prompts usually match a Keyword. 
                                    # Looking at previous debug prints, 'keyword' wasn't top level? 
                                    # Checking `debug_accuranker_fetch.txt` ...
                                    # Structure: {"id": 123, "tags": [...], "results": [...]}
                                    # It seems the 'keyword' text might be missing from the 'fields' I requested!
                                    # I requested: id,tags,results...
                                    # I should probably add 'keyword' to the fields in the backend to make this useful.
                                    # For now, I'll list the ID and Tags, but I should probably fix the backend to fetch 'keyword' text.
                                    # Let's assume 'keyword' field exists if requested.
                                    prompt_data.append({
                                        "Prompt": p.get('prompt', 'N/A'),
                                        "ID": p.get('id'),
                                        "Tags": ", ".join(p.get('tags', []))
                                    })
                                st.dataframe(prompt_data, width="stretch")
                            else:
                                st.write("No ID details available.")

                # Display Results if available
                if st.session_state['accuranker_data'] is not None:
                     # Create a container for the button ABOVE the table
                     toggles_container = st.container()

                     # Initialize dynamic key for data editor reset
                     if 'editor_key' not in st.session_state:
                         st.session_state['editor_key'] = 0

                     edited_df = st.data_editor(
                         st.session_state['accuranker_data'],
                         column_config={
                             "Select": st.column_config.CheckboxColumn("Analyze?", width="small"),
                             "Title": st.column_config.TextColumn("Title", width="medium"),
                             "Prompts": st.column_config.NumberColumn("Prompts", help="Number of prompts where this thread appeared"),
                             "Cited %": st.column_config.NumberColumn("Cited %", format="%.1f%%", help="Percentage of commercial prompts citing this thread"),
                             "URL": st.column_config.LinkColumn("Thread URL", width="large")
                         },
                         width="stretch",
                         hide_index=True,
                         key=f"accuranker_table_{st.session_state['editor_key']}"
                     )
                     
                     # Sync edits back to session state to preserve manual changes
                     # This ensures that if we do something else, we don't lose manual checks
                     st.session_state['accuranker_data'] = edited_df

                     # Calculate statistics for the button dynamically
                     selected_rows = edited_df[edited_df["Select"] == True]
                     count_selected = len(selected_rows)
                     total_rows = len(edited_df)

                     # Render button into the container above
                     with toggles_container:
                         col1, col2, col3 = st.columns([2, 1, 1])

                         with col1:
                             if count_selected > 0:
                                 btn_text = f"Analyze {count_selected} out of {total_rows} Reddit threads"
                                 if st.button(btn_text, type="primary"):
                                     # Create list of dicts with URL and Cited %
                                     subset = selected_rows[["URL", "Cited %"]].copy()
                                     subset.rename(columns={"URL": "url", "Cited %": "cited_pct"}, inplace=True)
                                     urls_to_analyze = subset.to_dict('records')
                             else:
                                 if st.button(f"Analyze 0 out of {total_rows} Reddit threads", type="primary"):
                                     st.warning("Please tick off at least one thread to analyze.")

                         with col2:
                             if st.button("Select All", width="stretch"):
                                 st.session_state['accuranker_data'].loc[:, 'Select'] = True
                                 st.session_state['editor_key'] += 1
                                 st.rerun()

                         with col3:
                             if st.button("Deselect All", width="stretch"):
                                 st.session_state['accuranker_data'].loc[:, 'Select'] = False
                                 st.session_state['editor_key'] += 1
                                 st.rerun()

                     with st.expander("Debug: Raw API Response", expanded=False):
                         st.write("Debug view of the underlying data structure.")
                         st.json(st.session_state['accuranker_data'].to_dict(orient='records'))

            # -------------------------------------------------------------------------
            # MAIN EXECUTION LOOP
            # -------------------------------------------------------------------------
            if urls_to_analyze:
                urls = urls_to_analyze # normalized name

                if not urls:
                     st.warning("No URLs to process.")
                else:
                    # Init Client (Duplicate check but safe)
                    try:
                        reddit_client = backend.get_reddit_client(st.secrets)
                    except FileNotFoundError:
                         reddit_client = None

                    if not reddit_client:
                        st.error("Missing Reddit API credentials in .streamlit/secrets.toml.")
                    else:
                        progress_bar = st.progress(0, text="Initializing...")
                        results = []

                        total = len(urls_to_analyze)
                        for idx, item in enumerate(urls_to_analyze):
                            url = item['url']
                            cited_pct = item.get('cited_pct')
                            
                            progress_bar.progress((idx) / total, text=f"Fetching thread {idx+1}/{total}...")

                            # 1. Fetch
                            data = backend.fetch_reddit_data(url, reddit_client)

                            if data["error"]:
                                results.append({
                                    "Sentiment": "Error",
                                    "Title": "Error fetching data",
                                    "Summary": data["error"],
                                    "Link": url,
                                    "Upvotes": 0,
                                    "Comments": 0,
                                    "Cited %": 0
                                })
                            else:
                                # 2. Analyze
                                progress_bar.progress((idx + 0.5) / total, text=f"Analyzing sentiment for thread {idx+1}...")
                                analysis = backend.analyze_reddit_sentiment(data["context"], client, target_brand=target_brand)

                                results.append({
                                    "Sentiment": analysis.get("sentiment", "Unknown"),
                                    "Subreddit": data.get("subreddit", "Unknown"),
                                    "Title": data["title"],
                                    "Summary": analysis.get("summary", "No summary generated."),
                                    "Link": url,
                                    "Upvotes": data.get("score", 0),
                                    "Comments": data.get("num_comments", 0),
                                    "Cited %": cited_pct if cited_pct is not None else 0
                                })

                        progress_bar.progress(1.0, text="Analysis Complete!")
                        st.session_state['reddit_results'] = pd.DataFrame(results)

            # Display Results
            if 'reddit_results' in st.session_state:
                st.divider()
                
                # --- Sorting Controls ---
                col_sort1, col_sort2 = st.columns([1, 3])
                
                with col_sort1:
                    # Determine Default Sort
                    default_sort_idx = 1 # Comments
                    if input_method == "AccuLLM Sources":
                        default_sort_idx = 2 # Cited %

                    sort_option = st.selectbox(
                        "Sort Results By", 
                        ["Upvotes", "Comments", "Cited %", "Sentiment"],
                        index=default_sort_idx
                    )
                
                with col_sort2:
                    view_mode = st.radio("Display Mode", ["Feed View", "Table View"], horizontal=True)

                df_results = st.session_state['reddit_results']
                
                # Apply Sorting
                if sort_option == "Upvotes":
                    df_results = df_results.sort_values(by="Upvotes", ascending=False)
                elif sort_option == "Comments":
                    df_results = df_results.sort_values(by="Comments", ascending=False)
                elif sort_option == "Cited %":
                    df_results = df_results.sort_values(by="Cited %", ascending=False)
                elif sort_option == "Sentiment":
                    df_results = df_results.sort_values(by="Sentiment")

                if view_mode == "Table View":
                    def color_sentiment(val):
                        color = ''
                        if val == 'Positive':
                            color = 'color: #2ecc71; font-weight: bold'
                        elif val == 'Negative':
                            color = 'color: #e74c3c; font-weight: bold'
                        elif val == 'Neutral':
                            color = 'color: #95a5a6'
                        elif val == 'Mixed':
                            color = 'color: #f39c12'
                        return color

                    st.dataframe(
                        df_results.style.map(color_sentiment, subset=['Sentiment']),
                        column_config={
                            "Subreddit": st.column_config.TextColumn("Subreddit", width="small"),
                            "Link": st.column_config.LinkColumn("Thread URL", display_text="Open Thread"),
                            "Summary": st.column_config.TextColumn("AI Summary", width="large"),
                            "Upvotes": st.column_config.NumberColumn("⬆️ Upvotes"),
                            "Comments": st.column_config.NumberColumn("💬 Comments"),
                            "Cited %": st.column_config.NumberColumn("🔗 Cited %", format="%.1f%%")
                        },
                        width="stretch",
                        hide_index=True
                    )

                else: # Feed View
                    st.markdown("---")
                    # Convert to list of dicts to avoid iterrows artifacts
                    results_list = df_results.to_dict('records')

                    for row in results_list:
                        sentiment = row['Sentiment']

                        # Sentiment Colors/Icons
                        header_color = "gray"
                        icon = "⚪"
                        if sentiment == 'Positive':
                            header_color = "#2ecc71"
                            icon = "🟢"
                        elif sentiment == 'Negative':
                            header_color = "#e74c3c"
                            icon = "🔴"
                        elif sentiment == 'Mixed':
                            header_color = "#f39c12"
                            icon = "🟠"

                        with st.container(border=True):
                            # Custom Header
                            st.markdown(f"##### {icon} <span style='color:{header_color}'>r/{row['Subreddit']}: {row['Title']}</span>", unsafe_allow_html=True)
                            
                            # Metadata Line
                            meta_parts = [f"**{sentiment}**"]
                            meta_parts.append(f"⬆️ {row.get('Upvotes', 0)}")
                            meta_parts.append(f"💬 {row.get('Comments', 0)}")
                            
                            if row.get('Cited %', 0) > 0:
                                meta_parts.append(f"🔗 Cited in {row['Cited %']:.1f}%")
                                
                            st.caption("  |  ".join(meta_parts))

                            # Full Body
                            st.markdown(row['Summary'])

                            # Link
                            st.link_button("Open on Reddit", row['Link'])

                # --- General Market Sentiment ---
                st.markdown("---")
                st.markdown("#### 🌎 General Market Sentiment")
                st.info("Synthesize the summarized Reddit threads above into a general market sentiment report using AI.")
                
                if st.button("Get Market Sentiment", type="primary", key="btn_market_sentiment"):
                    with st.spinner("Generating market sentiment report..."):
                        # Prepare data for LLM
                        threads_data = df_results.to_dict('records')
                        
                        # Use the appropriate client based on whether we have OpenAI client available
                        # In the current scope, 'client' might be defined from earlier analysis, 
                        # but to be safe, we init it if needed
                        gemini_key = st.secrets.get("GEMINI_API_KEY", None)
                        openai_key = st.secrets.get("OPENAI_API_KEY", None)
                        
                        if not openai_key:
                            st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml.")
                        else:
                            from openai import OpenAI
                            local_openai_client = OpenAI(api_key=openai_key)
                            
                            sentiment_report = backend.generate_general_market_sentiment(
                                analyzed_threads_data=threads_data,
                                client=local_openai_client,
                                model="gpt-4o"
                            )
                            
                            st.session_state['general_market_sentiment'] = sentiment_report
                
                if 'general_market_sentiment' in st.session_state:
                    with st.container(border=True):
                        st.markdown(st.session_state['general_market_sentiment'])

# --- PAGE: LLM MONITORING ---
elif current_page == "LLM Monitoring":
    # Brand Dictionary
    ACCURANKER_BRANDS = {
        "GEO Experiments": 10000419,
        "Saxo BE": 10000083,
        "Saxo CH": 10000084,
        "Saxo CZ": 10000085,
        "Saxo DK": 10000087,
        "Saxo FR": 10000090,
        "Saxo Institutional": 10000275,
        "Saxo IT": 10000092,
        "Saxo JP": 10000095,
        "Saxo MENA": 10000120,
        "Saxo NL": 10000097,
        "Saxo PL": 10000117,
        "Saxo SG": 10000124,
        "Saxo UK": 10000079
    }

    tab_kpi, tab_comp_overview, tab_cross_market, tab_truth, tab_extract, tab_source_trends, tab_scraper = st.tabs(["📈 KPI Monitoring", "🏎️ Competitive Overview", "🌍 Cross Market Analysis", "🛡️ LLM Truth Control", "⛏️ Source Extraction", "📈 Source Trends", "🕵️‍♂️ Competitor Scraper"])

    with tab_kpi:
        st.markdown("## 📈 KPI Monitoring")
        st.info("Monitor historical Visibility and Sentiment of AI Search Results.")
        
        with st.container(border=True):
            col_brand, col_comp, col_tag, col_date = st.columns([1, 1, 1, 1])
            with col_brand:
                brand_list = list(ACCURANKER_BRANDS.keys())
                default_brand_ix = brand_list.index("Saxo DK") if "Saxo DK" in brand_list else 0
                kpi_brand_name = st.selectbox("Select Brand", brand_list, index=default_brand_ix, key="kpi_brand")
                kpi_brand_id = ACCURANKER_BRANDS[kpi_brand_name]
                
            with col_comp:
                comp_options = ["None"] + list(ACCURANKER_BRANDS.keys())
                kpi_comp_name = st.selectbox("Compare With (Optional)", comp_options, key="kpi_comp_brand")
                kpi_comp_id = ACCURANKER_BRANDS.get(kpi_comp_name)
                
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                kpi_cache_key = f"tags_{kpi_brand_id}"
                
                if kpi_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              st.session_state[kpi_cache_key] = backend.fetch_unique_tags(kpi_brand_id, accuranker_token)
                          else:
                              st.session_state[kpi_cache_key] = {}
                
                tags_map = st.session_state.get(kpi_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                    
                kpi_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix, key="kpi_tag") if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial", key="kpi_tag_fallback")
                kpi_tag_clean = kpi_tag_str.split(" (")[0] if " (" in str(kpi_tag_str) else kpi_tag_str
                
            with col_date:
                from datetime import datetime, timedelta
                six_months_ago = datetime.today() - timedelta(days=180)
                date_range = st.date_input("Date Range", value=(six_months_ago, datetime.today()), max_value=datetime.today(), key="kpi_date")
                
        col_filters, col_vis, col_sen, col_btn = st.columns([1.5, 1, 1, 1])
        with col_filters:
            rolling_avg_options = ["None", "Weekly (7-day)", "Monthly (30-day)"]
            rolling_avg = st.selectbox("Rolling Average", rolling_avg_options, index=1, key="kpi_rolling")
            
        with col_vis:
            st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
            show_visibility = st.checkbox("Visibility", value=True, key="kpi_show_vis")
            
        with col_sen:
            st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
            show_sentiment = st.checkbox("Sentiment", value=True, key="kpi_show_sen")
            
        with col_btn:
            st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
            show_trendlines = st.checkbox("Show Trendlines", value=False, key="kpi_show_trend")
        
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            fetch_kpi_btn = st.button("Fetch KPI Data", type="primary", width="stretch")
            
        if fetch_kpi_btn:
             if len(date_range) != 2:
                 st.error("Please select both a start and end date.")
             elif not accuranker_token:
                 st.error("Missing ACCURANKER_TOKEN in secrets.")
             elif kpi_comp_id and kpi_brand_id == kpi_comp_id:
                 st.warning("Cannot compare a brand to itself. Please select a different comparison brand or set it to 'None'.")
             else:
                 start_date, end_date = date_range
                 with st.spinner(f"Fetching KPI data from {start_date} to {end_date}..."):
                     # Fetch primary brand
                     df_kpi_primary, df_prompts = backend.fetch_kpi_time_series(
                         kpi_brand_id,
                         kpi_tag_clean,
                         start_date,
                         end_date,
                         accuranker_token
                     )
                     
                     df_final = pd.DataFrame()
                     
                     if not df_kpi_primary.empty:
                         # Rename columns for clarity
                         df_kpi_primary = df_kpi_primary.rename(columns={
                             'Visibility': f'Visibility ({kpi_brand_name})', 
                             'Sentiment': f'Sentiment ({kpi_brand_name})'
                         })
                         df_final = df_kpi_primary
                         
                         # Fetch comparison brand if selected
                         if kpi_comp_id:
                             df_kpi_comp, _ = backend.fetch_kpi_time_series(
                                 kpi_comp_id,
                                 kpi_tag_clean,
                                 start_date,
                                 end_date,
                                 accuranker_token
                             )
                             if not df_kpi_comp.empty:
                                 df_kpi_comp = df_kpi_comp.rename(columns={
                                     'Visibility': f'Visibility ({kpi_comp_name})', 
                                     'Sentiment': f'Sentiment ({kpi_comp_name})'
                                 })
                                 # Merge the two DataFrames on Date
                                 df_final = pd.merge(df_kpi_primary, df_kpi_comp, on='Date', how='outer')
                                 df_final = df_final.sort_values('Date')
                     
                     if df_final.empty:
                         st.warning("No KPI data found for the selected criteria.")
                         st.session_state.kpi_series_df = None
                         st.session_state.kpi_chart_meta = None
                         st.session_state.kpi_prompts_df = None
                     else:
                         st.session_state.kpi_series_df = df_final
                         st.session_state.kpi_chart_meta = {
                             'primary_vis': f'Visibility ({kpi_brand_name})',
                             'primary_sen': f'Sentiment ({kpi_brand_name})',
                             'comp_vis': f'Visibility ({kpi_comp_name})' if kpi_comp_id else None,
                             'comp_sen': f'Sentiment ({kpi_comp_name})' if kpi_comp_id else None,
                         }
                         st.session_state.kpi_prompts_df = df_prompts
                         st.success("Fetched KPI Data.")

        if 'kpi_series_df' in st.session_state and st.session_state.kpi_series_df is not None:
            df_plot = st.session_state.kpi_series_df.copy()
            meta = st.session_state.kpi_chart_meta
            
            # Detect Competitors in DataFrame
            # Look for columns starting with "Visibility - " (excluding comparison brand)
            competitor_names = set()
            for col in df_plot.columns:
                 if col.startswith("Visibility - ") and col != meta.get('comp_vis'):
                      comp_name = col.replace("Visibility - ", "")
                      competitor_names.add(comp_name)
                      
            competitor_names = sorted(list(competitor_names))[:5] # Max 5
            
            # Render Competitor Toggles
            active_competitors = []
            if competitor_names:
                st.markdown("##### Pinned Competitors")
                comp_cols = st.columns(len(competitor_names) + (5 - len(competitor_names))) # Keep layout intact up to 5
                for i, comp_name in enumerate(competitor_names):
                     with comp_cols[i]:
                          # State is preserved automatically by Streamlit key if we don't override default
                          if st.checkbox(comp_name, value=False, key=f"kpi_comp_toggle_{comp_name}"):
                               active_competitors.append(comp_name)
                               
            # Optional: handle if comparison brand conflicts with a pinned comp
            
            val_columns = [c for c in df_plot.columns if c != 'Date']
            
            period = 1
            # Apply rolling average if selected
            if rolling_avg == "Weekly (7-day)":
                period = 7
                for c in val_columns:
                    df_plot[c] = df_plot[c].rolling(window=7, min_periods=1).mean()
            elif rolling_avg == "Monthly (30-day)":
                period = 30
                for c in val_columns:
                    df_plot[c] = df_plot[c].rolling(window=30, min_periods=1).mean()
                
            st.markdown("### Visibility & Sentiment Trends")
            
            # --- Score Cards ---
            orig_df = st.session_state.kpi_series_df
            p_vis_col = meta['primary_vis']
            p_sen_col = meta['primary_sen']
            
            if len(orig_df) > 0:
                calc_period = period if len(orig_df) >= period * 2 else max(1, len(orig_df) // 2)
                if calc_period == 0: calc_period = len(orig_df)
                
                base_vis = orig_df[p_vis_col].iloc[:calc_period].mean()
                base_sen = orig_df[p_sen_col].iloc[:calc_period].mean()
                
                curr_vis = orig_df[p_vis_col].iloc[-calc_period:].mean()
                curr_sen = orig_df[p_sen_col].iloc[-calc_period:].mean()
                
                d_vis = curr_vis - base_vis
                d_sen = curr_sen - base_sen
                
                p_vis = (d_vis / base_vis * 100) if base_vis > 0 else 0.0
                p_sen = (d_sen / base_sen * 100) if base_sen > 0 else 0.0
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.metric(f"Avg {p_vis_col} (Last {calc_period} entries)", f"{curr_vis:.1f}", f"{d_vis:+.1f} ({p_vis:+.1f}%)")
                with sc2:
                    st.metric(f"Avg {p_sen_col} (Last {calc_period} entries)", f"{curr_sen:.2f}", f"{d_sen:+.2f} ({p_sen:+.1f}%)")
            
            # --- Custom HTML Legend ---
            legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; font-size: 14px; align-items: center;'>"
            
            if show_visibility:
                legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; height: 3px; background-color: #2ecc71; margin-right: 8px;'></span> {meta['primary_vis']}</div>"
                
            if show_sentiment:
                legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; height: 3px; background-color: #e74c3c; margin-right: 8px;'></span> {meta['primary_sen']}</div>"
                
            if kpi_comp_id and meta['comp_vis']:
                if show_visibility:
                    legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; border-top: 3px dashed #27ae60; margin-right: 8px;'></span> {meta['comp_vis']}</div>"
                    
                if show_sentiment:
                    legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; border-top: 3px dashed #c0392b; margin-right: 8px;'></span> {meta['comp_sen']}</div>"
                    
            # Competitor colors
            comp_colors = ["#f1c40f", "#9b59b6", "#34495e", "#e67e22", "#1abc9c"]
            
            for i, comp_name in enumerate(active_competitors):
                 color = comp_colors[i % len(comp_colors)]
                 if show_visibility:
                      legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; border-top: 2px solid {color}; margin-right: 8px;'></span> Visibility ({comp_name})</div>"
                 if show_sentiment:
                      # We can adjust style slightly for sentiment to differentiate from visibility, e.g. dotted
                      legend_html += f"<div style='display: flex; align-items: center;'><span style='display: inline-block; width: 25px; border-top: 2px dotted {color}; margin-right: 8px;'></span> Sentiment ({comp_name})</div>"
            
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # Altair Dual Axis Chart - Base Component
            base = alt.Chart(df_plot).encode(
                x=alt.X('Date:T', axis=alt.Axis(title='Date'))
            )
            
            layers = []
            
            # Helper function to append chart and trendline
            def add_line(field, title, color, is_dashed=False, is_dotted=False, axis_title=None):
                stroke_dash = []
                if is_dashed: stroke_dash = [5, 5]
                elif is_dotted: stroke_dash = [2, 2]
                
                # Check if we should render on the shared scale
                y_axis = alt.Y(f"{field}:Q", scale=alt.Scale(zero=True))
                if axis_title is not None:
                     y_axis = alt.Y(f"{field}:Q", scale=alt.Scale(zero=True), axis=alt.Axis(title=axis_title, titleColor=color))
                else:
                     y_axis = alt.Y(f"{field}:Q", scale=alt.Scale(zero=True), axis=alt.Axis(title=None))

                line = base.mark_line(color=color, strokeWidth=3 if not is_dashed and not is_dotted else 2, strokeDash=stroke_dash).encode(
                    y=y_axis,
                    tooltip=['Date', alt.Tooltip(f"{field}:Q", format='.2f')]
                )
                layers.append(line)
                
                # Add trendline if requested
                if show_trendlines:
                     trend = line.transform_regression('Date', field, method='linear').mark_line(
                         color=color, strokeDash=[5,5], opacity=0.5, strokeWidth=2
                     )
                     layers.append(trend)
            
            # Primary lines
            if show_visibility:
                add_line(meta['primary_vis'], 'Visibility', '#2ecc71', axis_title='Visibility (%)')
                
            if show_sentiment:
                add_line(meta['primary_sen'], 'Sentiment', '#e74c3c', axis_title='Sentiment Score')
            
            # Secondary (Comparison) lines if data exists
            if show_visibility and meta['comp_vis'] and meta['comp_vis'] in df_plot.columns:
                add_line(meta['comp_vis'], 'Comp Vis', '#27ae60', is_dashed=True)
            if show_sentiment and meta['comp_sen'] and meta['comp_sen'] in df_plot.columns:
                add_line(meta['comp_sen'], 'Comp Sen', '#c0392b', is_dashed=True)
                
            # Active Pinned Competitor lines
            for i, comp_name in enumerate(active_competitors):
                 color = comp_colors[i % len(comp_colors)]
                 
                 vis_col = f"Visibility - {comp_name}"
                 sen_col = f"Sentiment - {comp_name}"
                 
                 if show_visibility and vis_col in df_plot.columns:
                      add_line(vis_col, f"{comp_name} Vis", color)
                      
                 if show_sentiment and sen_col in df_plot.columns:
                      add_line(sen_col, f"{comp_name} Sen", color, is_dotted=True)
            
            if layers:
                chart = alt.layer(*layers).resolve_scale(
                    y='shared'
                ).properties(
                    height=400
                ).interactive()
                
                st.altair_chart(chart, width="stretch")
            else:
                st.warning("Please select at least one metric to visualize.")
            
            st.dataframe(df_plot, width="stretch", hide_index=True)
            
            # Display Prompts List
            if 'kpi_prompts_df' in st.session_state and st.session_state.kpi_prompts_df is not None and not st.session_state.kpi_prompts_df.empty:
                 num_prompts = len(st.session_state.kpi_prompts_df)
                 with st.expander(f"View {num_prompts} Matched Prompts", expanded=False):
                      st.dataframe(st.session_state.kpi_prompts_df, use_container_width=True, hide_index=True)
            
    with tab_comp_overview:
        st.markdown("## 🏎️ Competitive Overview")
        st.info("Analyze and plot your primary brand against its competitive landscape.")
        
        with st.container(border=True):
            col_brand, col_tag, col_date, col_chk = st.columns([1, 1, 1, 1])
            with col_brand:
                brand_list = list(ACCURANKER_BRANDS.keys())
                default_brand_ix = brand_list.index("Saxo DK") if "Saxo DK" in brand_list else 0
                comp_brand_name = st.selectbox("Select Brand", brand_list, index=default_brand_ix, key="comp_ov_brand")
                comp_brand_id = ACCURANKER_BRANDS[comp_brand_name]
                
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                comp_cache_key = f"tags_{comp_brand_id}"
                
                if comp_cache_key not in st.session_state:
                     if accuranker_token:
                          st.session_state[comp_cache_key] = backend.fetch_unique_tags(comp_brand_id, accuranker_token)
                     else:
                          st.session_state[comp_cache_key] = {}
                          
                tags_map = st.session_state.get(comp_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                        
                comp_tag_val = st.selectbox("Select Tag (Optional)", ["All"] + tag_options, index=default_ix + 1 if tag_options else 0, key="comp_ov_tag")
                comp_tag_clean = None if comp_tag_val == "All" else comp_tag_val.split(" (")[0] if " (" in str(comp_tag_val) else comp_tag_val
                
            with col_date:
                from datetime import datetime, timedelta
                comp_end = datetime.now()
                comp_start = comp_end - timedelta(days=180)
                comp_date_range = st.date_input("Date Range", value=(comp_start, comp_end), key="comp_ov_date")
            
            with col_chk:
                st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
                comp_latest = st.checkbox("Latest Snapshot Data", value=False, key="comp_ov_latest")
                show_vector = st.checkbox("Show Competitive Vector", value=False, key="comp_ov_vector")
                icon_size_val = st.radio("Icon Size", ["Small", "Medium", "Large"], index=1, horizontal=True, key="comp_ov_icon_size")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            fetch_comp_btn = st.button("Generate Overview", type="primary", width="stretch", key="comp_ov_btn")
            
        if fetch_comp_btn:
            if len(comp_date_range) != 2:
                st.error("Please select both a start and end date.")
            elif not accuranker_token:
                st.error("Missing ACCURANKER_TOKEN in secrets.")
            else:
                s_date, e_date = comp_date_range
                with st.spinner(f"Fetching Competitive Overview data..."):
                    df_comp_raw = backend.fetch_competitive_overview(
                        comp_brand_id,
                        comp_brand_name,
                        comp_tag_clean,
                        s_date,
                        e_date,
                        accuranker_token
                    )
                    
                    if df_comp_raw.empty:
                        st.warning("No data found for the selected criteria.")
                        st.session_state.comp_ov_df = None
                    else:
                        st.session_state.comp_ov_df = df_comp_raw
                        st.session_state.comp_ov_latest_val = comp_latest
                        
                        # Build success message with vector period info
                        msg = "Overview Generated."
                        if show_vector:
                            all_dates_sorted = df_comp_raw['Date'].sort_values()
                            min_d = all_dates_sorted.iloc[0]
                            max_d = all_dates_sorted.iloc[-1]
                            first_end = (min_d + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                            last_start = (max_d - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                            msg += f" Vector compares **{min_d.strftime('%Y-%m-%d')} → {first_end}** vs **{last_start} → {max_d.strftime('%Y-%m-%d')}**."
                        st.success(msg)
                        
        if 'comp_ov_df' in st.session_state and st.session_state.comp_ov_df is not None:
            df_plot = st.session_state.comp_ov_df.copy()
            is_latest = st.session_state.get('comp_ov_latest_val', False)
            
            if is_latest:
                idx = df_plot.groupby(['Competitor', 'Domain'])['Date'].idxmax()
                df_agg = df_plot.loc[idx]
            else:
                df_agg = df_plot.groupby(['Competitor', 'Domain'], as_index=False)[['Visibility', 'Sentiment']].mean()
                
            df_agg['Visibility'] = df_agg['Visibility'].round(1)
            df_agg['Sentiment'] = df_agg['Sentiment'].round(1)

            # --- Top N filter ---
            total_competitors = len(df_agg)
            if total_competitors > 3:
                top_n = st.slider("Show Top N Competitors (by Visibility)", min_value=3, max_value=total_competitors, value=total_competitors, key="comp_ov_topn")
                # Always keep own brand, then fill with top N by visibility
                own_brand_rows = df_agg[df_agg['Competitor'] == comp_brand_name]
                other_rows = df_agg[df_agg['Competitor'] != comp_brand_name].nlargest(top_n, 'Visibility')
                df_agg = pd.concat([own_brand_rows, other_rows]).drop_duplicates(subset=['Competitor', 'Domain'])
                # Also filter df_plot so vector logic only uses visible competitors
                visible_competitors = set(df_agg['Competitor'].unique())
                df_plot = df_plot[df_plot['Competitor'].isin(visible_competitors)]

            st.markdown("### Visibility vs Sentiment")
            
            # Box coords (x >= 75, y >= 62) to max 100
            box_data = pd.DataFrame([{
                'x_start': 75, 'x_end': 100,
                'y_start': 62, 'y_end': 100
            }])
            
            quad_box = alt.Chart(box_data).mark_rect(
                color='#2ecc71', opacity=0.15
            ).encode(
                x=alt.X('x_start:Q', scale=alt.Scale(domain=[0, 100])),
                x2='x_end:Q',
                y=alt.Y('y_start:Q', scale=alt.Scale(domain=[0, 100])),
                y2='y_end:Q'
            )
            
            import base64
            import os
            def get_favicon_b64(domain):
                if pd.isna(domain) or not domain:
                    return None
                path = os.path.join("assets", "favicons", f"{domain}.png")
                if os.path.exists(path):
                    with open(path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()
                        return f"data:image/png;base64,{encoded_string}"
                return None
                
            df_agg['FaviconBase64'] = df_agg['Domain'].apply(get_favicon_b64)
            
            # Determine icon sizing based on user selection
            size_map = {"Small": 24, "Medium": 48, "Large": 72}
            circle_area_map = {"Small": 100, "Medium": 400, "Large": 900}
            icon_w = size_map.get(st.session_state.get('comp_ov_icon_size', "Medium"), 48)
            circle_s = circle_area_map.get(st.session_state.get('comp_ov_icon_size', "Medium"), 400)
            
            scatter_layers = [quad_box]
            
            # --- Competitive Vector Logic ---
            if show_vector and not df_plot.empty and 'Date' in df_plot.columns:
                all_dates = df_plot['Date'].sort_values().unique()
                if len(all_dates) >= 2:
                    # First 7 days and last 7 days
                    min_date = pd.Timestamp(all_dates[0])
                    max_date = pd.Timestamp(all_dates[-1])
                    first_cutoff = min_date + pd.Timedelta(days=7)
                    last_cutoff = max_date - pd.Timedelta(days=7)
                    
                    df_first = df_plot[df_plot['Date'] <= first_cutoff].groupby(['Competitor', 'Domain'], as_index=False)[['Visibility', 'Sentiment']].mean()
                    df_last = df_plot[df_plot['Date'] >= last_cutoff].groupby(['Competitor', 'Domain'], as_index=False)[['Visibility', 'Sentiment']].mean()
                    
                    df_first = df_first.rename(columns={'Visibility': 'Vis_old', 'Sentiment': 'Sen_old'})
                    df_last = df_last.rename(columns={'Visibility': 'Vis_new', 'Sentiment': 'Sen_new'})
                    
                    df_vec = pd.merge(df_first, df_last, on=['Competitor', 'Domain'], how='inner')
                    
                    if not df_vec.empty:
                        df_vec['Vis_old'] = df_vec['Vis_old'].round(1)
                        df_vec['Sen_old'] = df_vec['Sen_old'].round(1)
                        df_vec['Vis_new'] = df_vec['Vis_new'].round(1)
                        df_vec['Sen_new'] = df_vec['Sen_new'].round(1)
                        
                        # Determine vector direction for color
                        def get_vector_color(row):
                            vis_improved = row['Vis_new'] > row['Vis_old']
                            sen_improved = row['Sen_new'] > row['Sen_old']
                            if vis_improved and sen_improved:
                                return '#2ecc71'  # Green - both improved
                            elif vis_improved or sen_improved:
                                return '#f1c40f'  # Yellow - one improved
                            else:
                                return '#e74c3c'  # Red - both worsened
                        
                        df_vec['VectorColor'] = df_vec.apply(get_vector_color, axis=1)
                        
                        # Draw connecting lines (rules)
                        lines = alt.Chart(df_vec).mark_rule(strokeWidth=2, opacity=0.7).encode(
                            x=alt.X('Vis_old:Q'),
                            y=alt.Y('Sen_old:Q'),
                            x2='Vis_new:Q',
                            y2='Sen_new:Q',
                            color=alt.Color('VectorColor:N', scale=None),
                            tooltip=['Competitor', 'Vis_old', 'Sen_old', 'Vis_new', 'Sen_new']
                        )
                        scatter_layers.append(lines)
                        
                        # Faded old position icons
                        df_vec['FaviconBase64'] = df_vec['Domain'].apply(get_favicon_b64)
                        df_vec_img = df_vec[df_vec['FaviconBase64'].notnull()]
                        df_vec_no_img = df_vec[df_vec['FaviconBase64'].isnull()]
                        
                        if not df_vec_no_img.empty:
                            old_circles = alt.Chart(df_vec_no_img).mark_circle(size=circle_s * 0.8, opacity=0.25).encode(
                                x=alt.X('Vis_old:Q'),
                                y=alt.Y('Sen_old:Q'),
                                color=alt.Color('Competitor:N', legend=None),
                                tooltip=['Competitor', alt.Tooltip('Vis_old:Q', title='Old Visibility'), alt.Tooltip('Sen_old:Q', title='Old Sentiment')]
                            )
                            scatter_layers.append(old_circles)
                        
                        if not df_vec_img.empty:
                            old_images = alt.Chart(df_vec_img).mark_image(
                                width=icon_w, height=icon_w, opacity=0.25
                            ).encode(
                                x=alt.X('Vis_old:Q'),
                                y=alt.Y('Sen_old:Q'),
                                url='FaviconBase64:N',
                                tooltip=['Competitor', alt.Tooltip('Vis_old:Q', title='Old Visibility'), alt.Tooltip('Sen_old:Q', title='Old Sentiment')]
                            )
                            scatter_layers.append(old_images)
            
                        # Override df_agg so current icons match the line endpoints
                        df_vec_current = df_vec[['Competitor', 'Domain', 'Vis_new', 'Sen_new', 'Vis_old', 'Sen_old']].rename(
                            columns={'Vis_new': 'Visibility', 'Sen_new': 'Sentiment'}
                        )
                        df_vec_current['Δ Visibility'] = (df_vec_current['Visibility'] - df_vec_current['Vis_old']).round(1)
                        df_vec_current['Δ Sentiment'] = (df_vec_current['Sentiment'] - df_vec_current['Sen_old']).round(1)
                        df_vec_current = df_vec_current.drop(columns=['Vis_old', 'Sen_old'])
                        df_vec_current['FaviconBase64'] = df_vec_current['Domain'].apply(get_favicon_b64)
                        df_agg = df_vec_current
                        _vector_active = True
            
            # --- Current position marks (on top) ---
            df_with_img = df_agg[df_agg['FaviconBase64'].notnull()]
            df_no_img = df_agg[df_agg['FaviconBase64'].isnull()]
            
            # Build tooltip list based on whether vector deltas are available
            _vector_active = 'Δ Visibility' in df_agg.columns
            if _vector_active:
                current_tooltip = ['Competitor', 'Domain', 'Visibility', 'Sentiment',
                                   alt.Tooltip('Δ Visibility:Q', format='+.1f'),
                                   alt.Tooltip('Δ Sentiment:Q', format='+.1f')]
            else:
                current_tooltip = ['Competitor', 'Domain', 'Visibility', 'Sentiment']
            
            if not df_no_img.empty:
                points = alt.Chart(df_no_img).mark_circle(size=circle_s, opacity=0.8).encode(
                    x=alt.X('Visibility:Q', title='Visibility (0-100)'),
                    y=alt.Y('Sentiment:Q', title='Sentiment (0-100)'),
                    color=alt.Color('Competitor:N', legend=None),
                    tooltip=current_tooltip
                )
                scatter_layers.append(points)
                
            if not df_with_img.empty:
                images = alt.Chart(df_with_img).mark_image(
                    width=icon_w, height=icon_w
                ).encode(
                    x=alt.X('Visibility:Q', title='Visibility (0-100)'),
                    y=alt.Y('Sentiment:Q', title='Sentiment (0-100)'),
                    url='FaviconBase64:N',
                    tooltip=current_tooltip
                )
                scatter_layers.append(images)
            
            chart = alt.layer(*scatter_layers).resolve_scale(
                x='shared', y='shared'
            ).properties(
                height=800
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # Warn about missing favicons
            missing_favs = df_agg[df_agg['FaviconBase64'].isnull()]
            if not missing_favs.empty:
                missing_names = missing_favs[['Competitor', 'Domain']].drop_duplicates()
                missing_list = ', '.join([f"{row['Competitor']} ({row['Domain']})" for _, row in missing_names.iterrows()])
                st.warning(f"⚠️ Missing favicons for: {missing_list}. These competitors are shown as colored dots instead of icons.")
            
            st.markdown("### Competitive Data")
            table_cols = ['Competitor', 'Domain', 'Visibility', 'Sentiment']
            if 'Δ Visibility' in df_agg.columns:
                table_cols += ['Δ Visibility', 'Δ Sentiment']
            df_table = df_agg[table_cols].copy()
            df_table = df_table.sort_values(by='Visibility', ascending=False)
            st.dataframe(df_table, use_container_width=True, hide_index=True, height=500)
            
            # --- AI Analysis ---
            st.markdown("#### 🤖 AI Strategic Analysis")
            ai_state_key = "comp_ov_ai_analysis"
            
            if ai_state_key in st.session_state:
                st.info(st.session_state[ai_state_key]["text"], icon="🧠")
                st.caption(f"Estimated Cost: ${st.session_state[ai_state_key]['cost']:.4f} (~{st.session_state[ai_state_key]['tokens']} input tokens)")
                if st.button("Clear AI Analysis", key="btn_clear_comp_ov_ai"):
                    del st.session_state[ai_state_key]
                    st.rerun()
            else:
                if st.button("🤖 Analyze with AI", type="primary", key="btn_comp_ov_ai"):
                    with st.spinner("Analyzing competitive landscape with GPT-4o..."):
                        try:
                            md_table = df_table.to_csv(index=False)
                            prompt_path = "prompts/competitive_overview_analysis.txt"
                            try:
                                with open(prompt_path, "r", encoding="utf-8") as f:
                                    sys_prompt = f.read()
                            except FileNotFoundError:
                                sys_prompt = "Analyze this competitive visibility and sentiment data and provide strategic GEO insights."
                            
                            # Build date context
                            date_context = f"Data period: {comp_date_range[0]} to {comp_date_range[1]}."
                            if show_vector and not df_plot.empty and 'Date' in df_plot.columns:
                                all_d = df_plot['Date'].sort_values()
                                min_d = all_d.iloc[0]
                                max_d = all_d.iloc[-1]
                                date_context += f"\nVector comparison: first 7 days ({min_d.strftime('%Y-%m-%d')} to {(min_d + pd.Timedelta(days=7)).strftime('%Y-%m-%d')}) vs last 7 days ({(max_d - pd.Timedelta(days=7)).strftime('%Y-%m-%d')} to {max_d.strftime('%Y-%m-%d')})."
                            
                            user_prompt = f"{date_context}\n\nHere is the competitive overview data:\n\n{md_table}"
                            in_tokens, est_cost = estimate_tokens_and_cost([sys_prompt, user_prompt])
                            
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=1000,
                                stream=True
                            )
                            
                            analysis_text = st.write_stream(response)
                            st.session_state[ai_state_key] = {"text": analysis_text, "cost": est_cost, "tokens": in_tokens}
                        except Exception as e:
                            st.error(f"AI Analysis failed: {e}")
            
    with tab_cross_market:
        st.markdown("## 🌍 Cross Market Analysis")
        st.info("Analyze aggregate KPI data across multiple brands broken down by LLM.")
        
        with st.container(border=True):
            st.markdown("### Select Brands")
            # Create a horizontal layout for checkboxes
            cols = st.columns(4)
            selected_brands_for_cross = []
            
            brand_items = list(ACCURANKER_BRANDS.keys())
            for i, brand_name in enumerate(brand_items):
                col = cols[i % 4]
                # Default true except for GEO Experiments and Saxo Institutional
                default_val = brand_name not in ["GEO Experiments", "Saxo Institutional"]
                with col:
                    if st.checkbox(brand_name, value=default_val, key=f"cross_market_{brand_name}"):
                        selected_brands_for_cross.append(brand_name)
                        
            st.markdown("---")
            
            col_tag, col_date, col_btn = st.columns([1, 1, 1])
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                default_brand_id = ACCURANKER_BRANDS.get("Saxo DK", 10000087)
                cross_cache_key = f"tags_{default_brand_id}"
                
                if cross_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              st.session_state[cross_cache_key] = backend.fetch_unique_tags(default_brand_id, accuranker_token)
                          else:
                              st.session_state[cross_cache_key] = {}
                
                tags_map = st.session_state.get(cross_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                    
                cross_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix, key="cross_tag") if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial", key="cross_tag_fallback")
                cross_tag_clean = cross_tag_str.split(" (")[0] if " (" in str(cross_tag_str) else cross_tag_str
                
            with col_date:
                from datetime import datetime, timedelta
                six_months_ago = datetime.today() - timedelta(days=180)
                cross_date_range = st.date_input("Date Range", value=(six_months_ago, datetime.today()), max_value=datetime.today(), key="cross_date")
                
            with col_btn:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                fetch_cross_btn = st.button("Fetch Market Data", type="primary", width="stretch")
                
        if fetch_cross_btn:
            if len(cross_date_range) != 2:
                st.error("Please select both a start and end date.")
            elif not accuranker_token:
                st.error("Missing ACCURANKER_TOKEN in secrets.")
            elif not selected_brands_for_cross:
                st.error("Please select at least one brand.")
            else:
                start_date, end_date = cross_date_range
                all_records = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, brand in enumerate(selected_brands_for_cross):
                    status_text.text(f"Fetching data for {brand}...")
                    b_id = ACCURANKER_BRANDS[brand]
                    
                    market_data = backend.fetch_cross_market_data(
                        b_id,
                        cross_tag_clean,
                        start_date,
                        end_date,
                        accuranker_token
                    )
                    
                    for engine, metrics in market_data.items():
                        all_records.append({
                            "Brand": brand,
                            "LLM Engine": engine,
                            "Visibility (%)": round(metrics['Visibility'], 2),
                            "Sentiment Score": round(metrics['Sentiment'], 2),
                            "Web Search Rate (%)": round(metrics['Web Search Rate'], 2)
                        })
                        
                    progress_bar.progress((idx + 1) / len(selected_brands_for_cross))
                    
                status_text.empty()
                progress_bar.empty()
                
                if not all_records:
                    st.warning("No data found for the selected criteria.")
                    st.session_state.cross_market_df = None
                else:
                    df_cross = pd.DataFrame(all_records)
                    st.session_state.cross_market_df = df_cross
                    st.success("Fetched Cross Market Data.")
                    
        if 'cross_market_df' in st.session_state and st.session_state.cross_market_df is not None:
            df_disp = st.session_state.cross_market_df.copy()
            st.markdown("### Cross Market Performance by LLM")
            st.dataframe(df_disp, width="stretch", hide_index=True)
            
            st.markdown("---")
            # CSV Download Options
            col_csv1, col_csv2 = st.columns([2, 1])
            with col_csv1:
                csv_format = st.selectbox(
                    "CSV Export Format", 
                    ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"], 
                    key="csv_format_cross"
                )
            with col_csv2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                
                if "EU Excel" in csv_format:
                    df_csv = df_disp.copy()
                    for col in df_csv.select_dtypes(include=['float64', 'float32']).columns:
                        df_csv[col] = df_csv[col].apply(lambda x: str(x).replace('.', ','))
                    csv_bytes = df_csv.to_csv(index=False, sep=';').encode('utf-8')
                else:
                    csv_bytes = df_disp.to_csv(index=False).encode('utf-8')
                    
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"Cross_Market_{cross_tag_clean.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="cross_market_download"
                )
            
    with tab_truth:
        st.markdown("## 🛡️ LLM Truth Control")
        st.info("Verify if AI search results verify the 'Ground Truth' defined in AccuRanker.")

        # Controls
        with st.container(border=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                selected_brand_name = st.selectbox("Select Brand", list(ACCURANKER_BRANDS.keys()))
                selected_brand_id = ACCURANKER_BRANDS[selected_brand_name]
            with c2:
                # Tag Logic: Fetch on change or if missing
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
            
                # Unique Key for this brand's tags in session state
                brand_cache_key = f"tags_{selected_brand_id}"
            
                if brand_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              tags_dict = backend.fetch_unique_tags(selected_brand_id, accuranker_token)
                              st.session_state[brand_cache_key] = tags_dict
                          else:
                              st.session_state[brand_cache_key] = {}
            
                # Tags Dict: {TagName: Count}
                tags_map = st.session_state.get(brand_cache_key, {})
                # List for Dropdown: "Tag (Count)"
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
            
                # Find default index
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                
                selected_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix) if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial")
            
                # Extract raw tag name
                target_tag = selected_tag_str.split(" (")[0] if " (" in str(selected_tag_str) else selected_tag_str
            
                # Cost Calculation
                if selected_tag_str and "(" in str(selected_tag_str):
                     try:
                         count = int(selected_tag_str.split("(")[1].replace(")", ""))
                         est_cost = count * 4 * 0.002 # ~$0.002 per prompt * 4 engines
                         st.caption(f"💰 Est. Cost: ~${est_cost:.3f}")
                     except:
                         pass




            # 1. FETCH STAGE
            st.write("")
            if st.button("Fetch Prompts", type="primary", width="content", help="Fetch latest prompts from AccuRanker"):
                if not accuranker_token:
                    st.error("Missing ACCURANKER_TOKEN in secrets.")
                else:
                     with st.spinner("Fetching data from AccuRanker..."):
                         tasks = backend.fetch_accuranker_data(
                             selected_brand_id, 
                             target_tag, 
                             accuranker_token
                         )
                         st.session_state.fetched_tasks = tasks
                     
                         # Initialize Selection DataFrame
                         selection_data = []
                         for i, t in enumerate(tasks):
                             has_truth = bool(t.get('truth') and t.get('truth').strip())
                             truth_display = "✅ Yes" if has_truth else "❌ No"
                             selection_data.append({
                                 "Select": True, 
                                 "Prompt": t.get('prompt'),
                                 "Engine": t.get('engine'), 
                                 "Truth Defined?": truth_display,
                                 "TaskID": i
                             })
                         st.session_state.selection_df = pd.DataFrame(selection_data)
                     
                         st.success(f"Fetched {len(tasks)} items.")

            # 2. SELECTION STAGE
            if 'fetched_tasks' in st.session_state and st.session_state.fetched_tasks:
                tasks = st.session_state.fetched_tasks
            
                st.markdown("### Select Prompts to Verify")
            
                # Helper Buttons
                c_sel1, c_sel2, c_spacer = st.columns([1, 1, 8])
                with c_sel1:
                    if st.button("Select All", width="stretch"):
                        st.session_state.selection_df["Select"] = True
                        st.session_state['verification_key'] = st.session_state.get('verification_key', 0) + 1
                        st.rerun()
                with c_sel2:
                    if st.button("Unselect All", width="stretch"):
                        st.session_state.selection_df["Select"] = False
                        st.session_state['verification_key'] = st.session_state.get('verification_key', 0) + 1
                        st.rerun()

                # Editor
                if 'selection_df' not in st.session_state:
                    # Should have been created above, but fallback if state was cleared partially
                     selection_data = []
                     for i, t in enumerate(tasks):
                         has_truth = bool(t.get('truth') and t.get('truth').strip())
                         truth_display = "✅ Yes" if has_truth else "❌ No"
                         selection_data.append({
                             "Select": True, 
                             "Prompt": t.get('prompt'),
                             "Engine": t.get('engine'), 
                             "Truth Defined?": truth_display,
                             "TaskID": i
                         })
                     st.session_state.selection_df = pd.DataFrame(selection_data)

                edited_df = st.data_editor(
                    st.session_state.selection_df,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Verify?", default=True),
                        "Prompt": st.column_config.TextColumn("Prompt", disabled=True),
                        "Engine": st.column_config.TextColumn("Engine", disabled=True),
                        "Truth Defined?": st.column_config.TextColumn("Truth Defined?", disabled=True),
                        "TaskID": None 
                    },
                    disabled=["Prompt", "Engine", "Truth Defined?"],
                    hide_index=True,
                    width="stretch",
                    key=f"prompt_selector_{st.session_state.get('verification_key', 0)}" # Unique key
                )
            
                # Sync edits back to session state so they persist across reruns
                st.session_state.selection_df = edited_df
            
                # Filter based on selection
                selected_indices = edited_df[edited_df["Select"]]["TaskID"].tolist()
                tasks_to_verify = [tasks[i] for i in selected_indices]
            
                st.caption(f"Selected {len(tasks_to_verify)} out of {len(tasks)} items.")
            
                # Cost Update based on selection
                prompts_to_verify_text = [t.get('prompt', '') for t in tasks_to_verify]
                est_output_tokens = len(tasks_to_verify) * 10  # Roughly 10 tokens for a verify response
                est_input_tokens, est_cost = estimate_tokens_and_cost(prompts_to_verify_text, output_tokens_est=est_output_tokens)
                st.caption(f"💰 Est. Verify Cost: ~${est_cost:.4f} ({int(est_input_tokens)} input tokens)")

                # 3. VERIFY STAGE
                if st.button("Verify Selected", type="primary"):
                     # Check for Missing Truth
                    no_truth_count = sum(1 for t in tasks_to_verify if not t.get('truth') or not t.get('truth').strip())
                    if no_truth_count > 0:
                         st.warning(f"⚠️ **{no_truth_count}** selected prompts do not have a defined Ground Truth and will be skipped.")

                     # Progress Bar UI
                    prog_bar = st.progress(0, text="Starting verification...")
                
                    def update_progress(current, total, msg):
                        percent = min(current / total, 1.0) if total > 0 else 0
                        prog_bar.progress(percent, text=msg)
                
                    with st.spinner("Verifying with AI..."):
                        results = backend.verify_accuranker_data(
                            tasks_to_verify,
                            client,
                            progress_callback=update_progress
                        )
                        prog_bar.empty()
                        st.session_state.truth_results = sorted(results, key=lambda x: x.get('Score', 0), reverse=True)
                        st.session_state.last_truth_check = time.strftime("%H:%M:%S")

    
        # Results Display
        if 'truth_results' in st.session_state and st.session_state.truth_results:
            results = st.session_state.truth_results
        
            # Checking for API errors in the list
            if isinstance(results, list) and len(results) > 0 and "error" in results[0]:
                st.error(results[0]["error"])
            elif not results:
                st.warning("No results to display.")
            else:
                df_truth = pd.DataFrame(results)
            
                # View Toggle
                view_mode = st.radio("View Mode", ["Feed View", "Table View"], horizontal=True)
            
                st.markdown(f"**Verified {len(results)} items.** (Last checked: {st.session_state.get('last_truth_check')})")
            
                # Calculate Average Scores
                engine_stats = {}
                for res in results:
                    eng = res.get('Engine', 'Unknown')
                    score = res.get('Score', 0)
                    verdict = res.get('Verdict', 'Unknown')
                
                    if eng not in engine_stats:
                        engine_stats[eng] = {"scores": [], "pass": 0, "fail": 0, "partial": 0}
                
                    engine_stats[eng]["scores"].append(score)
                    if verdict == "Pass": engine_stats[eng]["pass"] += 1
                    elif verdict == "Fail": engine_stats[eng]["fail"] += 1
                    elif verdict == "Partial": engine_stats[eng]["partial"] += 1
            
                # Display Metrics
                st.markdown("### Truth Score")
                cols = st.columns(len(engine_stats)) if engine_stats else [st.empty()]
                for idx, (eng, stats) in enumerate(engine_stats.items()):
                    scores = stats["scores"]
                    avg = sum(scores) / len(scores) if scores else 0
                
                    # Color code average
                    score_color = "#e74c3c" # Red
                    if avg >= 80: score_color = "#2ecc71" # Green
                    elif avg >= 50: score_color = "#f1c40f" # Orange
                
                    with cols[idx]:
                        st.markdown(f"<div style='text-align: center;'><span style='font-size: 0.9em; font-weight: bold;'>{eng.upper()}</span><br><span style='font-size: 1.8em; font-weight: bold; color: {score_color};'>{avg:.1f}</span></div>", unsafe_allow_html=True)
                        # Breakdown
                        st.markdown(
                            f"""
                            <div style='text-align: center; font-size: 0.8em; color: #888;'>
                            <span style='color:#2ecc71'>Pass: {stats['pass']}</span> | 
                            <span style='color:#f1c40f'>Partial: {stats['partial']}</span> | 
                            <span style='color:#e74c3c'>Fail: {stats['fail']}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                st.divider()

                if view_mode == "Table View":
                    # Prepare data for table: Sources is list of dicts, map to count
                    df_table = df_truth.copy()
                    df_table['Source Count'] = df_table['Sources'].apply(lambda x: len(x) if isinstance(x, list) else 0)

                    # Color code verdict
                    def highlight_verdict(val):
                        color = 'grey'
                        if val == 'Pass': color = '#2ecc71'
                        elif val == 'Fail': color = '#e74c3c'
                        elif val == 'Partial': color = '#f1c40f'
                        return f'color: {color}; font-weight: bold'

                    st.dataframe(
                        df_table,
                        column_config={
                            "Verdict": st.column_config.TextColumn("Verdict"),
                            "Score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100),
                            "Engine": st.column_config.TextColumn("Engine", width="small"),
                            "Prompt": st.column_config.TextColumn("Prompt", width="medium"),
                            "Truth": st.column_config.TextColumn("Truth", width="medium"),
                            "Reason": st.column_config.TextColumn("Reason", width="medium"),
                            "Source Count": st.column_config.NumberColumn("Sources", help="Count of source URLs"),
                            "Sources": None, # Hide raw sources
                        },
                        width="stretch",
                        hide_index=True
                    )
                
                else: # Feed View
                    # Filter Controls
                    all_verdicts = sorted(list(set([r.get('Verdict', 'Unknown') for r in results])))
                    filtered_verdicts = st.multiselect("Filter by Status", all_verdicts, default=all_verdicts)
                
                    # Group by Prompt
                    grouped_results = {}
                    for item in results:
                        p_txt = item.get('Prompt')
                        if p_txt not in grouped_results:
                            grouped_results[p_txt] = []
                        grouped_results[p_txt].append(item)
                
                    for prompt_txt, items in grouped_results.items():
                        # Filter items based on verdict if filter is active
                        filtered_items = [i for i in items if not filtered_verdicts or i.get('Verdict') in filtered_verdicts]
                        if not filtered_items and filtered_verdicts:
                            continue # Skip prompt if no engines match filter
                        
                        # If we have items but some were filtered out, should we still show the prompt? 
                        # Probably yes, but only show tabs for matching engines.
                        # Or should we hide the whole prompt if ALL engines are filtered out? -> Yes (done above)
                    
                        with st.container(border=True):
                            st.markdown(f"**Prompt:** *{prompt_txt}*")
                        
                            # Tabs for Engines
                            # Use filtered items to determine which tabs to show? 
                            # Or show all but indicate filtered? 
                            # User said "filter the cards", which usually means hide non-matching.
                            # Since we group by prompt, we should show the prompt if AT LEAST ONE engine matches.
                            # And inside, what tabs? 
                            # Let's show only tabs that match the filter.
                        

                            tab_labels = []
                            for i in filtered_items:
                                eng = i.get('Engine', 'Unknown').upper()
                                verdict = i.get("Verdict", "Unknown")
                            
                                icon = "⚪"
                                if verdict == "Pass": icon = "🟢"
                                elif verdict == "Partial": icon = "🟡"
                                elif verdict == "Fail": icon = "🔴"
                            
                                tab_labels.append(f"{eng} {icon}")

                            tabs = st.tabs(tab_labels)
                        
                            for t, item in zip(tabs, filtered_items):
                                with t:
                                    verdict = item.get("Verdict", "Unknown")
                                    score = item.get("Score", 0)
                                
                                    # Score Color Logic
                                    score_color = "#e74c3c" # Red
                                    if score >= 80: score_color = "#2ecc71" # Green
                                    elif score >= 50: score_color = "#f1c40f" # Orange
                                
                                    verdict_colors = {
                                        "Pass": "green", "Fail": "red", "Partial": "orange", "Skipped": "grey", "Unknown": "grey"
                                    }
                                    v_color = verdict_colors.get(verdict, "blue")
                                
                                    # Header
                                    col_h1, col_h2 = st.columns([5, 1])
                                    with col_h1:
                                        st.markdown(f"**:{v_color}[{verdict.upper()}]**")
                                    with col_h2:
                                         st.markdown(f"<h3 style='text-align: right; color: {score_color}; margin:0; padding:0;'>{score}</h3>", unsafe_allow_html=True)
                                
                                    # Expandable Body
                                    c_a, c_b = st.columns(2)
                                    with c_a:
                                        with st.expander("Ground Truth", expanded=False):
                                            truth_text = item.get('Truth')
                                            if not truth_text:
                                                st.caption("No truth defined.")
                                            else:
                                                st.info(truth_text)
                                        
                                    with c_b:
                                        with st.expander("AI Answer", expanded=False):
                                            st.code(item.get('AI Response'), language="text", wrap_lines=True)
                                
                                    # Sources Expander
                                    sources = item.get('Sources', [])
                                    source_count = item.get('Source Count', 0)
                                    if sources:
                                         with st.expander(f"Source URLs ({source_count})", expanded=False):
                                             st.dataframe(sources, hide_index=True, width="stretch")
        
                                    # Analysis Footer (Always visible)
                                    if item.get('Reason'):
                                         st.markdown(f"**Analysis:** {item.get('Reason')}")

    with tab_extract:
        st.markdown("## ⛏️ Source Extraction")
        st.info("Extract and aggregate all external sources cited by AI engines for a specific brand and tag.")
        
        # We can reuse ACCURANKER_BRANDS from above
        with st.container(border=True):
            col_brand, col_tag, col_date = st.columns([1, 1, 1])
            with col_brand:
                brand_options = ["All Brands (Excl. GEO/Inst)"] + list(ACCURANKER_BRANDS.keys())
                ext_brand_name = st.selectbox("Select Brand", brand_options, index=1, key="ext_brand")
                if ext_brand_name == "All Brands (Excl. GEO/Inst)":
                    ext_brand_id = [v for k, v in ACCURANKER_BRANDS.items() if k not in ["GEO Experiments", "Saxo Institutional"]]
                    tag_brand_id = ACCURANKER_BRANDS.get("Saxo DK", 10000087)
                else:
                    ext_brand_id = ACCURANKER_BRANDS[ext_brand_name]
                    tag_brand_id = ext_brand_id
                
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                ext_cache_key = f"tags_{tag_brand_id}"
                
                if ext_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              st.session_state[ext_cache_key] = backend.fetch_unique_tags(tag_brand_id, accuranker_token)
                          else:
                              st.session_state[ext_cache_key] = {}
                
                tags_map = st.session_state.get(ext_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                    
                ext_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix, key="ext_tag") if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial", key="ext_tag_fallback")
                ext_tag_clean = ext_tag_str.split(" (")[0] if " (" in str(ext_tag_str) else ext_tag_str
                
            with col_date:
                from datetime import datetime, timedelta
                six_months_ago = datetime.today() - timedelta(days=180)
                date_range = st.date_input("Date Range", value=(six_months_ago, datetime.today()), max_value=datetime.today())
                
        # Filters and Button
        col_filters, col_btn = st.columns([2, 1])
        with col_filters:
            filter_reddit = st.checkbox("Filter out Reddit", value=True, help="Removes sources from reddit.com")
            filter_brokerchooser = st.checkbox("Filter out BrokerChooser", value=True, help="Removes sources from brokerchooser.com")
            
            calc_latest_only = st.checkbox("Calculate using Latest Snapshot only", value=True, help="Aligns with AccuRanker UI. Uncheck to include all snapshots in the date range.")
            
            comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 3])
            with comp_col1:
                filter_competitors = st.checkbox("Exclude competitors", value=False, help="Removes sources listed in competitors.txt")
            with comp_col2:
                with st.popover("See included competitors"):
                    try:
                        with open("competitors.txt", "r", encoding="utf-8") as f:
                            st.code(f.read(), language="text")
                    except FileNotFoundError:
                        st.info("competitors.txt not found.")
            with comp_col3:
                st.empty()
            
        with col_btn:
             fetch_sources_btn = st.button("Fetch Sources", type="primary", width="stretch")
             
        if fetch_sources_btn:
             if len(date_range) != 2:
                 st.error("Please select both a start and end date.")
             elif not accuranker_token:
                 st.error("Missing ACCURANKER_TOKEN in secrets.")
             else:
                 start_date, end_date = date_range
                 with st.spinner(f"Fetching sources for {ext_brand_name} from {start_date} to {end_date}..."):
                     df_sources = backend.fetch_accuranker_sources(
                         ext_brand_id,
                         ext_tag_clean,
                         start_date,
                         end_date,
                         accuranker_token,
                         calculate_latest_only=calc_latest_only
                     )
                     
                     if df_sources.empty:
                         st.warning("No sources found for the selected criteria.")
                         st.session_state.source_extraction_df = None
                     else:
                         # Apply Filters
                         if filter_reddit:
                             df_sources = df_sources[~df_sources['Domain'].str.contains('reddit.com', case=False, na=False)]
                         if filter_brokerchooser:
                             df_sources = df_sources[~df_sources['Domain'].str.contains('brokerchooser.com', case=False, na=False)]
                         if filter_competitors:
                             try:
                                 with open('competitors.txt', 'r', encoding='utf-8') as f:
                                     competitors = [line.strip().lower() for line in f if line.strip() and not line.startswith('#')]
                                 if competitors:
                                     # Filter out any domain that contains any of the competitor strings
                                     mask = df_sources['Domain'].apply(lambda d: any(comp in str(d).lower() for comp in competitors))
                                     df_sources = df_sources[~mask]
                             except FileNotFoundError:
                                 st.warning("competitors.txt not found in the root directory. Filtering skipped.")

                         # Recalculate percentages after filtering? 
                         # Usually percentages should be of the *total* prompts, so we might want to keep the original Cited percentage even if we filter.
                         # Based on user description, filtering out results means they just don't show up in the table.
                         st.session_state.source_extraction_df = df_sources

        # Display Results
        if 'source_extraction_df' in st.session_state and st.session_state.source_extraction_df is not None:
             df_disp = st.session_state.source_extraction_df
             
             unique_domains = df_disp['Domain'].nunique()
             st.success(f"Extracted {unique_domains} unique domains and {len(df_disp)} unique URLs.")
             st.markdown("### Latest Extracted Sources")
             
             # Bar Chart visualization
             df_chart = df_disp[['Domain', 'Domain Cited (%)']].drop_duplicates()
             if not df_chart.empty:
                 max_domains = len(df_chart)
                 num_results = st.slider("Number of domains to show in chart", min_value=1, max_value=max_domains, value=(1, min(50, max_domains)))
                 df_chart_filtered = df_chart.iloc[num_results[0]-1:num_results[1]]

                 chart = alt.Chart(df_chart_filtered).mark_bar(color='#3498db').encode(
                     x=alt.X('Domain:N', sort='-y', axis=alt.Axis(labelAngle=-45, title='Domain')),
                     y=alt.Y('Domain Cited (%):Q', title='Domain Cited (%)'),
                     tooltip=['Domain', alt.Tooltip('Domain Cited (%):Q', format='.1f')]
                 ).properties(
                     height=400
                 ).interactive()
                 st.altair_chart(chart, use_container_width=True)
                 st.markdown("---")
             
             # CSV Download Options
             col_dl1, col_dl2 = st.columns([1, 2])
             with col_dl1:
                 csv_format = st.selectbox(
                     "CSV Export Format", 
                     ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"], 
                     key="csv_format_source",
                     label_visibility="collapsed"
                 )
             
             if "EU Excel" in csv_format:
                 df_csv = df_disp.copy()
                 for col in df_csv.select_dtypes(include=['float64', 'float32']).columns:
                     df_csv[col] = df_csv[col].apply(lambda x: str(x).replace('.', ','))
                 csv = df_csv.to_csv(index=False, sep=';').encode('utf-8')
             else:
                 csv = df_disp.to_csv(index=False).encode('utf-8')
             
             with col_dl2:
                 st.download_button(
                     label="⬇️ Download CSV",
                     data=csv,
                     file_name=f"{ext_brand_name.replace(' ', '_')}_{ext_tag_clean.replace(' ', '_')}_LLM_Sources.csv",
                     mime="text/csv",
                 )
             
             st.dataframe(
                 df_disp,
                 column_config={
                     "Domain": st.column_config.TextColumn("Domain"),
                     "Domain Prompts": st.column_config.NumberColumn("Domain Prompts"),
                     "Domain Cited (%)": st.column_config.ProgressColumn(
                         "Domain Cited (%)",
                         format="%.1f%%",
                         min_value=0,
                         max_value=100
                     ),
                     "Full URL": st.column_config.LinkColumn("Full URL"),
                     "Prompts": st.column_config.NumberColumn("URL Prompts"),
                     "URL Cited (%)": st.column_config.ProgressColumn(
                         "URL Cited (%)",
                         format="%.1f%%",
                         min_value=0,
                         max_value=100
                     )
                 },
                 hide_index=True,
                 width="stretch",
                 height=600
             )

    with tab_source_trends:
        st.markdown("## 📈 Source Trends")
        st.info("Track the historical usage of specific domains cited by AI engines over time.")
        
        with st.container(border=True):
            col_brand, col_comp, col_tag, col_date = st.columns([1, 1, 1, 1])
            with col_brand:
                brand_options_trends = ["All Brands (Excl. GEO/Inst)"] + list(ACCURANKER_BRANDS.keys())
                trends_brand_name = st.selectbox("Select Brand", brand_options_trends, index=1, key="trends_brand")
                if trends_brand_name == "All Brands (Excl. GEO/Inst)":
                    trends_brand_id = [v for k, v in ACCURANKER_BRANDS.items() if k not in ["GEO Experiments", "Saxo Institutional"]]
                    # Use Saxo DK for tag fetching as a fallback
                    tag_brand_id_trends = ACCURANKER_BRANDS.get("Saxo DK", 10000087)
                else:
                    trends_brand_id = ACCURANKER_BRANDS[trends_brand_name]
                    tag_brand_id_trends = trends_brand_id
                    
            with col_comp:
                comp_options_trends = ["None"] + list(ACCURANKER_BRANDS.keys())
                trends_comp_name = st.selectbox("Compare With (Optional)", comp_options_trends, key="trends_comp_brand")
                trends_comp_id = ACCURANKER_BRANDS.get(trends_comp_name)
                    
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                trends_cache_key = f"tags_{tag_brand_id_trends}"
                
                if trends_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              st.session_state[trends_cache_key] = backend.fetch_unique_tags(tag_brand_id_trends, accuranker_token)
                          else:
                              st.session_state[trends_cache_key] = {}
                
                tags_map = st.session_state.get(trends_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                    
                trends_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix, key="trends_tag") if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial", key="trends_tag_fallback")
                trends_tag_clean = trends_tag_str.split(" (")[0] if " (" in str(trends_tag_str) else trends_tag_str
                
            with col_date:
                from datetime import datetime, timedelta
                six_months_ago = datetime.today() - timedelta(days=180)
                trends_date_range = st.date_input("Date Range", value=(six_months_ago, datetime.today()), max_value=datetime.today(), key="trends_date")
                
        col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.5, 1])
        with col_btn1:
            rolling_avg_options = ["None", "3-day", "Weekly (7-day)", "Bi-weekly (14-day)", "Monthly (30-day)"]
            trends_rolling_avg = st.selectbox("Rolling Average", rolling_avg_options, index=2, key="trends_rolling")
        with col_btn2:
            st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
            breakdown_llm = st.checkbox("Breakdown by LLM (Display data per engine)", value=False, key="trends_breakdown")
        with col_btn3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            fetch_trends_btn = st.button("Fetch Source Trends", type="primary", width="stretch")
             
        if fetch_trends_btn:
             if len(trends_date_range) != 2:
                 st.error("Please select both a start and end date.")
             elif not accuranker_token:
                 st.error("Missing ACCURANKER_TOKEN in secrets.")
             elif trends_comp_id and trends_brand_id == trends_comp_id:
                 st.warning("Cannot compare a brand to itself.")
             else:
                 start_date, end_date = trends_date_range
                 with st.spinner(f"Fetching source trends..."):
                     df_primary = backend.fetch_source_trends(
                         trends_brand_id,
                         trends_tag_clean,
                         start_date,
                         end_date,
                         accuranker_token
                     )
                     
                     if not df_primary.empty:
                         if trends_comp_id:
                             df_primary['Domain'] = df_primary['Domain'] + f" ({trends_brand_name})"
                             
                         df_final = df_primary
                         
                         if trends_comp_id:
                             df_comp = backend.fetch_source_trends(
                                 trends_comp_id,
                                 trends_tag_clean,
                                 start_date,
                                 end_date,
                                 accuranker_token
                             )
                             if not df_comp.empty:
                                 df_comp['Domain'] = df_comp['Domain'] + f" ({trends_comp_name})"
                                 df_final = pd.concat([df_primary, df_comp], ignore_index=True)
                                 
                         st.session_state.source_trends_df = df_final
                         st.success(f"Successfully fetched trend data for {len(df_final['Domain'].unique())} unique domains.")
                     else:
                         st.warning("No trend data found for the selected criteria.")
                         st.session_state.source_trends_df = None
                         
        if 'source_trends_df' in st.session_state and st.session_state.source_trends_df is not None:
             df_trends = st.session_state.source_trends_df.copy()
             
             # Extract unique domains and find latest usage for sorting (Aggregated only)
             agg_df = df_trends[df_trends['Engine'] == 'Aggregated']
             if agg_df.empty:
                 agg_df = df_trends # fallback
                 
             latest_date = agg_df['Date'].max()
             latest_df = agg_df[agg_df['Date'] == latest_date]
             
             # Create a dictionary of domains -> latest %
             domain_latest_pct = {}
             for d in agg_df['Domain'].unique():
                 match = latest_df[latest_df['Domain'] == d]
                 if not match.empty:
                     domain_latest_pct[d] = match['Domain Cited (%)'].values[0]
                 else:
                     domain_latest_pct[d] = 0.0
                     
             # Sort domains based on latest pct (descending)
             all_domains_sorted = sorted(domain_latest_pct.keys(), key=lambda x: domain_latest_pct[x], reverse=True)
             
             st.markdown("### Select Domains to Track")
             col_dom1, col_dom2, col_dom3, col_dom4 = st.columns([1, 1, 1, 2])
             
             with col_dom1:
                 # Standard Reddit checkboxes. Handle comparison naming.
                 reddit_names = [d for d in all_domains_sorted if d.startswith("reddit.com")]
                 track_reddit = st.checkbox("reddit.com", value=len(reddit_names)>0, disabled=len(reddit_names)==0, key="tr_reddit")
             
             with col_dom2:
                 bc_names = [d for d in all_domains_sorted if d.startswith("brokerchooser.com")]
                 track_bc = st.checkbox("brokerchooser.com", value=False, disabled=len(bc_names)==0, key="tr_bc")
                 
             with col_dom3:
                 yt_names = [d for d in all_domains_sorted if d.startswith("youtube.com")]
                 track_yt = st.checkbox("youtube.com", value=False, disabled=len(yt_names)==0, key="tr_yt")
                 
             with col_dom4:
                 # Exclude reddit, bc, and yt from others
                 other_domains = [d for d in all_domains_sorted if not (d.startswith("reddit.com") or d.startswith("brokerchooser.com") or d.startswith("youtube.com"))]
                 selected_others = st.multiselect("Other Found Sources (Sorted by latest %)", options=other_domains, default=[], key="tr_others")
                 
             # Combine selected domains
             selected_domains = selected_others.copy()
             if track_reddit:
                 selected_domains.extend(reddit_names)
             if track_bc:
                 selected_domains.extend(bc_names)
             if track_yt:
                 selected_domains.extend(yt_names)
                 
             if not selected_domains:
                 st.info("Please select at least one domain to visualize.")
             else:
                 # Filter Dataframe for plot
                 plot_df = df_trends[df_trends['Domain'].isin(selected_domains)].copy()
                 
                 if breakdown_llm:
                     plot_df = plot_df[plot_df['Engine'] != 'Aggregated'].copy()
                     plot_df['Series'] = plot_df['Domain'] + " [" + plot_df['Engine'] + "]"
                     
                     chart_color = alt.Color('Series:N', legend=alt.Legend(title="Domain [LLM]", orient="right"))
                     chart_title = "Trend of AI Citing Domains by LLM"
                     chart_tooltips = [
                         alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date'),
                         'Domain:N',
                         'Engine:N',
                         alt.Tooltip('Domain Cited (%):Q', format='.1f', title='Cited (%)'),
                         'Domain Prompts:Q',
                         'Total Prompts:Q'
                     ]
                 else:
                     plot_df = plot_df[plot_df['Engine'] == 'Aggregated'].copy()
                     plot_df['Series'] = plot_df['Domain']
                     
                     chart_color = alt.Color('Domain:N', legend=alt.Legend(title="Domain", orient="bottom"))
                     chart_title = "Trend of AI Citing Domains"
                     chart_tooltips = [
                         alt.Tooltip('Date:T', format='%Y-%m-%d', title='Date'),
                         'Domain:N',
                         alt.Tooltip('Domain Cited (%):Q', format='.1f', title='Cited (%)'),
                         'Domain Prompts:Q',
                         'Total Prompts:Q'
                     ]

                 # 2. Apply Rolling Average
                 period = 1
                 if trends_rolling_avg == "3-day": period = 3
                 elif trends_rolling_avg == "Weekly (7-day)": period = 7
                 elif trends_rolling_avg == "Bi-weekly (14-day)": period = 14
                 elif trends_rolling_avg == "Monthly (30-day)": period = 30
                 
                 if period > 1:
                     plot_df = plot_df.sort_values(by=['Date'])
                     plot_df['Domain Cited (%)'] = plot_df.groupby('Series')['Domain Cited (%)'].transform(lambda x: x.rolling(window=period, min_periods=1).mean())
                 
                 # 3. Render Scorecards
                 st.markdown("### Domain Performance Trends")
                 if period > 1:
                     st.caption(f"Scorecards display the {trends_rolling_avg.lower()} average, and absolute change from period start.")
                 else:
                     st.caption("Scorecards display the absolute change from period start.")
                     
                 unique_series = plot_df['Series'].unique()
                 score_cols = st.columns(min(len(unique_series), 4) if len(unique_series) > 0 else 1)
                 
                 for i, series_name in enumerate(unique_series):
                     s_data = plot_df[plot_df['Series'] == series_name].sort_values('Date')
                     if not s_data.empty:
                         start_val = s_data['Domain Cited (%)'].iloc[0]
                         end_val = s_data['Domain Cited (%)'].iloc[-1]
                         diff = end_val - start_val
                         diff_str = f"{diff:+.2f}%"
                         
                         with score_cols[i % 4]:
                             st.metric(label=series_name, value=f"{end_val:.2f}%", delta=diff_str)
                             
                 st.markdown("---")
                     
                 # 4. Render Chart
                 chart = alt.Chart(plot_df).mark_line(point=True, strokeWidth=3).encode(
                     x=alt.X('Date:T', axis=alt.Axis(title='Date', format='%b %d')),
                     y=alt.Y('Domain Cited (%):Q', axis=alt.Axis(title='Citations (%)')),
                     color=chart_color,
                     tooltip=chart_tooltips
                 ).properties(
                     height=400,
                     title=chart_title
                 ).interactive()
                 
                 st.altair_chart(chart, use_container_width=True)
                 
                 # Display Data Table
                 st.markdown("### Data Details")
                 # Pivot table for better viewing
                 pivot_base = plot_df.copy()
                 if breakdown_llm:
                     pivot_col = 'Series'
                 else:
                     pivot_col = 'Domain'
                     
                 # Important: if multiple rows map to the same date and series, we should aggregate or guarantee uniqueness
                 # They should be unique, but let's be safe.
                 pivot_df = pivot_base.pivot_table(index='Date', columns=pivot_col, values='Domain Cited (%)', aggfunc='mean').reset_index()
                 
                 # Ensure Date is a string for display
                 if pd.api.types.is_datetime64_any_dtype(pivot_df['Date']):
                     pivot_df['Date'] = pivot_df['Date'].dt.strftime('%Y-%m-%d')
                 else:
                     pivot_df['Date'] = pivot_df['Date'].astype(str)
                 pivot_df = pivot_df.fillna(0) # Fill missing dates/domains with 0
                 
                 # Format percentages
                 for col in pivot_df.columns:
                     if col != 'Date':
                         pivot_df[col] = pivot_df[col].map(lambda x: f"{x:.2f}%")
                         
                 st.dataframe(pivot_df, use_container_width=True, hide_index=True)
                 
                 # Raw Data underneath for exact numbers
                 with st.expander("View Raw Export Data", expanded=False):
                     st.dataframe(plot_df, use_container_width=True, hide_index=True)
                 
                 # Export buttons
                 export_col1, export_col2 = st.columns([2, 1])
                 with export_col1:
                     csv_standard = plot_df.to_csv(index=False).encode('utf-8')
                     csv_eu = plot_df.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
                     
                     sub_c1, sub_c2 = st.columns(2)
                     with sub_c1:
                         st.download_button(
                             label="📥 Download Standard CSV",
                             data=csv_standard,
                             file_name=f"source_trends_{trends_brand_name.replace(' ', '_')}.csv",
                             mime="text/csv",
                             use_container_width=True,
                             key="dl_trends_std"
                         )
                     with sub_c2:
                         st.download_button(
                             label="📥 Download EU CSV (; delimited)",
                             data=csv_eu,
                             file_name=f"source_trends_eu_{trends_brand_name.replace(' ', '_')}.csv",
                             mime="text/csv",
                             use_container_width=True,
                             key="dl_trends_eu"
                         )

    with tab_scraper:
        st.markdown("## 🕵️‍♂️ Competitor Scraper")
        st.info("Find unnamed or untracked competitor brands in AI search responses.")
        
        with st.container(border=True):
            col_brand, col_tag = st.columns([1, 1])
            with col_brand:
                scrap_brand_name = st.selectbox("Select Market (Brand)", list(ACCURANKER_BRANDS.keys()), key="scrap_brand")
                scrap_brand_id = ACCURANKER_BRANDS[scrap_brand_name]
                
            with col_tag:
                accuranker_token = st.secrets.get("ACCURANKER_TOKEN")
                scrap_cache_key = f"tags_{scrap_brand_id}"
                
                if scrap_cache_key not in st.session_state:
                     with st.spinner("Loading Tags..."):
                          if accuranker_token:
                              st.session_state[scrap_cache_key] = backend.fetch_unique_tags(scrap_brand_id, accuranker_token)
                          else:
                              st.session_state[scrap_cache_key] = {}
                
                tags_map = st.session_state.get(scrap_cache_key, {})
                tag_options = [f"{t} ({c})" for t, c in tags_map.items()]
                
                default_ix = 0
                for i, opt in enumerate(tag_options):
                    if "Commercial" in opt:
                        default_ix = i
                        break
                    
                scrap_tag_str = st.selectbox("Tag Filter", tag_options, index=default_ix, key="scrap_tag") if tag_options else st.text_input("Tag Filter (No tags found)", value="Commercial", key="scrap_tag_fallback")
                scrap_tag_clean = scrap_tag_str.split(" (")[0] if " (" in str(scrap_tag_str) else scrap_tag_str
                
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                fetch_scrap_btn = st.button("Discover Competitors", type="primary", width="stretch", key="scrap_btn")
            with col_btn2:
                view_tracked_btn = st.button("View Tracked Competitors", width="stretch", key="view_tracked_btn")
            
        if view_tracked_btn:
            if not accuranker_token:
                st.error("Missing ACCURANKER_TOKEN in secrets.")
            else:
                with st.spinner(f"Fetching tracked competitors for {scrap_brand_name}..."):
                    tracked_details = backend.fetch_competitor_details(scrap_brand_id, accuranker_token)
                    
                    if not tracked_details:
                        st.warning("No tracked competitors found for this brand.")
                        st.session_state.scraper_df = None
                    else:
                        table_data = []
                        for t_brand, details in tracked_details.items():
                            alt_names = ", ".join(details.get("brand_list", []))
                            table_data.append({
                                "Status": "Tracked ✅",
                                "Brand Name": t_brand,
                                "Website URL": details.get("domain", ""),
                                "Alternative Names": alt_names
                            })
                        st.session_state.scraper_df = pd.DataFrame(table_data)
                        st.success(f"Found {len(tracked_details)} tracked competitors.")

        if fetch_scrap_btn:
            if not accuranker_token:
                st.error("Missing ACCURANKER_TOKEN in secrets.")
            else:
                with st.spinner(f"Fetching prompts for {scrap_brand_name}..."):
                    raw_prompts = backend.fetch_accuranker_prompts_raw(scrap_brand_id, accuranker_token)
                    
                    if not raw_prompts:
                        st.warning("No prompts found for this brand.")
                    else:
                        # Filter by tag
                        tag_lower = scrap_tag_clean.lower()
                        filtered_prompts = []
                        for p in raw_prompts:
                            p_tags = [t.lower() for t in (p.get('tags') or [])]
                            if tag_lower in p_tags:
                                filtered_prompts.append(p)
                                
                        if not filtered_prompts:
                            st.warning(f"No prompts found with tag: {scrap_tag_clean}")
                        else:
                            # Debug info
                            responses_found = 0
                            for p in filtered_prompts:
                                for r in p.get('results', []):
                                    if r.get('prompt_response'):
                                        responses_found += 1
                            
                            st.info(f"Analyzing {len(filtered_prompts)} prompts ({responses_found} total responses) to discover new competitors...")
                            
                            tracked_details = backend.fetch_competitor_details(scrap_brand_id, accuranker_token)
                            api_keys = {
                                'google': st.secrets.get("GEMINI_API_KEY"),
                                'openai': st.secrets.get("OPENAI_API_KEY")
                            }
                            
                            new_competitors = backend.discover_new_competitors(filtered_prompts, tracked_details, api_keys)
                            
                            if not new_competitors:
                                st.success("No new untracked competitors were found.")
                                st.session_state.scraper_df = None
                            else:
                                st.session_state.scraper_df = pd.DataFrame(new_competitors)
                                st.success(f"Discovered {len(new_competitors)} potential new competitors!")
                                
        if 'scraper_df' in st.session_state and st.session_state.scraper_df is not None:
            df_scrap = st.session_state.scraper_df
            
            st.markdown("### Discovered Competitors")
            st.dataframe(
                df_scrap,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Brand Name": st.column_config.TextColumn("Competitor Brand Name"),
                    "Website URL": st.column_config.LinkColumn("Website URL"),
                    "Alternative Names": st.column_config.TextColumn("Alternative Spelling Versions")
                },
                hide_index=True,
                width="stretch"
            )
            
            # CSV Download Options
            col_csv1, col_csv2 = st.columns([2, 1])
            with col_csv1:
                csv_format = st.selectbox(
                    "CSV Export Format", 
                    ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"], 
                    key="csv_format_scraper"
                )
            with col_csv2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                
                if "EU Excel" in csv_format:
                    df_csv = df_scrap.copy()
                    csv_bytes = df_csv.to_csv(index=False, sep=';').encode('utf-8')
                else:
                    csv_bytes = df_scrap.to_csv(index=False).encode('utf-8')
                    
                st.download_button(
                    label="⬇️ Download Discovered Competitors",
                    data=csv_bytes,
                    file_name=f"Discovered_Competitors_{scrap_brand_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="scraper_download"
                )

# --- PAGE: AI POWERED TOOLS ---
elif current_page == "AI Powered Tools":
    st.markdown("## 🤖 AI Powered Tools")
    tabs = st.tabs(["AI FlexSheet"])
    
    with tabs[0]:
        col_title, col_clear = st.columns([0.8, 0.2])
        with col_title:
            st.markdown("#### 📓 AI FlexSheet")
        with col_clear:
            if st.button("🧹 Clear All", width="stretch"):
                # Clear everything related to flexsheet
                for key in list(st.session_state.keys()):
                    if "flex" in key:
                        del st.session_state[key]
                st.rerun()

        st.info("Upload your spreadsheet, map columns to variables, and process your dataset row by row using an AI prompt.")
        
        uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"], key="flex_uploader")
        if uploaded_file is not None:
            # Read file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_flex = load_csv_robustly(uploaded_file)
                else:
                    df_flex = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df_flex = None
                
            if df_flex is not None and not df_flex.empty:
                st.success(f"✅ Successfully loaded dataset with **{len(df_flex)}** total rows.")
                st.markdown("**Data Preview (first 5 rows)**")
                st.dataframe(df_flex.head(5))
                
                # Setup controls
                col1, col2 = st.columns([1, 1])
                with col1:
                    selected_cols = st.multiselect("Select Columns to use as variables", options=df_flex.columns)
                with col2:
                    model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5-mini", "gpt-5.2"]
                    selected_model = st.selectbox("Select OpenAI Model", model_options, index=0)
                    
                # Prompts management
                try:
                    with open("prompts/flexsheet_prompts.json", "r") as f:
                        flex_prompts_data = json.load(f)
                except FileNotFoundError:
                    flex_prompts_data = []
                    
                prompt_titles = [p.get("title", "Untitled") for p in flex_prompts_data]
                prompt_titles.insert(0, "Custom Prompt")
                
                selected_prompt_title = st.selectbox("Predefined Prompts", prompt_titles)
                
                default_prompt_text = ""
                output_format = "text"
                output_columns = []
                if selected_prompt_title != "Custom Prompt":
                    for p in flex_prompts_data:
                        if p.get("title") == selected_prompt_title:
                            default_prompt_text = p.get("prompt", "")
                            output_format = p.get("output_format", "text")
                            output_columns = p.get("output_columns", [])
                            break
                            
                if "last_prompt_title" not in st.session_state:
                    st.session_state.last_prompt_title = selected_prompt_title

                if selected_prompt_title != st.session_state.last_prompt_title:
                    st.session_state.flex_prompt = default_prompt_text
                    st.session_state.last_prompt_title = selected_prompt_title
                elif "flex_prompt" not in st.session_state:
                    st.session_state.flex_prompt = default_prompt_text
                    
                if selected_cols:
                    st.markdown("**Insert Variables:**")
                    # Display buttons in rows of 4, keeping them reasonably sized
                    for i in range(0, len(selected_cols), 4):
                        chunk = selected_cols[i:i+4]
                        # Create 4 columns, plus a 5th spacer column to push them left if there aren't 4 items
                        cols = st.columns(4)
                        for j, col_name in enumerate(chunk):
                            if cols[j].button(f"➕ [{col_name}]", key=f"btn_var_{col_name}_{i}_{j}", width="stretch"):
                                st.session_state.flex_prompt += f" [{col_name}]"
                                st.rerun()
                            
                if selected_prompt_title == "Custom Prompt":
                    custom_json_toggle = st.toggle("Enable JSON Output Mode")
                    if custom_json_toggle:
                        output_format = "json"
                        
                        col_input, col_btn = st.columns([0.85, 0.15])
                        with col_input:
                            custom_cols_input = st.text_input("Define JSON Output Keys (comma-separated). Press Enter or click Add! 👇", placeholder="e.g. title, summary, score")
                        with col_btn:
                            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
                            st.button("Add", width="stretch", key="add_json_keys_btn")
                            
                        if custom_cols_input:
                            output_columns = [x.strip() for x in custom_cols_input.split(",") if x.strip()]
                        
                if output_format == "json":
                    st.success(f"**JSON Mode Active:** Output will be parsed into {len(output_columns) if output_columns else 'multiple'} distinct columns: `{', '.join(output_columns)}`")
                            
                user_prompt = st.text_area("User Prompt", key="flex_prompt", height=150)
                st.caption("Use `[Column Name]` to insert variables from your selected columns. E.g. `[Title]`.")
                
                json_instruction = ""
                if output_format == "json":
                    if output_columns:
                        formatted_keys = ", ".join([f'"{k}"' for k in output_columns])
                        json_instruction = f"\n\nYou must respond strictly with a valid JSON object containing exactly the following keys: {formatted_keys}"
                    else:
                        json_instruction = "\n\nYou must respond strictly with a valid JSON object."
                    st.info(f"🤖 **Auto-Appended Instruction:** *{json_instruction}*")
                
                # Live Preview with highlighting
                if user_prompt:
                    preview_text = user_prompt
                    example_text = user_prompt
                    has_cols = bool(selected_cols)
                    
                    if has_cols:
                        first_row = df_flex.iloc[0] if not df_flex.empty else None
                        for col in selected_cols:
                            tag = f"[{col}]"
                            # Blueprint preview
                            preview_text = preview_text.replace(tag, f"<span style='color:#3498db; font-weight:bold;'>{tag}</span>")
                            
                            # Actual Example preview
                            if first_row is not None:
                                val = str(first_row[col]) if pd.notna(first_row[col]) else ""
                                example_text = example_text.replace(tag, f"<span style='color:#2ecc71; font-weight:bold;'>{val}</span>")
                                
                    st.markdown("**Live Prompt Preview:**")
                    st.markdown(f"<div style='padding:15px; border:1px solid #444; border-radius:5px; background-color: #1e1e1e; margin-bottom: 20px;'>{preview_text}</div>", unsafe_allow_html=True)
                    
                    if has_cols and not df_flex.empty:
                        st.markdown("**Example Output (Row 1):**")
                        st.markdown(f"<div style='padding:15px; border:1px solid #444; border-radius:5px; background-color: #1e1e1e; margin-bottom: 20px;'>{example_text}</div>", unsafe_allow_html=True)
                
                
                # Estimate Cost
                if user_prompt:
                    base_tokens, _ = estimate_tokens_and_cost(user_prompt, model_name=selected_model)
                    
                    dynamic_tokens = 0
                    if selected_cols:
                        # Estimate avg sequence length by looking at first 10 rows
                        sample_df = df_flex.head(10)
                        sample_texts = []
                        for col in selected_cols:
                            sample_texts.extend([str(x) for x in sample_df[col].dropna()])
                        
                        if sample_texts:
                            t_count, _ = estimate_tokens_and_cost(sample_texts, model_name=selected_model)
                            avg_col_tokens = t_count / len(sample_texts)
                            dynamic_tokens = avg_col_tokens * len(selected_cols) * len(df_flex)
                            
                    total_tokens_est = (base_tokens * len(df_flex)) + dynamic_tokens
                    
                    cost_est = (total_tokens_est / 1_000_000) * 2.50
                    st.info(f"**Estimated API Input Cost:** ~${cost_est:.4f} (Approx. {int(total_tokens_est)} tokens)")

                st.markdown("---")
                csv_format_flex = st.selectbox(
                    "CSV Export Format (Select before running)", 
                    ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"], 
                    key="csv_format_flex"
                )
                if st.button("Run FlexSheet", type="primary"):
                    if not user_prompt:
                        st.error("Please provide a User Prompt.")
                    else:
                        ai_outputs = []
                        if output_format == "json" and output_columns:
                            for col in output_columns:
                                df_flex[col] = None
                                
                        progress_bar = st.progress(0, text="Starting execution...")
                        total_rows = len(df_flex)
                        
                        for idx, row in df_flex.iterrows():
                            # Construct dynamic prompt
                            current_prompt = user_prompt + json_instruction
                            if selected_cols:
                                for col in selected_cols:
                                    tag = f"[{col}]"
                                    if tag in current_prompt:
                                        val = str(row[col]) if pd.notna(row[col]) else ""
                                        current_prompt = current_prompt.replace(tag, val)
                            
                            # Update progress
                            progress_bar.progress((idx) / total_rows, text=f"Processing row {idx + 1} of {total_rows}...")
                            
                            # API Call
                            is_json_mode = (output_format == "json")
                            res = backend.run_flexsheet_prompt(current_prompt, client, selected_model, is_json=is_json_mode)
                            
                            if is_json_mode and output_columns:
                                try:
                                    parsed_json = json.loads(res) if res else {}
                                    for col in output_columns:
                                        df_flex.at[idx, col] = parsed_json.get(col, "JSON Error")
                                except (json.JSONDecodeError, TypeError, AttributeError):
                                    for col in output_columns:
                                        df_flex.at[idx, col] = "JSON Error"
                            else:
                                ai_outputs.append(res)
                            
                        progress_bar.progress(1.0, text="FlexSheet Execution Complete!")
                        st.success("Execution complete. Check the updated data below.")
                        
                        if not (output_format == "json" and output_columns):
                            df_flex["AI Output"] = ai_outputs
                            
                        st.dataframe(df_flex)
                        
                        if "EU Excel" in csv_format_flex:
                            df_csv_flex = df_flex.copy()
                            for col in df_csv_flex.select_dtypes(include=['float64', 'float32']).columns:
                                df_csv_flex[col] = df_csv_flex[col].apply(lambda x: str(x).replace('.', ','))
                            csv_data = df_csv_flex.to_csv(index=False, sep=';').encode('utf-8')
                        else:
                            csv_data = df_flex.to_csv(index=False).encode('utf-8')
                            
                        st.download_button(
                            label="Download Updated Data as CSV",
                            data=csv_data,
                            file_name="flexsheet_results.csv",
                            mime="text/csv"
                        )

# --- PAGE: RANDOM TOOLS ---
elif current_page == "Random Tools":
    tabs = st.tabs(["Sitemap Checker", "URL Classifier", "Ahrefs Top Pages Visualizer", "GSC BigQuery Visualizer"])
    
    with tabs[0]:
        st.markdown("#### Sitemap Checker")
        
        # Initialize session state for this tool
        if "sitemap_processed" not in st.session_state:
            st.session_state.sitemap_processed = False
            st.session_state.sitemap_df = None
            st.session_state.sitemap_summary = None
            st.session_state.sitemap_category_summary = None
            st.session_state.sitemap_excel_buffer = None
            
        sitemap_urls_input = st.text_area(
            "Enter Sitemap URLs (one per line, e.g., https://www.example.com/sitemap.xml)",
            value="https://www.home.saxo/sitemap.xml\nhttps://www.bgsaxo.it/sitemap.xml",
            height=150
        )
        
        col_run, col_clear = st.columns([1, 5])
        with col_run:
            run_btn = st.button("Crawl Sitemaps", type="primary")
        with col_clear:
            clear_btn = st.button("Clear Data")
            
        if clear_btn:
            st.session_state.sitemap_processed = False
            st.session_state.sitemap_df = None
            st.session_state.sitemap_summary = None
            st.session_state.sitemap_category_summary = None
            st.session_state.sitemap_excel_buffer = None
            st.rerun()
        
        if run_btn:
            if sitemap_urls_input.strip():
                # Split by newline and filter out empty strings
                sitemap_list = [url.strip() for url in sitemap_urls_input.split('\n') if url.strip()]
                
                with st.spinner(f"Crawling {len(sitemap_list)} sitemap roots..."):
                    
                    status_placeholder = st.empty()
                    def update_status(current_url):
                        status_placeholder.info(f"Crawling: {current_url}")
                        
                    all_urls = backend.extract_all_sitemap_urls(sitemap_list, progress_callback=update_status)
                    
                    status_placeholder.empty()
                    
                    if all_urls:
                        # Categorize URLs
                        data = [{"URL": u, "Market": backend.categorize_market(u), "Website Category": backend.categorize_website(u)} for u in all_urls]
                        df_sitemap = pd.DataFrame(data)
                        summary_df = df_sitemap.groupby("Market").size().reset_index(name="Count")
                        summary_category_df = df_sitemap.groupby("Website Category").size().reset_index(name="Count")
                        
                        # Export formatting
                        import io
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            df_sitemap.to_excel(writer, sheet_name="Raw_URLs", index=False)
                            summary_df.to_excel(writer, sheet_name="Market_Summary", index=False)
                            summary_category_df.to_excel(writer, sheet_name="Category_Summary", index=False)
                            
                        # Save to session state
                        st.session_state.sitemap_processed = True
                        st.session_state.sitemap_df = df_sitemap
                        st.session_state.sitemap_summary = summary_df
                        st.session_state.sitemap_category_summary = summary_category_df
                        st.session_state.sitemap_excel_buffer = excel_buffer.getvalue()
                    else:
                        st.warning("No URLs found or unable to fetch the sitemaps.")
            else:
                st.warning("Please enter at least one sitemap URL.")

        # Display if processed
        if st.session_state.sitemap_processed:
            st.success(f"Found {len(st.session_state.sitemap_df)} unique URLs.")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("##### Extracted URLs")
                st.dataframe(st.session_state.sitemap_df, width="stretch")
                
            with c2:
                st.markdown("##### Summaries")
                sum_tabs = st.tabs(["Market Summary", "Category Summary"])
                with sum_tabs[0]:
                    st.dataframe(st.session_state.sitemap_summary, width="stretch")
                with sum_tabs[1]:
                    st.dataframe(st.session_state.sitemap_category_summary, width="stretch")
                
            st.download_button(
                label="Download Excel File",
                data=st.session_state.sitemap_excel_buffer,
                file_name="sitemap_extract.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tabs[1]:
        st.markdown("#### URL Classifier")
        st.info("💡 **Note:** All classifications are made instantly based solely on the URL structure. No external web requests or fetches are performed.")
        
        if "classifier_processed" not in st.session_state:
            st.session_state.classifier_processed = False
            st.session_state.classifier_df = None
            st.session_state.classifier_summary = None
            st.session_state.classifier_cat_summary = None
            st.session_state.classifier_lang_summary = None
            st.session_state.classifier_excel_buffer = None
            
        input_method = st.radio("Input Method", ["Paste URLs", "Upload File (CSV/XLSX)"], horizontal=True)
        
        urls_to_process = []
        original_df = None
        url_col = None
        
        if input_method == "Paste URLs":
            raw_urls = st.text_area("Paste URLs (one per line)", height=150)
            if raw_urls.strip():
                urls_to_process = [u.strip() for u in raw_urls.split('\n') if u.strip()]
                original_df = pd.DataFrame({"URL": urls_to_process})
                url_col = "URL"
                
        else:
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        original_df = load_csv_robustly(uploaded_file)
                    else:
                        original_df = pd.read_excel(uploaded_file)
                        
                    url_col = st.selectbox("Select column containing URLs", options=original_df.columns, index=0)
                    if url_col:
                        urls_to_process = original_df[url_col].dropna().astype(str).tolist()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    
        col_run_c, col_cl_c = st.columns([1, 5])
        with col_run_c:
            run_c_btn = st.button("Classify URLs", key="btn_classify", type="primary")
        with col_cl_c:
            cl_c_btn = st.button("Clear Data", key="btn_clear_classify")
            
        if cl_c_btn:
            st.session_state.classifier_processed = False
            st.session_state.classifier_df = None
            st.session_state.classifier_summary = None
            st.session_state.classifier_cat_summary = None
            st.session_state.classifier_lang_summary = None
            st.session_state.classifier_excel_buffer = None
            st.rerun()
            
        if run_c_btn:
            if not urls_to_process:
                st.warning("Please provide valid URLs to classify.")
            else:
                with st.spinner(f"Classifying {len(urls_to_process)} URLs..."):
                    markets = [backend.categorize_market(u) for u in urls_to_process]
                    websites = [backend.categorize_website(u) for u in urls_to_process]
                    languages = [backend.categorize_language(u) for u in urls_to_process]
                    
                    # Ensure original columns are kept and new ones appended at the end
                    df_out = original_df.copy()
                    df_out["Market"] = markets
                    df_out["Website Category"] = websites
                    df_out["Language"] = languages
                    
                    summary_df = df_out.groupby("Market").size().reset_index(name="Count")
                    summary_category_df = df_out.groupby("Website Category").size().reset_index(name="Count")
                    summary_lang_df = df_out.groupby("Language").size().reset_index(name="Count")
                    
                    import io
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df_out.to_excel(writer, sheet_name="Categorized_URLs", index=False)
                        summary_lang_df.to_excel(writer, sheet_name="Language_Summary", index=False)
                        summary_df.to_excel(writer, sheet_name="Market_Summary", index=False)
                        summary_category_df.to_excel(writer, sheet_name="Category_Summary", index=False)
                        
                    st.session_state.classifier_processed = True
                    st.session_state.classifier_df = df_out
                    st.session_state.classifier_summary = summary_df
                    st.session_state.classifier_cat_summary = summary_category_df
                    st.session_state.classifier_lang_summary = summary_lang_df
                    st.session_state.classifier_excel_buffer = excel_buffer.getvalue()

        if st.session_state.classifier_processed:
            st.success(f"Successfully classified {len(st.session_state.classifier_df)} URLs.")
            
            c1_c, c2_c = st.columns([2, 1])
            with c1_c:
                st.markdown("##### Categorized Data")
                st.dataframe(st.session_state.classifier_df, width="stretch")
                
            with c2_c:
                st.markdown("##### Summaries")
                sum_tabs_c = st.tabs(["Language Summary", "Market Summary", "Category Summary"])
                with sum_tabs_c[0]:
                    st.dataframe(st.session_state.classifier_lang_summary, width="stretch")
                with sum_tabs_c[1]:
                    st.dataframe(st.session_state.classifier_summary, width="stretch")
                with sum_tabs_c[2]:
                    st.dataframe(st.session_state.classifier_cat_summary, width="stretch")
                
            st.download_button(
                label="Download Excel Export",
                data=st.session_state.classifier_excel_buffer,
                file_name="url_classification.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with tabs[2]:
        st.markdown("#### Ahrefs Top Pages Visualizer")
        st.info("💡 **Instructions:** Upload an Ahrefs 'Top Pages' export CSV to visualize traffic and keyword changes grouped by markets and categories.")
        
        # UI Elements
        ahrefs_file = st.file_uploader("Upload Ahrefs Top Pages CSV", type=["csv"])
        
        if "ahrefs_processed" not in st.session_state:
            st.session_state.ahrefs_processed = False
            st.session_state.ahrefs_df = None
            
        if st.button("Clear Top Pages Data"):
            st.session_state.ahrefs_processed = False
            st.session_state.ahrefs_df = None
            st.rerun()
            
        if ahrefs_file:
            if not st.session_state.ahrefs_processed:
                try:
                    df_ahrefs = load_csv_robustly(ahrefs_file)
                    
                    if "URL" not in df_ahrefs.columns:
                        st.error("The uploaded CSV does not contain a 'URL' column. Please check the file.")
                    else:
                        with st.spinner("Classifying and processing URLs..."):
                            df_ahrefs["URL"] = df_ahrefs["URL"].astype(str)
                            
                            # Clean up relevant columns to numeric (coerce errors to NaN, then fill with 0)
                            numeric_cols = ["Traffic change", "Keywords change", "UR"]
                            for col in numeric_cols:
                                if col in df_ahrefs.columns:
                                    df_ahrefs[col] = pd.to_numeric(df_ahrefs[col], errors='coerce').fillna(0)
                                else:
                                    df_ahrefs[col] = 0.0 # Create it if missing for some reason

                            # Apply classifications
                            df_ahrefs["Market"] = df_ahrefs["URL"].apply(backend.categorize_market)
                            df_ahrefs["Website Category"] = df_ahrefs["URL"].apply(backend.categorize_website)
                            df_ahrefs["Language"] = df_ahrefs["URL"].apply(backend.categorize_language)
                            
                            st.session_state.ahrefs_df = df_ahrefs
                            st.session_state.ahrefs_processed = True
                            
                            import re
                            date_matches = re.findall(r'\d{4}-\d{2}-\d{2}', ahrefs_file.name)
                            if date_matches:
                                if len(date_matches) == 2:
                                    date_str = f"**Comparing Dates:** {date_matches[0]} vs {date_matches[1]}"
                                elif len(date_matches) == 1:
                                    date_str = f"**Date:** {date_matches[0]}"
                                else:
                                    date_str = f"**Dates:** {', '.join(date_matches)}"
                                st.session_state.ahrefs_comparison_dates = date_str
                            else:
                                st.session_state.ahrefs_comparison_dates = None
                                
                            st.success("Successfully processed Ahrefs export!")
                except Exception as e:
                    st.error(f"Error processing the Ahrefs CSV file: {e}")
                    
        if st.session_state.ahrefs_processed and st.session_state.ahrefs_df is not None:
            if st.session_state.get("ahrefs_comparison_dates"):
                st.info(f"📅 {st.session_state.ahrefs_comparison_dates} (Parsed from filename)")
                
            df = st.session_state.ahrefs_df
            
            view_tab1, view_tab3 = st.tabs(["🌍 Market Performance", "📁 Category Performance"])
            
            def style_change_gradient(s):
                s_num = pd.to_numeric(s, errors='coerce')
                max_val = s_num[s_num > 0].max() if (s_num > 0).any() else 0
                min_val = s_num[s_num < 0].min() if (s_num < 0).any() else 0
                
                styles = []
                for val in s_num:
                    if pd.isna(val) or val == 0:
                        styles.append('')
                    elif val > 0:
                        intensity = 0.05 + 0.45 * (val / max_val) if max_val > 0 else 0.5
                        styles.append(f'background-color: rgba(46, 204, 113, {intensity:.2f});')
                    else:
                        intensity = 0.05 + 0.45 * (val / min_val) if min_val < 0 else 0.5
                        styles.append(f'background-color: rgba(231, 76, 60, {intensity:.2f});')
                return styles

            # --- AI Analysis Integration Platform (Isolated Fragment) ---
            @st.fragment
            def render_ai_block(group_col, grouped_table):
                st.markdown("#### 🤖 AI Strategic Analysis")
                ai_state_key = f"ai_analysis_{group_col}"
                
                if ai_state_key in st.session_state:
                    st.info(st.session_state[ai_state_key]["text"], icon="🧠")
                    st.caption(f"Estimated Cost: ${st.session_state[ai_state_key]['cost']:.4f} (~{st.session_state[ai_state_key]['tokens']} input tokens)")
                    if st.button("Clear AI Analysis", key=f"btn_clear_ai_{group_col}"):
                        del st.session_state[ai_state_key]
                        st.rerun(scope="fragment")
                else:
                    if st.button("Generate Strategic Analysis", key=f"btn_ai_ahrefs_{group_col}"):
                        with st.spinner("Analyzing data with GPT-4o..."):
                            try:
                                md_table = grouped_table.to_csv(index=False)
                                prompt_path = "prompts/ahrefs_analysis_system.txt"
                                try:
                                    with open(prompt_path, "r", encoding="utf-8") as f:
                                        sys_prompt = f.read()
                                except FileNotFoundError:
                                    sys_prompt = "Analyze this Ahrefs Top Pages data table and provide strategic SEO insights."
                                    
                                date_context = ""
                                if st.session_state.get("ahrefs_comparison_dates"):
                                    date_context = f"\n\n{st.session_state.ahrefs_comparison_dates}\n\n"
                                    
                                user_prompt = f"Here is the data table:\n\n{md_table}"
                                in_tokens, est_cost = estimate_tokens_and_cost([sys_prompt, date_context, user_prompt])
                                
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system", "content": sys_prompt + date_context},
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000,
                                    stream=True
                                )
                                
                                analysis_text = st.write_stream(response)
                                st.session_state[ai_state_key] = {"text": analysis_text, "cost": est_cost, "tokens": in_tokens}
                                st.rerun(scope="fragment")
                            except Exception as e:
                                st.error(f"Error generating AI analysis: {e}")

            def render_ahrefs_view(df_view, group_col):
                if df_view.empty:
                    st.warning("No data matches the selected filters.")
                    return

                # Calculate metrics grouped by standard columns
                grouped = df_view.groupby(group_col).agg(
                    Total_URLs=("URL", "count"),
                    Total_Traffic_Change=("Traffic change", "sum"),
                    Total_Keywords_Change=("Keywords change", "sum"),
                    Avg_URL_Rating=("UR", "mean") # Adding UR averaging
                ).reset_index()
                
                # Sort by traffic change to find big winners/losers
                grouped = grouped.sort_values(by="Total_Traffic_Change", ascending=False)
                
                # Apply styling and limit decimals
                styled_grouped = grouped.style.format(precision=1).apply(style_change_gradient, subset=["Total_Traffic_Change", "Total_Keywords_Change"])
                
                # Display dataframe
                st.dataframe(styled_grouped, width="stretch", hide_index=True)
                
                # Render the isolated fragment block right here
                render_ai_block(group_col, grouped)
                
                st.markdown("##### Traffic Change Visualized")
                
                # Altair Chart for Traffic Change
                chart_traffic = alt.Chart(grouped).mark_bar().encode(
                    x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
                    y=alt.Y("Total_Traffic_Change:Q", title="Total Traffic Change"),
                    color=alt.condition(
                        alt.datum.Total_Traffic_Change > 0,
                        alt.value("#2ecc71"),  # Green for positive
                        alt.value("#e74c3c")   # Red for negative
                    ),
                    tooltip=[group_col, "Total_Traffic_Change", "Total_Keywords_Change", "Total_URLs", "Avg_URL_Rating"]
                ).properties(
                    height=500
                )
                
                st.altair_chart(chart_traffic, width="stretch")

                st.markdown("##### Keywords Change Visualized")

                # Altair Chart for Keyword Change
                chart_keywords = alt.Chart(grouped).mark_bar().encode(
                    x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
                    y=alt.Y("Total_Keywords_Change:Q", title="Total Keywords Change"),
                    color=alt.condition(
                        alt.datum.Total_Keywords_Change > 0,
                        alt.value("#2ecc71"),
                        alt.value("#e74c3c")
                    ),
                    tooltip=[group_col, "Total_Traffic_Change", "Total_Keywords_Change", "Total_URLs", "Avg_URL_Rating"]
                ).properties(
                    height=500
                )

                st.altair_chart(chart_keywords, width="stretch")

                st.markdown("##### Average URL Rating (UR)")

                # Altair Chart for Average UR
                chart_ur = alt.Chart(grouped).mark_bar(color="#f39c12").encode(
                    x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
                    y=alt.Y("Avg_URL_Rating:Q", title="Avg URL Rating"),
                    tooltip=[group_col, "Total_Traffic_Change", "Total_Keywords_Change", "Total_URLs", "Avg_URL_Rating"]
                ).properties(
                    height=500
                )

                st.altair_chart(chart_ur, width="stretch")
                
                # --- Detailed URL View ---
                st.markdown("---")
                st.markdown("##### Detailed URL View")
                # Sort the raw data for the table by Traffic Change
                df_detailed = df_view.sort_values(by="Traffic change", ascending=False).copy()
                
                # Increase rendering limits for large dataframes before applying style
                pd.set_option("styler.render.max_elements", 2000000)
                
                styled_detailed = df_detailed.style.format(precision=1).apply(style_change_gradient, subset=["Traffic change", "Keywords change"])
                st.dataframe(styled_detailed, width="stretch", hide_index=True)
                
                # We limit to top 50 rows for individual URL charts to avoid clutter, 
                # but if the user wants all of them, they are in the table.
                top_urls = df_detailed.head(50)
                if not top_urls.empty:
                    st.markdown("###### Top 50 URLs by Traffic Change Visualized")
                    
                    chart_url_traffic = alt.Chart(top_urls).mark_bar().encode(
                        x=alt.X("URL:N", sort="-y", title="URL", axis=alt.Axis(labelLimit=300)),
                        y=alt.Y("Traffic change:Q", title="Traffic Change"),
                        color=alt.condition(
                            alt.datum["Traffic change"] > 0,
                            alt.value("#2ecc71"),
                            alt.value("#e74c3c")
                        ),
                        tooltip=["URL", "Traffic change", "Keywords change", "UR", "Market", "Website Category"]
                    ).properties(height=400)
                    st.altair_chart(chart_url_traffic, width="stretch")

                    chart_url_keywords = alt.Chart(top_urls).mark_bar().encode(
                        x=alt.X("URL:N", sort="-y", title="URL", axis=alt.Axis(labelLimit=300)),
                        y=alt.Y("Keywords change:Q", title="Keywords Change"),
                        color=alt.condition(
                            alt.datum["Keywords change"] > 0,
                            alt.value("#2ecc71"),
                            alt.value("#e74c3c")
                        ),
                        tooltip=["URL", "Traffic change", "Keywords change", "UR", "Market", "Website Category"]
                    ).properties(height=400)
                    st.altair_chart(chart_url_keywords, width="stretch")

                    chart_url_ur = alt.Chart(top_urls).mark_bar(color="#f39c12").encode(
                        x=alt.X("URL:N", sort="-y", title="URL", axis=alt.Axis(labelLimit=300)),
                        y=alt.Y("UR:Q", title="URL Rating"),
                        tooltip=["URL", "Traffic change", "Keywords change", "UR", "Market", "Website Category"]
                    ).properties(height=400)
                    st.altair_chart(chart_url_ur, width="stretch")
                    
            with view_tab1:
                st.markdown("### Market Performance")
                c1, c2 = st.columns(2)
                
                available_markets = ["All"] + sorted([m for m in df["Market"].unique() if pd.notna(m) and str(m).strip() != ""])
                available_categories = ["All"] + sorted([c for c in df["Website Category"].unique() if pd.notna(c) and str(c).strip() != ""])
                
                with c1:
                    selected_market_filter = st.selectbox("Filter by Market", available_markets, key="filter_market_t1")
                with c2:
                    selected_cat_filter = st.selectbox("Filter by Category", available_categories, key="filter_cat_t1")
                
                filter_df1 = df.copy()
                if selected_market_filter != "All":
                    filter_df1 = filter_df1[filter_df1["Market"] == selected_market_filter]
                if selected_cat_filter != "All":
                    filter_df1 = filter_df1[filter_df1["Website Category"] == selected_cat_filter]

                render_ahrefs_view(filter_df1, "Market")
                
            with view_tab3:
                st.markdown("### Category Performance")
                c_c1, c_c2 = st.columns(2)
                
                with c_c1:
                    selected_market_filter_2 = st.selectbox("Filter by Market", available_markets, key="filter_market_t2")
                with c_c2:
                    selected_cat_filter_2 = st.selectbox("Filter by Category", available_categories, key="filter_cat_t2")

                filter_df2 = df.copy()
                if selected_market_filter_2 != "All":
                    filter_df2 = filter_df2[filter_df2["Market"] == selected_market_filter_2]
                if selected_cat_filter_2 != "All":
                    filter_df2 = filter_df2[filter_df2["Website Category"] == selected_cat_filter_2]

                render_ahrefs_view(filter_df2, "Website Category")
                
            # Offer download
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classified Data",
                data=csv_out,
                file_name="ahrefs_classified_data.csv",
                mime="text/csv",
            )

    with tabs[3]:
        st.markdown("#### GSC BigQuery Visualizer")
        st.info("💡 **Instructions:** Upload a Google Search Console (via BigQuery) export CSV to visualize clicks and impressions changes grouped by markets and categories.")
        
        # UI Elements
        gsc_file = st.file_uploader("Upload GSC BigQuery CSV", type=["csv"], key="gsc_file_uploader")
        
        if "gsc_processed" not in st.session_state:
            st.session_state.gsc_processed = False
            st.session_state.gsc_df = None
            
        if st.button("Clear GSC Data"):
            st.session_state.gsc_processed = False
            st.session_state.gsc_df = None
            st.rerun()
            
        if gsc_file:
            if not st.session_state.gsc_processed:
                try:
                    df_gsc = load_csv_robustly(gsc_file)
                    
                    if "Landing Page" not in df_gsc.columns:
                        st.error("The uploaded CSV does not contain a 'Landing Page' column. Please check the file.")
                    else:
                        with st.spinner("Classifying and processing URLs..."):
                            df_gsc["Landing Page"] = df_gsc["Landing Page"].astype(str)
                            
                            # Rename delta columns mapping to Pandas default naming duplicates
                            # Based on typical BigQuery exports
                            rename_map = {}
                            if "Δ" in df_gsc.columns:
                                rename_map["Δ"] = "Clicks Change"
                            if "Δ.1" in df_gsc.columns:
                                rename_map["Δ.1"] = "Impressions Change"
                            
                            if rename_map:
                                df_gsc.rename(columns=rename_map, inplace=True)
                            
                            # Clean up relevant columns to numeric (coerce errors to NaN, then fill with 0)
                            numeric_cols = ["clicks", "impressions", "Clicks Change", "Impressions Change"]
                            for col in numeric_cols:
                                if col in df_gsc.columns:
                                    df_gsc[col] = pd.to_numeric(df_gsc[col], errors='coerce').fillna(0)
                                else:
                                    df_gsc[col] = 0.0 # Create it if missing for some reason

                            # Apply classifications based on the Landing Page URL
                            df_gsc["Inferred Market (from URL)"] = df_gsc["Landing Page"].apply(backend.categorize_market)
                            df_gsc["Website Category"] = df_gsc["Landing Page"].apply(backend.categorize_website)
                            df_gsc["Inferred Language (from URL)"] = df_gsc["Landing Page"].apply(backend.categorize_language)
                            
                            # Keep Original Traffic Origin 
                            if "traffic origin" in df_gsc.columns:
                                df_gsc["Traffic Origin (Actual)"] = df_gsc["traffic origin"]
                            else:
                                df_gsc["Traffic Origin (Actual)"] = "Unknown"
                                
                            st.session_state.gsc_df = df_gsc
                            st.session_state.gsc_processed = True
                            
                            st.success("Successfully processed GSC BigQuery export!")
                except Exception as e:
                    st.error(f"Error processing the GSC CSV file: {e}")

        if st.session_state.gsc_processed and st.session_state.gsc_df is not None:
            df_gsc = st.session_state.gsc_df
            
            view_tab_gsc_market, view_tab_gsc_cat, view_tab_gsc_origin = st.tabs(["🌍 Market Performance", "📁 Category Performance", "📍 Traffic Origin Performance"])

            def style_change_gradient(s):
                s_num = pd.to_numeric(s, errors='coerce')
                max_val = s_num[s_num > 0].max() if (s_num > 0).any() else 0
                min_val = s_num[s_num < 0].min() if (s_num < 0).any() else 0
                
                styles = []
                for val in s_num:
                    if pd.isna(val) or val == 0:
                        styles.append('')
                    elif val > 0:
                        intensity = 0.05 + 0.45 * (val / max_val) if max_val > 0 else 0.5
                        styles.append(f'background-color: rgba(46, 204, 113, {intensity:.2f});')
                    else:
                        intensity = 0.05 + 0.45 * (val / min_val) if min_val < 0 else 0.5
                        styles.append(f'background-color: rgba(231, 76, 60, {intensity:.2f});')
                return styles

            def render_gsc_view(df_view, group_col):
                if df_view.empty:
                    st.warning("No data matches the selected filters.")
                    return

                # Calculate metrics grouped by standard columns
                grouped = df_view.groupby(group_col).agg(
                    Total_URLs=("Landing Page", "count"),
                    Total_Clicks=("clicks", "sum"),
                    Total_Impressions=("impressions", "sum"),
                    Clicks_Change=("Clicks Change", "sum"),
                    Impressions_Change=("Impressions Change", "sum")
                ).reset_index()
                
                # Sort by traffic change to find big winners/losers
                grouped = grouped.sort_values(by="Clicks_Change", ascending=False)
                
                # Apply styling and limit decimals
                styled_grouped = grouped.style.format(precision=1).apply(style_change_gradient, subset=["Clicks_Change", "Impressions_Change"])
                
                # Display dataframe
                st.dataframe(styled_grouped, width="stretch", hide_index=True)
                
                st.markdown("##### Clicks Change Visualized")
                
                # Altair Chart for Clicks Change
                chart_clicks = alt.Chart(grouped).mark_bar().encode(
                    x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
                    y=alt.Y("Clicks_Change:Q", title="Clicks Change"),
                    color=alt.condition(
                        alt.datum.Clicks_Change > 0,
                        alt.value("#2ecc71"),  # Green for positive
                        alt.value("#e74c3c")   # Red for negative
                    ),
                    tooltip=[group_col, "Clicks_Change", "Total_Clicks", "Impressions_Change", "Total_Impressions", "Total_URLs"]
                ).properties(
                    height=500
                )
                
                st.altair_chart(chart_clicks, width="stretch")

                st.markdown("##### Impressions Change Visualized")

                # Altair Chart for Impressions Change
                chart_impressions = alt.Chart(grouped).mark_bar().encode(
                    x=alt.X(f"{group_col}:N", sort="-y", title=group_col),
                    y=alt.Y("Impressions_Change:Q", title="Impressions Change"),
                    color=alt.condition(
                        alt.datum.Impressions_Change > 0,
                        alt.value("#2ecc71"),
                        alt.value("#e74c3c")
                    ),
                    tooltip=[group_col, "Clicks_Change", "Total_Clicks", "Impressions_Change", "Total_Impressions", "Total_URLs"]
                ).properties(
                    height=500
                )

                st.altair_chart(chart_impressions, width="stretch")
                
                # --- Detailed URL View ---
                st.markdown("---")
                st.markdown("##### Detailed URL View (Top 1000 URLs)")
                # Sort the raw data for the table by Clicks Change
                df_detailed = df_view.sort_values(by="Clicks Change", ascending=False).copy()
                
                # Limit the displayed data to top 1000 rows to avoid Pandas Styler cell limits
                df_detailed_display = df_detailed.head(1000)
                
                # Increase rendering limits for large dataframes before applying style
                pd.set_option("styler.render.max_elements", 2000000)
                
                styled_detailed = df_detailed_display.style.format(precision=1).apply(style_change_gradient, subset=["Clicks Change", "Impressions Change"])
                st.dataframe(styled_detailed, width="stretch", hide_index=True)
                
                # We limit to top 50 rows for individual URL charts to avoid clutter
                top_urls = df_detailed.head(50)
                if not top_urls.empty:
                    st.markdown("###### Top 50 URLs by Clicks Change Visualized")
                    
                    chart_url_clicks = alt.Chart(top_urls).mark_bar().encode(
                        x=alt.X("Landing Page:N", sort="-y", title="Landing Page", axis=alt.Axis(labelLimit=300)),
                        y=alt.Y("Clicks Change:Q", title="Clicks Change"),
                        color=alt.condition(
                            alt.datum["Clicks Change"] > 0,
                            alt.value("#2ecc71"),
                            alt.value("#e74c3c")
                        ),
                        tooltip=["Landing Page", "Clicks Change", "clicks", "Impressions Change", "impressions", "Inferred Market (from URL)", "Traffic Origin (Actual)", "Website Category"]
                    ).properties(height=400)
                    st.altair_chart(chart_url_clicks, width="stretch")

            # Shared filters across views for consistency
            gsc_markets = ["All"] + sorted([m for m in df_gsc["Inferred Market (from URL)"].unique() if pd.notna(m) and str(m).strip() != ""])
            gsc_categories = ["All"] + sorted([c for c in df_gsc["Website Category"].unique() if pd.notna(c) and str(c).strip() != ""])
            gsc_origins = ["All"] + sorted([o for o in df_gsc["Traffic Origin (Actual)"].unique() if pd.notna(o) and str(o).strip() != ""])

            with view_tab_gsc_market:
                st.markdown("### Market Performance (Inferred from URL)")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    filter_m1 = st.selectbox("Filter by Inferred Market", gsc_markets, key="gsc_m_m1")
                with c2:
                    filter_c1 = st.selectbox("Filter by Category", gsc_categories, key="gsc_c_m1")
                with c3:
                    filter_o1 = st.selectbox("Filter by Traffic Origin", gsc_origins, key="gsc_o_m1")
                
                f_df1 = df_gsc.copy()
                if filter_m1 != "All": f_df1 = f_df1[f_df1["Inferred Market (from URL)"] == filter_m1]
                if filter_c1 != "All": f_df1 = f_df1[f_df1["Website Category"] == filter_c1]
                if filter_o1 != "All": f_df1 = f_df1[f_df1["Traffic Origin (Actual)"] == filter_o1]

                render_gsc_view(f_df1, "Inferred Market (from URL)")
                
            with view_tab_gsc_cat:
                st.markdown("### Category Performance")
                c_c1, c_c2, c_c3 = st.columns(3)
                
                with c_c1:
                    filter_m2 = st.selectbox("Filter by Inferred Market", gsc_markets, key="gsc_m_c1")
                with c_c2:
                    filter_c2 = st.selectbox("Filter by Category", gsc_categories, key="gsc_c_c1")
                with c_c3:
                    filter_o2 = st.selectbox("Filter by Traffic Origin", gsc_origins, key="gsc_o_c1")

                f_df2 = df_gsc.copy()
                if filter_m2 != "All": f_df2 = f_df2[f_df2["Inferred Market (from URL)"] == filter_m2]
                if filter_c2 != "All": f_df2 = f_df2[f_df2["Website Category"] == filter_c2]
                if filter_o2 != "All": f_df2 = f_df2[f_df2["Traffic Origin (Actual)"] == filter_o2]

                render_gsc_view(f_df2, "Website Category")

            with view_tab_gsc_origin:
                st.markdown("### Traffic Origin Performance (Actual Visitor Location)")
                c_o1, c_o2, c_o3 = st.columns(3)
                
                with c_o1:
                    filter_m3 = st.selectbox("Filter by Inferred Market", gsc_markets, key="gsc_m_o1")
                with c_o2:
                    filter_c3 = st.selectbox("Filter by Category", gsc_categories, key="gsc_c_o1")
                with c_o3:
                    filter_o3 = st.selectbox("Filter by Traffic Origin", gsc_origins, key="gsc_o_o1")

                f_df3 = df_gsc.copy()
                if filter_m3 != "All": f_df3 = f_df3[f_df3["Inferred Market (from URL)"] == filter_m3]
                if filter_c3 != "All": f_df3 = f_df3[f_df3["Website Category"] == filter_c3]
                if filter_o3 != "All": f_df3 = f_df3[f_df3["Traffic Origin (Actual)"] == filter_o3]

                render_gsc_view(f_df3, "Traffic Origin (Actual)")
                
            # Offer download
            st.markdown("---")
            st.markdown("##### Export Data")
            csv_format_gsc = st.selectbox(
                "CSV Export Format", 
                ["Standard CSV (, separator, . decimal)", "EU Excel Ready (; separator, , decimal)"], 
                key="csv_format_gsc"
            )
            
            if "EU Excel" in csv_format_gsc:
                df_csv_gsc = df_gsc.copy()
                # Replace dots with commas for float columns
                for col in df_csv_gsc.select_dtypes(include=['float64', 'float32']).columns:
                    df_csv_gsc[col] = df_csv_gsc[col].apply(lambda x: str(x).replace('.', ','))
                csv_data_gsc = df_csv_gsc.to_csv(index=False, sep=';').encode('utf-8')
            else:
                csv_data_gsc = df_gsc.to_csv(index=False).encode('utf-8')
                
            st.download_button(
                label="📥 Download Classified GSC Data",
                data=csv_data_gsc,
                file_name="gsc_classified_data.csv",
                mime="text/csv",
            )


