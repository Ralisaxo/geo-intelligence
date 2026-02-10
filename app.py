import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import altair as alt
import geo_backend as backend
import auth

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
current_page = st.sidebar.radio("Navigation", ["Overview", "Knowledge Management", "Semantic Triples", "Reddit Analysis", "LLM Monitoring"])
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
    with st.tabs(["📊 Dashboard & Metrics"])[0]:
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
                    },
                    use_container_width=True
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
                            use_container_width=True,
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
                    use_container_width=True,
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
    tab5, tab7 = st.tabs(["📐 Semantic Alignment Lab", "☁️ AI Word Cloud"])

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
                calc_btn = st.button("Calculate Match", type="primary", use_container_width=True)

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
                        use_container_width=True,
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

                            st.altair_chart(chart_final.interactive(), use_container_width=True)

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
                        use_container_width=True,
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

                if b1.button("Inside-Out", use_container_width=True, help="Adjectives for specific brand"):
                    st.session_state['prompt_mode'] = 'inside_out'
                    st.session_state['prompts_content'] = "\n".join(QUESTIONS_INSIDE_OUT)
                    st.rerun()

                if b2.button("Outside-In", use_container_width=True, help="Category questions"):
                    st.session_state['prompt_mode'] = 'outside_in'
                    st.session_state['prompts_content'] = "\n".join(QUESTIONS_OUTSIDE_IN)
                    st.rerun()

                if b3.button("Custom", use_container_width=True):
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

                    # Estimates
                    est_input_tokens = total_requests * 30 
                    est_output_tokens = total_requests * 5 
                    est_cost = (est_input_tokens / 1_000_000 * 5.00) + (est_output_tokens / 1_000_000 * 15.00)

                    # Vertical Cards
                    with st.container(border=True):
                        st.metric("Requests", total_requests)
                    with st.container(border=True):
                        st.metric("Est. Cost", f"${est_cost:.4f}")
                    with st.container(border=True):
                        st.metric("Est. Tokens", f"{est_input_tokens + est_output_tokens}")


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
                            df_sources.insert(0, "Select", False)
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
                                st.dataframe(prompt_data, use_container_width=True)
                            else:
                                st.write("No ID details available.")

                # Display Results if available
                if st.session_state['accuranker_data'] is not None:
                     # Create a container for the button ABOVE the table
                     toggles_container = st.container()

                     edited_df = st.data_editor(
                         st.session_state['accuranker_data'],
                         column_config={
                             "Select": st.column_config.CheckboxColumn("Analyze?", width="small"),
                             "Title": st.column_config.TextColumn("Title", width="medium"),
                             "Prompts": st.column_config.NumberColumn("Prompts", help="Number of prompts where this thread appeared"),
                             "Cited %": st.column_config.NumberColumn("Cited %", format="%.1f%%", help="Percentage of commercial prompts citing this thread"),
                             "URL": st.column_config.LinkColumn("Thread URL", width="large")
                         },
                         use_container_width=True,
                         hide_index=True
                     )

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
                             if st.button("Select All", use_container_width=True):
                                 st.session_state['accuranker_data']['Select'] = True
                                 st.rerun()

                         with col3:
                             if st.button("Deselect All", use_container_width=True):
                                 st.session_state['accuranker_data']['Select'] = False
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
                        use_container_width=True,
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

# --- PAGE: LLM MONITORING ---
elif current_page == "LLM Monitoring":
    st.markdown("## 🛡️ LLM Truth Control")
    st.info("Verify if AI search results verify the 'Ground Truth' defined in AccuRanker.")

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
        if st.button("Fetch Prompts", type="primary", use_container_width=False, help="Fetch latest prompts from AccuRanker"):
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
                if st.button("Select All", use_container_width=True):
                    st.session_state.selection_df["Select"] = True
                    st.rerun()
            with c_sel2:
                if st.button("Unselect All", use_container_width=True):
                    st.session_state.selection_df["Select"] = False
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
                use_container_width=True,
                key="prompt_selector" # Unique key
            )
            
            # Sync edits back to session state so they persist across reruns
            st.session_state.selection_df = edited_df
            
            # Filter based on selection
            selected_indices = edited_df[edited_df["Select"]]["TaskID"].tolist()
            tasks_to_verify = [tasks[i] for i in selected_indices]
            
            st.caption(f"Selected {len(tasks_to_verify)} out of {len(tasks)} items.")
            
            # Cost Update based on selection
            est_cost = len(tasks_to_verify) * 0.002
            st.caption(f"💰 Est. Verify Cost: ~${est_cost:.3f}")

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
                    use_container_width=True,
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
                                         st.dataframe(sources, hide_index=True, use_container_width=True)
        
                                # Analysis Footer (Always visible)
                                if item.get('Reason'):
                                     st.markdown(f"**Analysis:** {item.get('Reason')}")
