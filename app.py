import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import geo_backend as backend
import auth

# -----------------------------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------------------------
auth.check_password()

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GEO Command Center", layout="wide", page_icon="🌍")

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
        color: #0f52ba; /* Saxo Blue-ish */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #666;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
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
st.title("🌍 GEO Command Center")
st.markdown("### Authority Gap & Brand Intelligence Dashboard")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
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
if st.session_state.df_final is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard & Metrics", "🔎 Data Explorer", "🤖 AI Strategist", "🌐 Google KG Data", "🔮 RAG Simulation"])
    
    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.markdown("#### Higher Ground Overview")
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

    # --- TAB 2: DATA EXPLORER ---
    with tab2:
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

    # --- TAB 3: AI STRATEGIST ---
    with tab3:
        st.markdown("#### 🤖 AI GEO Gap Analysis")
        st.info("The AI analyzes the data to find semantic gaps between Saxo Bank and competitors in the Knowledge Graph context.")
        
        if st.button("🧠 Run AI Analysis", type="primary"):
            with st.spinner("Analyzing semantics and authority signals..."):
                analysis_text = backend.run_geo_analysis(st.session_state.df_final, client)
                st.divider()
                st.markdown(analysis_text)

    # --- TAB 4: GOOGLE KNOWLEDGE GRAPH ---
    with tab4:
        st.markdown("#### 🌐 Google Knowledge Graph Data Only")
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

    # --- TAB 5: RAG SIMULATION ---
    with tab5:
        st.markdown("#### 🔮 Reality Check: How AI Sees Us")
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
            st.code(backend.load_prompt_file("rag_prompt.txt"))

else:
    # Empty State
    st.info("👈 Please select competitors and click 'Load from Cache' or 'Fetch Data' to begin.")
