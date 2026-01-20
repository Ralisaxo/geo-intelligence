import pandas as pd
import requests
import json
import time
import urllib.parse
from openai import OpenAI
import sqlite3
from datetime import datetime
import os

DB_FILE = "geo_cache.db"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "SaxoBankGEOIntelligenceTool/1.0 (contact@example.com)"}

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS
# -----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS competitor_data (
            q_id TEXT PRIMARY KEY,
            data_json TEXT,
            last_fetched DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_data(q_ids):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(q_ids))
    query = f"SELECT q_id, data_json FROM competitor_data WHERE q_id IN ({placeholders})"
    c.execute(query, q_ids)
    results = c.fetchall()
    conn.close()
    
    cached_data = []
    found_ids = set()
    for q_id, data_json in results:
        try:
            data = json.loads(data_json)
            cached_data.append(data)
            found_ids.add(q_id)
        except json.JSONDecodeError:
            continue
            
    return cached_data, found_ids

def save_to_cache(df):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now().isoformat()
    # Convert dataframe results to list of dicts for storage
    # Handle NaN values by replacing with None before serialization ensures valid JSON
    records = df.where(pd.notnull(df), None).to_dict(orient='records')
    
    for record in records:
        q_id = record.get('qid')
        if q_id:
            data_json = json.dumps(record)
            c.execute('''
                INSERT OR REPLACE INTO competitor_data (q_id, data_json, last_fetched)
                VALUES (?, ?, ?)
            ''', (q_id, data_json, now))
            
    conn.commit()
    conn.close()

def check_all_cache():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Ensure table exists just in case, though init_db runs at startup
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='competitor_data'")
    if not c.fetchone():
        return set()
        
    c.execute("SELECT q_id FROM competitor_data")
    cached_ids = {row[0] for row in c.fetchall()}
    conn.close()
    return cached_ids

# -----------------------------------------------------------------------------
# WIKIDATA FUNCTIONS
# -----------------------------------------------------------------------------
def fetch_wikidata_entities(id_list, progress_callback=None):
    all_entities = {}
    BATCH_SIZE = 50
    total_ids = len(id_list)
    
    for i in range(0, total_ids, BATCH_SIZE):
        batch = id_list[i:i + BATCH_SIZE]
        ids_string = "|".join(batch)
        
        params = {
            "action": "wbgetentities",
            "ids": ids_string,
            "format": "json"
        }
        
        try:
            response = requests.get(WIKIDATA_API_URL, params=params, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            
            if "entities" in data:
                all_entities.update(data["entities"])
            
            if progress_callback:
                progress = min((i + BATCH_SIZE) / total_ids, 1.0)
                progress_callback(progress, f"Fetched {len(all_entities)} of {total_ids} entities from Wikidata")
                
            time.sleep(1) # Gentle rate limiting
            
        except Exception as e:
            # We can print error or let it fail, keeping it silent for batch errors to not crash all
            print(f"Error fetching Wikidata batch: {e}")
            
    return all_entities

def fetch_all_labels(id_list):
    translation_map = {}
    BATCH_SIZE = 50
    
    for i in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[i:i+BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
            "format": "json"
        }
        try:
            res = requests.get(WIKIDATA_API_URL, params=params, headers=HEADERS).json()
            entities = res.get("entities", {})
            for target_id, d in entities.items():
                label = d.get("labels", {}).get("en", {}).get("value", target_id)
                translation_map[target_id] = label
            time.sleep(0.5)
        except Exception:
            pass # Fail silently for labels
            
    return translation_map

def get_readable_value(claim_list, label_map, p_id=None):
    parts = []
    for claim in claim_list:
        try:
            mainsnak = claim.get('mainsnak', {})
            dv = mainsnak.get('datavalue', {})
            val = dv.get('value')
            v_type = dv.get('type')

            if v_type == "wikibase-entityid":
                qid = val.get('id')
                parts.append(f"{label_map.get(qid, qid)} ({qid})")
            elif v_type == "time":
                parts.append(val.get('time'))
            elif v_type == "quantity":
                parts.append(val.get('amount'))
            elif isinstance(val, str):
                # Normalization for Google KG ID (P2671)
                if p_id == "P2671":
                    val = clean_kg_id(val)
                parts.append(val)
            elif isinstance(val, dict) and 'text' in val:
                parts.append(val['text'])
        except: continue
    return " | ".join(list(dict.fromkeys(parts)))

def process_wikidata_data(full_data, label_progress_callback=None):
    all_p_ids = set()
    all_referenced_q_ids = set()

    # Discovery
    for qid in full_data:
        claims = full_data[qid].get("claims", {})
        for p_id, claim_list in claims.items():
            all_p_ids.add(p_id)
            for claim in claim_list:
                try:
                    val = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {})
                    if isinstance(val, dict) and 'id' in val and str(val['id']).startswith('Q'):
                        all_referenced_q_ids.add(val['id'])
                except: continue

    all_referenced_q_ids.update(full_data.keys())
    all_ids_to_translate = list(all_p_ids | all_referenced_q_ids)
    
    # Translation
    if label_progress_callback:
        label_progress_callback("Translating Property IDs and Values...")
    label_map = fetch_all_labels(all_ids_to_translate)

    # Extraction
    rows = []
    for qid, entity in full_data.items():
        row = {
            "qid": qid,
            "label_en": entity.get("labels", {}).get("en", {}).get("value", ""),
            "description_en": entity.get("descriptions", {}).get("en", {}).get("value", ""),
        }
        
        # Calculate Authority Metrics
        total_claims_count = 0
        total_references_count = 0
        
        claims = entity.get("claims", {})
        
        # We iterate over ALL claims to calculate the holistic score
        for p_id, claim_list in claims.items():
            total_claims_count += len(claim_list)
            for claim in claim_list:
                refs = claim.get('references', [])
                total_references_count += len(refs)
                
        # Metric: Average References per Claim
        if total_claims_count > 0:
            auth_score = total_references_count / total_claims_count
        else:
            auth_score = 0.0
            
        row["Source_Authority_Score"] = auth_score
        row["Total_Claims"] = total_claims_count
        row["Total_References"] = total_references_count
        
        # Populate columns
        for p_id in all_p_ids:
            header = f"{label_map.get(p_id, p_id)} ({p_id})"
            if p_id in claims:
                row[header] = get_readable_value(claims[p_id], label_map, p_id)
            else:
                row[header] = None
        rows.append(row)
        
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# GOOGLE KNOWLEDGE GRAPH FUNCTIONS
# -----------------------------------------------------------------------------
def clean_kg_id(raw_id):
    if not raw_id or not isinstance(raw_id, str):
        return None
    clean_id = raw_id.strip()
    if clean_id.startswith("kg:"):
        clean_id = clean_id[3:]
    if not clean_id.startswith("/"):
        return None
    return clean_id

def search_knowledge_graph(query, api_key, kg_id=None):
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'indent': True,
        'key': api_key,
        'limit': 1
    }

    if kg_id:
        params['ids'] = kg_id
    else:
        params['query'] = query
        
    url = service_url + '?' + urllib.parse.urlencode(params)
    
    try:
        response = requests.get(url)
        if response.status_code == 404:
             return {
                "KG_Name": None,
                "KG_Score": 0,
                "KG_Description": None,
                "KG_Types": None,
                "KG_Result_ID": None,
                "KG_Detailed_Bio": None,
                "KG_Wiki_URL": None,
                "KG_License": None,
                "KG_URL": None,
                "KG_Image_URL": None,
                "KG_Image_Source": None,
                "Error": "404 Not Found"
            }
            
        response.raise_for_status()
        data = json.loads(response.text)
        
        if 'itemListElement' in data and len(data['itemListElement']) > 0:
            result = data['itemListElement'][0].get('result', {})
            score = data['itemListElement'][0].get('resultScore', 0)
            
            detailed_obj = result.get('detailedDescription', {})
            image_obj = result.get('image', {})
            
            return {
                "KG_Name": result.get('name'),
                "KG_Score": score,
                "KG_Description": result.get('description'),
                "KG_Types": ", ".join(result.get('@type', [])) if isinstance(result.get('@type'), list) else result.get('@type'),
                "KG_Result_ID": result.get('@id'),
                "KG_Detailed_Bio": detailed_obj.get('articleBody'),
                "KG_Wiki_URL": detailed_obj.get('url'),
                "KG_License": detailed_obj.get('license'),
                "KG_URL": result.get('url'),
                "KG_Image_URL": image_obj.get('contentUrl'),
                "KG_Image_Source": image_obj.get('url')
            }
        else:
             return {
                "KG_Name": None,
                "KG_Score": 0,
                "KG_Description": None,
                "KG_Types": None,
                "KG_Result_ID": None,
                "KG_Detailed_Bio": None,
                "KG_Wiki_URL": None,
                "KG_License": None,
                "KG_URL": None,
                "KG_Image_URL": None,
                "KG_Image_Source": None
            }
            
    except Exception as e:
        return {"Error": str(e)}

# -----------------------------------------------------------------------------
# AI ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------
def run_geo_analysis(df, client):
    saxo_qid = "Q1325291"
    saxo_row = df[df['qid'] == saxo_qid]
    
    if saxo_row.empty:
        # Just grab top competitors if Saxo isn't there
        top_competitors = df.sort_values(by="KG_Score", ascending=False).head(4)
    else:
        filetered_df = df[df['qid'] != saxo_qid]
        top_competitors = filetered_df.sort_values(by="KG_Score", ascending=False).head(3)
        top_competitors = pd.concat([saxo_row, top_competitors])
        
    csv_data = top_competitors.to_csv(index=False)
    
    # Load system prompt from file
    try:
        with open("prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "You are a GEO Expert. Compare Saxo Bank to competitors."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the data for Saxo Bank and top competitors:\n\n{csv_data}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error running AI analysis: {e}"

def load_prompt_file(filename):
    """Safely load a text file content."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: {filename} not found."

def generate_rag_bio(df, company_name, source_mode, client):
    """
    Generates a RAG-based bio for a specific company using restricted data context.
    """
    # Filter for the specific company
    # We try to match by 'label_en' (Name)
    company_row = df[df['label_en'] == company_name]
    
    if company_row.empty:
        return f"Error: Company '{company_name}' not found in the current loaded data."
    
    # Extract the single row as a dictionary
    full_data = company_row.iloc[0].to_dict()
    
    # Construct Context based on Mode
    context_data = {}
    
    if source_mode == "Wikidata Only":
        # Exclude keys starting with KG_
        context_data = {k: v for k, v in full_data.items() if not k.startswith("KG_")}
        
    elif source_mode == "Knowledge Graph Only":
        # Include ONLY keys starting with KG_ or basic identifiers
        context_data = {k: v for k, v in full_data.items() if k.startswith("KG_") or k in ['qid', 'label_en']}
        
    else: # Combined
        context_data = full_data  # Send everything
        
    context_json = json.dumps(context_data, indent=2, default=str)
    
    # Load RAG Prompt
    system_instruction = load_prompt_file("rag_prompt.txt")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context: {context_json} \n\n Task: Write a professional bio for {company_name}. Stick strictly to the context."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating bio: {e}"
