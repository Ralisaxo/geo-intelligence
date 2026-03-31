import pandas as pd
import praw
import requests
import json
import time
import urllib.parse
from openai import OpenAI
import sqlite3
from datetime import datetime
import os
import re
import io
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from google import genai
from google.genai import types
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
from bs4 import BeautifulSoup

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
def run_geo_analysis(df, client, model="gpt-4o"):
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
    system_prompt = load_prompt_file("prompts/geo_analysis_system.txt")
    if "Error:" in system_prompt:
         system_prompt = "You are a GEO Expert. Compare Saxo Bank to competitors."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the data for Saxo Bank and top competitors:\n\n{csv_data}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error running AI analysis: {e}"

def run_flexsheet_prompt(prompt_text, client, model="gpt-4o", is_json=False):
    """
    Executes a synthesized prompt for AI FlexSheet without a system prompt.
    """
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

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
    system_instruction = load_prompt_file("prompts/rag_bio_system.txt")
    
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

def get_semantic_similarity(text1, text2, api_key):
    """
    Computes semantic similarity between two texts using Gemini embeddings.
    Returns: (Score (0-100 float), vector1, vector2)
    """
    if not api_key:
        return 0.0, None, None

    client = genai.Client(api_key=api_key)
    
    try:
        # Get embeddings
        # New SDK returns an object with .embeddings attribute which is a list
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text1, text2],
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY"
            )
        )
        
        # Accessing embeddings from the response
        # Accessing embeddings from the response
        if not response.embeddings or len(response.embeddings) < 2:
             # Fallback if batch fails for some reason or returns fewer
             v1_resp = client.models.embed_content(
                 model="models/gemini-embedding-001", 
                 contents=text1,
                 config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
             )
             v2_resp = client.models.embed_content(
                 model="models/gemini-embedding-001", 
                 contents=text2,
                 config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
             )
             v1 = v1_resp.embeddings[0].values
             v2 = v2_resp.embeddings[0].values
        else:
             v1 = response.embeddings[0].values
             v2 = response.embeddings[1].values

        # Convert to numpy arrays
        vec1 = np.array(v1)
        vec2 = np.array(v2)

        # Cosine Similarity
        dot_product = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0, vec1, vec2

        similarity = dot_product / (norm_v1 * norm_v2)
        # Return Raw Cosine Similarity (0.0 - 1.0)
        return float(similarity), vec1, vec2

    except Exception as e:
        print(f"Error in semantic similarity: {e}")
        return 0.0, None, str(e)

def calculate_display_score(raw_score):
    """
    Normalizes raw cosine similarity (typically 0.35-0.65) to a 0-100% human-readable scale.
    """
    # Anchor points based on data analysis
    # Anchor points based on data analysis
    min_anchor = 0.81  # Scores below this become 0%
    max_anchor = 0.89  # Scores above this become 100%

    # Linear normalization
    if raw_score <= min_anchor:
        return 0.0
    if raw_score >= max_anchor:
        return 100.0

    return ((raw_score - min_anchor) / (max_anchor - min_anchor)) * 100

def batch_sentiment_analysis(phrases, api_key):
    """
    Classifies sentiment for a list of phrases using Gemini Flash.
    Returns: Dict {phrase: "Positive"/"Negative"/"Neutral"}
    """
    if not phrases or not api_key:
        return {}

    client = genai.Client(api_key=api_key)
    # Use specific version requested
    model_name = 'gemini-2.5-flash-lite' 
    
    prompt_template = load_prompt_file("prompts/sentiment_analysis.txt")
    if "Error:" in prompt_template:
        # Fallback if file missing
        prompt = f"""
        You are a Brand Reputation Analyst for an investment bank. 
        Classify the sentiment of the following adjectives strictly as 'Positive', 'Negative', or 'Neutral' in the context of a financial trading platform. 
        Input: {json.dumps(phrases)}
        Output JSON: {{phrase: sentiment}}
        """
    else:
        # Use format to inject the JSON
        # Note: The text file uses {input_json} as placeholder
        try:
             prompt = prompt_template.format(input_json=json.dumps(phrases))
        except Exception as e:
             print(f"Error formatting prompt: {e}")
             return {p: "Neutral" for p in phrases}
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        text = response.text.strip()
        # Clean potential markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        # LOGGING FOR DEBUGGING
        try:
            with open("sentiment_debug.txt", "w", encoding="utf-8") as f:
                f.write(f"Prompt Input: {json.dumps(phrases)}\n")
                f.write(f"Raw Response: {text}\n")
        except: pass

        try:
            raw_result = json.loads(text)
        except json.JSONDecodeError:
             # Try to find JSON in text if extra text exists
             import re
             json_match = re.search(r'\{.*\}', text, re.DOTALL) or re.search(r'\[.*\]', text, re.DOTALL)
             if json_match:
                 raw_result = json.loads(json_match.group())
             else:
                 raise ValueError("No JSON found")

        # Parsing Logic
        lookup_map = {}
        
        # Case 1: Dict {phrase: sentiment}
        if isinstance(raw_result, dict):
            lookup_map = {str(k).lower(): v for k, v in raw_result.items()}
            
        # Case 2: List of Dicts [{phrase: "...", sentiment: "..."}, ...] or [{"phrase": "...", "sentiment": "..."}]
        elif isinstance(raw_result, list):
            for item in raw_result:
                if isinstance(item, dict):
                    # Try to find values generically
                    vals = list(item.values())
                    if len(vals) >= 2:
                        # Assumption: One is the phrase, one is the sentiment
                        # Heuristic: Phrase is in input list (lower)
                        k = None
                        v = None
                        for val in vals:
                            if str(val).lower() in [p.lower() for p in phrases]:
                                k = val
                            elif str(val).title() in ["Positive", "Negative", "Neutral"]:
                                v = val
                        
                        if k and v:
                            lookup_map[str(k).lower()] = v
        
        normalized = {}
        for phrase in phrases:
            p_lower = phrase.lower()
            val = "Neutral"
            
            if p_lower in lookup_map:
                val = str(lookup_map[p_lower]).title()
            
            if val not in ["Positive", "Negative", "Neutral"]:
                 val = "Neutral"
            
            normalized[phrase] = val
             
        return normalized

    except Exception as e:
        with open("sentiment_debug_error.txt", "w") as f:
            f.write(f"Error: {e}")
        print(f"Error in batch sentiment: {e}")
        return {p: "Neutral" for p in phrases}

def reduce_dimensions(vectors_list):
    """
    Reduces list of vectors to 2D using PCA.
    Returns: List of [x, y] coordinates.
    """
    if len(vectors_list) < 2:
        return []
    
    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors_list)
        return reduced.tolist()
    except Exception as e:
        print(f"PCA Error: {e}")
        return []

# -----------------------------------------------------------------------------
# WORD CLOUD ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def normalize_brand(text):
    mapping = {
        "saxo bank": "saxo",
        "ig group": "ig",
        "interactive brokers": "ibkr",
        "degiro": "degiro",
    }
    return mapping.get(text, text)

def generate_brand_analysis(prompts_list, models_config, iterations, api_keys, progress_callback=None):
    """
    Runs the brand analysis using OpenAI and Gemini models.
    
    Args:
        prompts_list (list): List of text prompts.
        models_config (dict): {'openai': ['gpt-4o', ...], 'gemini': ['gemini-pro', ...]}
        iterations (int): Number of times to run each prompt.
        api_keys (dict): {'openai': '...', 'gemini': '...'}
        progress_callback (func): Optional callback(progress_float, status_text)
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    all_results = []
    selected_openai_models = models_config.get('openai', [])
    selected_gemini_models = models_config.get('gemini', [])
    
    openai_key = api_keys.get('openai')
    gemini_key = api_keys.get('gemini')
    
    total_steps = len(prompts_list) * (len(selected_openai_models) + len(selected_gemini_models))
    step_count = 0
    
    # --- OpenAI Loop ---
    if openai_key and selected_openai_models:
        # We import OpenAI here or pass client? Better to create client here or reuse
        # Since this might be long running, new client is fine.
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        for model in selected_openai_models:
            for prompt in prompts_list:
                if progress_callback:
                    progress_callback(min(step_count / total_steps, 1.0), f"Running OpenAI {model}: {prompt[:40]}...")
                
                # Determine clean question text
                if "Sentence:" in prompt:
                    question_text = prompt.split("Sentence:")[-1].strip()
                else:
                    question_text = prompt

                is_reasoning = "o1-" in model or "gpt-5" in model
                
                responses = []
                for _ in range(iterations):
                    try:
                        if is_reasoning:
                            completion = client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": prompt}],
                                max_completion_tokens=500
                            )
                            responses.append(completion.choices[0].message.content.strip())
                        else:
                            completion = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": load_prompt_file("prompts/openai_default_system.txt")},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=1.0,
                                max_tokens=15
                            )
                            responses.append(completion.choices[0].message.content.strip())
                    except Exception as e:
                        responses.append(f"Error: {str(e)}")
                        time.sleep(1)
                
                for ans in responses:
                    all_results.append({
                        "Prompt": prompt,
                        "Question": question_text,
                        "Model": f"OpenAI - {model}",
                        "Answer": ans
                    })
                
                step_count += 1

    # --- Gemini Loop ---
    if gemini_key and selected_gemini_models:
        client = genai.Client(api_key=gemini_key)
        
        # New SDK Safety Settings
        # They are passed as part of GenerateContentConfig
        # https://github.com/googleapis/python-genai
        
        # Mapping HarmCategory to new types if needed, but the types are available in google.genai.types
        # Actually in new SDK, it is cleaner.
        
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]
        
        for model_name in selected_gemini_models:
            # model_instance = genai.GenerativeModel(model_name) # OLD
            for prompt in prompts_list:
                if progress_callback:
                    progress_callback(min(step_count / total_steps, 1.0), f"Running Gemini {model_name}: {prompt[:40]}...")
                
                if "Sentence:" in prompt:
                    question_text = prompt.split("Sentence:")[-1].strip()
                else:
                    question_text = prompt
                
                responses = []
                for i in range(iterations):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                candidate_count=1,
                                max_output_tokens=100,
                                temperature=1.0,
                                safety_settings=safety_settings
                            )
                        )
                        # New response structure
                        if response.candidates and response.candidates[0].content.parts:
                             responses.append(response.candidates[0].content.parts[0].text.strip())
                        elif response.text:
                             responses.append(response.text.strip())
                        else:
                             responses.append("")

                    except Exception as e:
                        responses.append(f"Error: {str(e)}")
                        time.sleep(1)

                for ans in responses:
                    all_results.append({
                        "Prompt": prompt,
                        "Question": question_text,
                        "Model": f"Gemini - {model_name}",
                        "Answer": ans
                    })
                
                step_count += 1
    
    if progress_callback:
        progress_callback(1.0, "Analysis Complete!")
        
    return pd.DataFrame(all_results)

def generate_wordclouds(df):
    """
    Generates word cloud images for each question and model.
    Returns a dictionary: { (question_text, model_name): image_bytes }
    """
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "output", "exactly", "one", "brand", "name", "do", "not", "more", "than", "only",
        "is", "a", "an", "the", "for", "in", "of", "question", "sentence", "complete", "error"
    ])
    
    generated_images = {}
    unique_questions = df['Question'].unique()
    
    for question in unique_questions:
        models_for_q = df[df['Question'] == question]['Model'].unique()
        
        for model in models_for_q:
            subset = df[(df['Question'] == question) & (df['Model'] == model)]
            valid_answers = []
            for ans in subset['Answer']:
                c = clean_text(ans)
                n = normalize_brand(c)
                if n and n not in custom_stopwords:
                    valid_answers.append(n)
            
            if valid_answers:
                phrase_counts = Counter(valid_answers)
                if phrase_counts:
                    try:
                        wc = WordCloud(
                            width=1200, height=600, background_color='white',
                            stopwords=custom_stopwords, max_words=30, normalize_plurals=False
                        ).generate_from_frequencies(phrase_counts)
                    except ValueError as e:
                        print(f"Skipping wordcloud for {model} due to error: {e}")
                        continue
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(f"{model}", fontsize=10)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    generated_images[(question, model)] = buf.getvalue()
                    plt.close(fig)
                    
    return generated_images

# -----------------------------------------------------------------------------
# REDDIT INTEL FUNCTIONS
# -----------------------------------------------------------------------------

def get_reddit_client(secrets):
    """
    Initializes PRAW Reddit client from secrets.
    """
    try:
        if "reddit" not in secrets:
            return None
        
        r_creds = secrets["reddit"]
        reddit = praw.Reddit(
            client_id=r_creds["client_id"],
            client_secret=r_creds["client_secret"],
            user_agent=r_creds["user_agent"]
        )
        # Quick check if read-only mode works (doesn't auth full user, just app)
        return reddit
    except Exception as e:
        print(f"Error initializing Reddit client: {e}")
        return None

def fetch_reddit_data(url, reddit_client):
    """
    Fetches thread title, body, and top comments.
    Returns: dict with details or error.
    """
    try:
        submission = reddit_client.submission(url=url)
        
        # Trigger fetch
        _ = submission.title
        
        # Flatten comments (top level only or flattened tree? User asked for Top 20)
        submission.comments.replace_more(limit=0)
        top_comments = submission.comments.list()[:20]
        
        comments_text = []
        for c in top_comments:
            comments_text.append(f"- {c.body}")
            
        full_context = f"TITLE: {submission.title}\n\nBODY: {submission.selftext}\n\nTOP COMMENTS:\n" + "\n".join(comments_text)
        
        return {
            "title": submission.title,
            "subreddit": submission.subreddit.display_name,
            "url": url,
            "context": full_context,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "error": None
        }
    except Exception as e:
        return {
            "title": "Error",
            "url": url,
            "context": "",
            "error": str(e)
        }

def analyze_reddit_sentiment(context_text, client, target_brand="Saxo Bank", model="gpt-4o"):
    """
    Analyzes sentiment specifically for {target_brand} using OpenAI.
    """
    system_prompt = f"""You are a Brand Reputation Analyst. Analyze this Reddit thread context. Your specific focus is {target_brand}.
    
    1. Sentiment: Determine the sentiment specifically towards {target_brand} (Positive, Negative, Neutral, Mixed). If {target_brand} is not mentioned, return 'Irrelevant'.
    2. Summary: Write a 2-3 sentence summary of why the sentiment is such. Quote specific user complaints or praises if found.
    
    Output strictly as JSON: {{"sentiment": "...", "summary": "..."}}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this thread:\n\n{context_text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        return {"sentiment": "Error", "summary": str(e)}

def generate_general_market_sentiment(analyzed_threads_data, client, model="gpt-4o"):
    """
    Generates a general market sentiment report based on a list of analyzed Reddit threads.
    """
    prompt_template = load_prompt_file("prompts/general_market_sentiment.txt")
    
    # Format the data for the LLM
    import json
    formatted_data = json.dumps(analyzed_threads_data, indent=2)
    
    system_prompt = prompt_template.replace("{input_data}", formatted_data)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please generate the market sentiment report based on the provided data."}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating market sentiment: {str(e)}"

def search_reddit(query, sort_by="relevance", time_filter="year", limit=5, reddit_client=None):
    """
    Searches Reddit for threads matching the query.
    Returns: List of URLs.
    """
    if not reddit_client or not query:
        return []
        
    try:
        # Search all subreddits
        # synxtax: subreddit("all").search(query, sort=..., time_filter=..., limit=...)
        search_results = reddit_client.subreddit("all").search(
            query, 
            sort=sort_by, 
            time_filter=time_filter, 
            limit=limit
        )
        
        urls = []
        for submission in search_results:
            urls.append(f"https://www.reddit.com{submission.permalink}")
            
        return urls
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def extract_thread_id(url):
    """
    Extracts the Reddit Thread ID from a URL using regex.
    Handles formats like:
    - .../comments/1ias4s0/title/
    - .../comments/1ias4s0/
    - .../comments/1ias4s0/title/?query=param
    """
    if not url:
        return None
    # Look for /comments/ followed by alphanumeric ID
    match = re.search(r'/comments/([a-z0-9]+)', url)
    if match:
        return match.group(1)
    return None

def fetch_accuranker_prompts_raw(brand_id, api_token):
    """
    Fetches all prompts from AccuRanker for a given brand.
    Returns: List of prompt dictionaries (raw data).
    """
    if not brand_id or not api_token:
        return []
        
    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    # Request fields needed for both tag extraction and processing
    params = {
        "fields": "id,prompt,tags,results.created_at,results.prompt_response,results.sources.url,results.sources.title,results.sources.rank",
        "limit": 1000
    }
    
    prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract results
            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None # simple list means no pagination usually
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next') # Update URL for next page
                # Params are part of the 'next' URL usually, so we reset params to avoid duplication or conflict
                # However, usually 'next' includes everything.
                if url:
                   params = None 
            
            prompts.extend(chunk)
            
            # Safety break
            if len(prompts) > 10000:
                print("Hit safety limit of 10000 prompts")
                break
                
        return prompts

    except Exception as e:
        print(f"AccuRanker API Error: {e}")
        return prompts

def process_accuranker_prompts_for_reddit(prompts, tag_filter="Commercial"):
    """
    Processes a list of AccuRanker prompts.
    1. Filters by tag (case-insensitive).
    2. Aggregates Reddit sources by Thread ID.
    
    Returns: 
        - curated_list: List of dicts for the table [{'url': ..., 'count': ...}]
        - relevant_prompts: List of prompt objects that matched the filter (for display)
    """
    # Store aggregated counts: {thread_id: {'count': 0, 'slug': None}}
    thread_data = {}
    
    matched_prompts = []
    total_tag_prompts = 0
    
    tag_filter_lower = tag_filter.lower() if tag_filter else None
    
    for p in prompts:
        # 1. Filter by Tag
        tags = p.get('tags') or []
        tags_lower = [t.lower() for t in tags]
        
        # If tag_filter is provided, check strict membership
        if tag_filter_lower:
            if tag_filter_lower not in tags_lower:
                continue
        
        # This prompt matches our filter
        total_tag_prompts += 1
        matched_prompts.append(p)
            
        # 2. Get Results History
        results_history = p.get('results', [])
        if not results_history:
            continue
            
        # 3. Find Latest Date (YYYY-MM-DD)
        dates = [r.get('created_at', '')[:10] for r in results_history if r.get('created_at')]
        if not dates:
            continue
            
        latest_date_str = max(dates)
        
        # 4. Filter results to this date
        current_results = [r for r in results_history if r.get('created_at', '').startswith(latest_date_str)]

        # 5. Extract Sources and Deduplicate PER PROMPT by Thread ID
        unique_threads_in_prompt = {} # id -> url found
        
        for res in current_results:
            sources = res.get('sources', [])
            for s in sources:
                s_url = s.get('url', '')
                if "reddit.com" in s_url.lower():
                        t_id = extract_thread_id(s_url)
                        if t_id:
                            # Store URL for slug extraction later
                            unique_threads_in_prompt[t_id] = s_url
                        
        # 6. Update Global Counts
        for t_id, s_url in unique_threads_in_prompt.items():
            if t_id not in thread_data:
                thread_data[t_id] = {'count': 0, 'slug': None}
            
            thread_data[t_id]['count'] += 1
            
            # Attempt to extract slug if not yet found
            if not thread_data[t_id]['slug'] and s_url:
                    # Check if URL has a slug segment after the ID
                    clean_s_url = s_url.split('?')[0].rstrip('/')
                    parts = clean_s_url.split('/')
                    # Find 'comments' index
                    try:
                        idx_comments = parts.index('comments')
                        if len(parts) > idx_comments + 2:
                            raw_slug = parts[idx_comments + 2]
                            formatted_slug = raw_slug.replace('_', ' ').capitalize()
                            thread_data[t_id]['slug'] = formatted_slug
                    except ValueError:
                        pass
        
    # Convert to list of dicts for DataFrame
    final_list = []
    for t_id, info in thread_data.items():
        # Calculation: (Unique Prompts Citing Thread / Total Tag Prompts)
        count = info['count']
        pct = (count / total_tag_prompts * 100) if total_tag_prompts > 0 else 0
        
        # Construct clean compatible URL
        clean_url = f"https://www.reddit.com/comments/{t_id}/"
        
        final_list.append({
            "url": clean_url, 
            "count": count,
            "percentage": pct,
            "title": info['slug'] if info['slug'] else "Unknown Title"
        })
    
    # Sort by count descending
    final_list.sort(key=lambda x: x['count'], reverse=True)
    
    return final_list, matched_prompts

def fetch_accuranker_data(brand_id, tag, api_token):
    """
    Fetches prompts from AccuRanker, filters by tag, and prepares verification tasks.
    Returns: list of task dicts.
    """
    if not brand_id or not api_token:
        return []

    # 1. Fetch Prompts (Reuse logic from raw fetch but tailored for this flow)
    # We need specific fields: description (Truth), results.prompt_response (AI Answer)
    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "id,prompt,tags,description,results.created_at,results.prompt_response,results.response_type,results.search_engine,results.sources.title,results.sources.url",
        "limit": 1000
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next')
                if url: params = None
            
            all_prompts.extend(chunk)
            
            if len(all_prompts) > 5000: # Safety
                break

    except Exception as e:
        return [{"error": f"API Error: {e}"}]

    # 2. Filter & Process
    verified_results = []
    tag_lower = tag.lower() if tag else None
    
    # Pre-process to identify all verification tasks
    verification_tasks = []
    supported_engines = ['chatgpt', 'perplexity', 'ai_overview', 'ai_mode']
    
    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        description = p.get('description', '')
        prompt_text = p.get('prompt', '')
        
        # Skip if no truth
        if not description or not description.strip():
             # We might want to still return a skipped result for visibility, 
             # but for now let's just skip processing or handle later?
             # Let's add a "placeholder" task that marks it as skipped for the first engine 
             # so the user sees "No Truth"
             verification_tasks.append({
                 "type": "skipped",
                 "prompt": prompt_text,
                 "truth": description,
                 "engine": "All"
             })
             continue

        results = p.get('results', [])
        if not results:
            continue

        # Group by engine
        engine_results = {e: [] for e in supported_engines}
        for r in results:
            se = r.get('search_engine')
            if se in engine_results:
                engine_results[se].append(r)
        
        # For each engine, find best result
        for engine, res_list in engine_results.items():
            if not res_list:
                continue
            
            # Sort by date desc
            res_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            # Find candidate with response
            candidate = None
            for r in res_list:
                if r.get('prompt_response'):
                    candidate = r
                    # Prefer one with sources if available? 
                    # For now just first with response is good basic check. 
                    # If we want to be fancy we can also check sources... 
                    # Let's stick to "latest with response" for robustness
                    break
            
            if candidate:
                 verification_tasks.append({
                     "type": "verify",
                     "prompt": prompt_text,
                     "truth": description,
                     "engine": engine,
                     "result_obj": candidate
                 })
    total = len(verification_tasks)
    
    return verification_tasks

def fetch_accuranker_sources(brand_id, tag, start_date, end_date, api_token, calculate_latest_only=True):
    """
    Fetches all prompts for a given date range and extracts all cited source URLs.
    Aggregates them by URL and Domain and calculates percentages based on prompts count.
    If calculate_latest_only is True, it matches AccuRanker UI behavior by only looking at the latest snapshot date in the period per prompt.
    """
    if not brand_id or not api_token or not start_date or not end_date:
        return pd.DataFrame()

    brand_ids = brand_id if isinstance(brand_id, list) else [brand_id]
    all_prompts = []

    for b_id in brand_ids:
        url = f"https://app.accuranker.com/api/v4/brands/{b_id}/prompts/"
        headers = {
            "Authorization": f"Token {api_token}",
            "Accept": "application/json",
        }
        
        # We need created_at to filter dates properly, and engine to count responses per engine
        params = {
            "fields": "id,prompt,tags,results.created_at,results.search_engine,results.sources.url",
            "limit": 1000,
            "period_from": start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date,
            "period_to": end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
        }
        
        try:
            while url:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                chunk = []
                if isinstance(data, list):
                    chunk = data
                    url = None
                elif isinstance(data, dict):
                    chunk = data.get('results', [])
                    url = data.get('next')
                    if url: params = None
                
                all_prompts.extend(chunk)
                
                if len(all_prompts) > 10000: # Safety
                    break

        except Exception as e:
            print(f"API Error fetching sources for brand {b_id}: {e}")
            continue

    if not all_prompts:
        return pd.DataFrame()

    # Filter by tag and process sources
    tag_lower = tag.lower() if tag else None
    
    source_counts = {} # url -> total engine responses
    url_counts = {} # url -> count of prompts having AT LEAST ONE occurrence of this url
    domain_prompts = {} # domain -> total engine responses
    domain_counts = {} # domain -> count of promos having AT LEAST ONE url from this domain
    total_matching_prompts = 0
    
    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        total_matching_prompts += 1
        
        # Collect unique sources in this prompt (for URL stats)
        unique_urls_in_prompt = set()
        # Collect unique domains in this prompt (for Domain stats)
        unique_domains_in_prompt = set()
        
        results = p.get('results', [])
        
        if calculate_latest_only:
            # Find latest date in this prompt's results
            dates = set(r.get('created_at', '')[:10] for r in results if r.get('created_at'))
            if not dates:
                continue
            latest_date = max(dates)
        
        for r in results:
            if calculate_latest_only and not r.get('created_at', '').startswith(latest_date):
                continue
                
            sources = r.get('sources', [])
            
            domains_this_response = set()
            urls_this_response = set()
            
            for s in sources:
                s_url = s.get('url')
                if s_url:
                    urls_this_response.add(s_url)
                    unique_urls_in_prompt.add(s_url)
                    try:
                        parsed = urllib.parse.urlparse(s_url)
                        domain = parsed.netloc
                        if domain.startswith("www."):
                            domain = domain[4:]
                        domains_this_response.add(domain)
                        unique_domains_in_prompt.add(domain)
                    except Exception:
                        pass
            
            # Count occurrences per response (how many times it showed up across all engines on latest day)
            for d in domains_this_response:
                domain_prompts[d] = domain_prompts.get(d, 0) + 1
            for u in urls_this_response:
                source_counts[u] = source_counts.get(u, 0) + 1
                    
        # Add to global unique keyword counts
        for d in unique_domains_in_prompt:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for u in unique_urls_in_prompt:
            url_counts[u] = url_counts.get(u, 0) + 1

    # Format output
    output_data = []
    
    # We will build output data keyed by URL and Domain
    for url, count in source_counts.items():
        # Clean URL to get Domain
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = "Unknown"
            
        url_kw_count = url_counts.get(url, 0)
        url_pct = (url_kw_count / total_matching_prompts * 100) if total_matching_prompts > 0 else 0
        
        # Domain citations: how many prompts cite this domain at least once
        domain_kw_count = domain_counts.get(domain, 0)
        domain_pct = (domain_kw_count / total_matching_prompts * 100) if total_matching_prompts > 0 else 0
        domain_total_prompts = domain_prompts.get(domain, 0)
        
        output_data.append({
            "Domain": domain,
            "Domain Cited (%)": domain_pct,
            "Domain Prompts": domain_total_prompts,
            "Full URL": url,
            "URL Cited (%)": url_pct,
            "Prompts": count
        })

    df = pd.DataFrame(output_data)
    if not df.empty:
        df = df.sort_values(by=["Domain Cited (%)", "URL Cited (%)"], ascending=[False, False]).reset_index(drop=True)
    return df

def verify_urls_with_dataforseo(urls_list, brand_name, login, password, progress_callback=None, tracked_details=None):
    """
    Verifies if a list of URLs mention a given brand name using the DataForSEO
    OnPage Content Parsing Live API.
    
    Uses ThreadPoolExecutor to send parallel requests (API limits Live endpoint to 1 URL/request).
    Returns: dict mapping url -> status dict (Mentions Brand, Mentioned Competitors, Competitor Count)
    """
    import base64
    
    if not urls_list or not brand_name or not login or not password:
        return {}
    
    api_url = "https://api.dataforseo.com/v3/on_page/content_parsing/live"
    
    # Handle if the password is ALREADY the combined base64 'login:password'
    cred = None
    try:
        decoded = base64.b64decode(password).decode('utf-8')
        if decoded.startswith(login + ":"):
            cred = password
    except Exception:
        pass
        
    if not cred:
        cred = base64.b64encode(f"{login}:{password}".encode()).decode()
        
    headers = {
        "Authorization": f"Basic {cred}",
        "Content-Type": "application/json"
    }
    
    brand_lower = brand_name.lower()
    results = {}
    total = len(urls_list)
    
    import concurrent.futures
    
    # Prepare competitor tracking variations
    competitor_tracking = {}
    if tracked_details:
        for comp_name, comp_info in tracked_details.items():
            if comp_name.lower() == brand_lower or brand_lower in comp_name.lower():
                continue
            aliases = [a.lower() for a in comp_info.get("brand_list", [])]
            aliases.append(comp_name.lower())
            competitor_tracking[comp_name] = list(set(aliases))
            
    def process_url(url):
        post_data = [{
            "url": url,
            "disable_cookie_popup": True,
            "enable_javascript": False,
            "enable_browser_rendering": False
        }]
        
        try:
            response = requests.post(api_url, headers=headers, json=post_data, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            tasks = data.get("tasks", [])
            if not tasks:
                return url, {
                    "Mentions Brand": "⚠️ Crawl Failed",
                    "Mentioned Competitors": "",
                    "Competitor Count": 0
                }
            
            task = tasks[0]
            if task.get("status_code") != 20000:
                return url, {
                    "Mentions Brand": "⚠️ Crawl Failed",
                    "Mentioned Competitors": "",
                    "Competitor Count": 0
                }
                
            task_result = task.get("result")
            if not task_result:
                return url, {
                    "Mentions Brand": "⚠️ Crawl Failed",
                    "Mentioned Competitors": "",
                    "Competitor Count": 0
                }
                
            items = task_result[0].get("items")
            if not items:
                return url, {
                    "Mentions Brand": "⚠️ Crawl Failed",
                    "Mentioned Competitors": "",
                    "Competitor Count": 0
                }
                
            items_str = json.dumps(items).lower()
            import re
            
            # Check main brand
            mentions_main_brand = False
            if brand_lower in ["ig", "ing"]:
                if re.search(r'\b' + re.escape(brand_lower) + r'\b', items_str):
                    mentions_main_brand = True
            else:
                if brand_lower in items_str:
                    mentions_main_brand = True
            
            found_comps = []
            if competitor_tracking:
                for comp_name, aliases in competitor_tracking.items():
                    comp_name_lower = comp_name.lower()
                    if comp_name_lower in ["ig", "ing"]:
                        found_comp_alias = False
                        for alias in aliases:
                            if re.search(r'\b' + re.escape(alias) + r'\b', items_str):
                                found_comp_alias = True
                                break
                        if found_comp_alias:
                            found_comps.append(comp_name)
                    else:
                        if any(alias in items_str for alias in aliases):
                            found_comps.append(comp_name)
            
            return url, {
                "Mentions Brand": "✅ Yes" if mentions_main_brand else "❌ No",
                "Mentioned Competitors": ", ".join(sorted(found_comps)),
                "Competitor Count": len(found_comps)
            }
                
        except Exception as e:
            print(f"DataForSEO API Error for {url}: {e}")
            return url, {
                "Mentions Brand": "⚠️ Crawl Failed",
                "Mentioned Competitors": "",
                "Competitor Count": 0
            }

    processed = 0
    # Process concurrently (using 5 workers to be safe with rate limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls_list}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result_url, status_dict = future.result()
                results[result_url] = status_dict
            except Exception as e:
                print(f"Exception for {url}: {e}")
                results[url] = {
                    "Mentions Brand": "⚠️ Crawl Failed",
                    "Mentioned Competitors": "",
                    "Competitor Count": 0
                }
                
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
    
    return results

def fetch_source_trends(brand_id, tag, start_date, end_date, api_token):
    """
    Fetches all prompts for a given date range and extracts all cited source domains over time.
    Aggregates them by Date and Domain and calculates percentages based on total prompts for that day.
    """
    if not brand_id or not api_token or not start_date or not end_date:
        return pd.DataFrame()

    brand_ids = brand_id if isinstance(brand_id, list) else [brand_id]
    all_prompts = []

    for b_id in brand_ids:
        url = f"https://app.accuranker.com/api/v4/brands/{b_id}/prompts/"
        headers = {
            "Authorization": f"Token {api_token}",
            "Accept": "application/json",
        }
        
        params = {
            "fields": "id,prompt,tags,results.created_at,results.search_engine,results.sources.url",
            "limit": 1000,
            "period_from": start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date,
            "period_to": end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
        }
        
        try:
            while url:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                chunk = []
                if isinstance(data, list):
                    chunk = data
                    url = None
                elif isinstance(data, dict):
                    chunk = data.get('results', [])
                    url = data.get('next')
                    if url: params = None
                
                all_prompts.extend(chunk)
                
                if len(all_prompts) > 10000: # Safety
                    break

        except Exception as e:
            print(f"API Error fetching source trends for brand {b_id}: {e}")
            continue

    if not all_prompts:
        return pd.DataFrame()

    tag_lower = tag.lower() if tag else None
    
    daily_total_prompts = {} # engine -> date -> count of prompts evaluated on that date
    daily_domain_counts = {} # engine -> date -> {domain -> count of prompts citing domain on that date}
    
    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        # Group results by date
        results = p.get('results', [])
        results_by_date = {}
        for r in results:
            date_str = r.get('created_at', '')[:10]
            if date_str:
                if date_str not in results_by_date:
                    results_by_date[date_str] = []
                results_by_date[date_str].append(r)
                
        # For each date this prompt has results
        for current_date, date_results in results_by_date.items():
            # 1. Aggregated Level
            daily_total_prompts.setdefault("Aggregated", {}).setdefault(current_date, 0)
            daily_total_prompts["Aggregated"][current_date] += 1
            
            # 2. Engine Level (Find unique engines for this prompt on this date)
            engines_for_prompt = set()
            for r in date_results:
                eng = r.get('search_engine') or r.get('engine', 'Unknown')
                engines_for_prompt.add(eng)
                
            for eng in engines_for_prompt:
                daily_total_prompts.setdefault(eng, {}).setdefault(current_date, 0)
                daily_total_prompts[eng][current_date] += 1
                
            # Now domains
            unique_domains_agg = set()
            unique_domains_eng = {} # eng -> set of domains
            
            for r in date_results:
                eng = r.get('search_engine') or r.get('engine', 'Unknown')
                if eng not in unique_domains_eng:
                    unique_domains_eng[eng] = set()
                    
                sources = r.get('sources', [])
                for s in sources:
                    s_url = s.get('url')
                    if s_url:
                        try:
                            parsed = urllib.parse.urlparse(s_url)
                            domain = parsed.netloc
                            if domain.startswith("www."):
                                domain = domain[4:]
                            unique_domains_agg.add(domain)
                            unique_domains_eng[eng].add(domain)
                        except Exception:
                            pass
                            
            # Save counts
            daily_domain_counts.setdefault("Aggregated", {}).setdefault(current_date, {})
            for d in unique_domains_agg:
                daily_domain_counts["Aggregated"][current_date][d] = daily_domain_counts["Aggregated"][current_date].get(d, 0) + 1
                
            for eng, doms in unique_domains_eng.items():
                daily_domain_counts.setdefault(eng, {}).setdefault(current_date, {})
                for d in doms:
                    daily_domain_counts[eng][current_date][d] = daily_domain_counts[eng][current_date].get(d, 0) + 1

    # Format output
    output_data = []
    
    for engine, engine_dates in daily_domain_counts.items():
        for date_str, domains in engine_dates.items():
            total_prompts_on_date = daily_total_prompts.get(engine, {}).get(date_str, 0)
            
            for domain, count in domains.items():
                pct = (count / total_prompts_on_date * 100) if total_prompts_on_date > 0 else 0
                
                output_data.append({
                    "Engine": engine,
                    "Date": date_str,
                    "Domain": domain,
                    "Domain Prompts": count,
                    "Total Prompts": total_prompts_on_date,
                    "Domain Cited (%)": pct
                })

    df = pd.DataFrame(output_data)
    if not df.empty:
        # Convert Date to datetime for proper sorting but keep as string if we want it to be compatible with Altair date stuff out of the box, or Altair handles datetime
        # Let's keep Date as strings (YYYY-MM-DD), Altair parses them nicely as 'Date:T'. Time-series chart works best when we use Date objects or strings that Altair maps directly.
        df = df.sort_values(by=["Date", "Domain Cited (%)"], ascending=[True, False]).reset_index(drop=True)
    return df


def verify_accuranker_data(tasks, openai_client, progress_callback=None):
    """
    Verifies a list of tasks against Ground Truth using OpenAI.
    """
    verified_results = []
    
    # Load System Prompt
    try:
        system_prompt = Path("prompts/llm_truth_check.txt").read_text()
    except Exception:
        system_prompt = "You are an expert fact-checker. Verify if the LLM response is consistent with the grounded truth description."
        
    total = len(tasks)
    
    for idx, task in enumerate(tasks):
        # Update Progress
        if progress_callback:
            progress_callback(idx, total, f"Verifying {idx+1}/{total} ({task.get('engine')}): {task.get('prompt')}")
        
        if task["type"] == "skipped":
            verified_results.append({
                "Prompt": task["prompt"],
                "Truth": task["truth"],
                "Engine": task["engine"],
                "Verdict": "Skipped",
                "Reason": "No truth defined",
                "Score": 0,
                "Status": "No Truth",
                "AI Response": "", 
                "Sources": [],
                "Source Count": 0
            })
            continue
            
        # Process Verification
        engine = task["engine"]
        res = task["result_obj"]
        
        ai_response = res.get('prompt_response')
        sources = res.get('sources', [])
        
        clean_sources = []
        for s in sources:
            url = s.get('url', '').strip()
            if not url:
                continue
            clean_sources.append({"URL": url})

        row = {
            "Prompt": task["prompt"],
            "Truth": task["truth"],
            "Engine": engine,
            "AI Response": ai_response,
            "Sources": clean_sources,
            "Source Count": len(clean_sources),
            "Verdict": "Pending",
            "Reason": "",
            "Score": 0,
            "Status": "Processed"
        }
        
        try:
            user_msg = f"Ground Truth: {task['truth']}\n\nLLM Response: {ai_response}"
            
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content
            # Handle potential JSON errors
            try:
                analysis = json.loads(content) if content else {}
                row["Verdict"] = analysis.get("verdict", "Unknown") if isinstance(analysis, dict) else "Error"
                row["Reason"] = analysis.get("reason", "No reason provided") if isinstance(analysis, dict) else "Invalid format"
                row["Score"] = analysis.get("score", 0) if isinstance(analysis, dict) else 0
            except (json.JSONDecodeError, TypeError, AttributeError):
                row["Verdict"] = "Error"
                row["Reason"] = f"Failed to parse JSON response: {content}"
                row["Score"] = 0
            
        except Exception as e:
            row["Verdict"] = "Error"
            row["Reason"] = str(e)
            
        verified_results.append(row)
        
    return verified_results

def fetch_unique_tags(brand_id, api_token):
    """
    Fetches all unique tags used in prompts for a specific brand.
    """
    if not brand_id or not api_token:
        return []

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    # We only need the tags field
    params = {
        "fields": "tags",
        "limit": 1000
    }
    
    unique_tags = {} # tag: count
    
    try:
        while url:
            # Short timeout for tag fetching
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next')
                if url: params = None
            
            for item in chunk:
                tags = item.get('tags') or []
                if tags:
                    for t in tags:
                        unique_tags[t] = unique_tags.get(t, 0) + 1
            
            if len(unique_tags) > 1000: # Safety
                break

    except Exception as e:
        print(f"Error fetching tags: {e}")
        return {}
        
    return unique_tags

# -----------------------------------------------------------------------------
# RANDOM TOOLS FUNCTIONS
# -----------------------------------------------------------------------------
def extract_all_sitemap_urls(sitemap_urls, progress_callback=None, error_callback=None):
    all_urls = []
    
    def crawl_sitemap(url):
        if progress_callback:
            progress_callback(url)
            
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            # Using 'xml' which defaults to lxml's xml parser if installed,
            # or the builtin if lxml isn't available. "lxml-xml" forces lxml.
            soup = BeautifulSoup(response.content, "xml")
            
            # Find sub-sitemaps
            sitemaps = soup.find_all("sitemap")
            for s in sitemaps:
                loc = s.find("loc")
                if loc and loc.text:
                    crawl_sitemap(loc.text.strip())
                    
            # Find URLs
            urls = soup.find_all("url")
            for u in urls:
                loc = u.find("loc")
                if loc and loc.text:
                    all_urls.append(loc.text.strip())
                    
        except Exception as e:
            print(f"Error fetching sitemap {url}: {e}")
            if error_callback:
                error_callback(url, str(e))
            
    for url in sitemap_urls:
        if url.strip():
            crawl_sitemap(url.strip())
            
    return list(set(all_urls))

def categorize_market(url):
    u = url.lower()
    if "bgsaxo.it" in u:
        return "IT"
    if "/da-dk" in u:
        return "DK"
    if "/nb-no" in u:
        return "NO"
    if "/fr-be" in u or "/nl-be" in u:
        return "BE"
    if "/rs-rs" in u or "/en-au" in u or "/en-hk" in u or "/zh-hk" in u:
        return "Non-Active"
    if "/cs-cz" in u:
        return "CZ"
    if "/fr-fr" in u:
        return "FR"
    if "/en-mena" in u or "/ar-mena" in u:
        return "MENA"
    if "/nl-nl" in u:
        return "NL"
    if "/pl-pl" in u:
        return "PL"
    if "/en-sg" in u:
        return "SG"
    if "/sk-sk" in u:
        return "SK"
    if "/en-ch" in u or "/fr-ch" in u or "/de-ch" in u:
        return "CH"
    if "/en-uk" in u or "/en-gb" in u:
        return "UK"
    if "/ja-jp" in u:
        return "JP"
    return "GL"

def categorize_language(url):
    u = url.lower()
    if "bgsaxo.it" in u:
        return "IT"
    
    match = re.search(r'/([a-z]{2})-[a-z]{2,4}(?:/|$)', u)
    if match:
        return match.group(1).upper()
        
    return "EN"

def categorize_website(url):
    """
    Categorizes a URL based on the Oncrawl segmentation rules for Website Categories.
    Prioritizes top-to-bottom as per the original JSON structure.
    """
    u = url.lower()
    
    # 2. Global rule: If it has query parameters, dump it to "Other"
    if "?" in u:
        return "Other"
        
    # Products
    if "/products" in u and not ("/google/products" in u or "/saxowealthcare" in u or "/products/platforms" in u or "/login" in u or "/rates-and-conditions" in u):
        return "Products"
        
    # Discover Hub
    if ("/learn" in u or "/glossary" in u) and not ("/learn-options" in u or "/learn-to-trade-in-uncertain-markets" in u or "/saxoinvestor" in u or "/us-election" in u or "/uk-isa" in u or "/education" in u or "test" in u or "content/articles" in u or "/campaigns/" in u):
        return "Discover Hub"
        
    # Accounts
    if "/accounts" in u and "/login" not in u:
        return "Accounts"
        
    # Legal
    if "/legal" in u:
        return "Legal"
        
    # Rates & Conditions
    if "/rates-and-conditions" in u:
        return "Rates & Conditions"
        
    # Campaigns
    if "/campaigns" in u:
        return "Campaigns"
        
    # Institutional
    if "/institutional-and-partners" in u:
        return "Institutional"
        
    # About Us & Contact
    if ("/about" in u or "/contact-us" in u) and not ("/institutional-and-partners" in u or "/legal" in u):
        return "About Us & Contact"
        
    # Insights & Commentaries
    if ("/insights" in u or "/content/commentaries" in u or "/content/articles" in u) and not ("/ja-jp/content/commentaries/wnu/" in u):
        return "Insights & Commentaries"
        
    # Platforms
    if "/platforms" in u and not ("campaigns/platforms" in u or "/webinars" in u or "/login" in u or "/institutional-and-partners" in u):
        return "Platforms"
        
    # Login
    if "/login" in u and not ("/insights" in u or "/commentaries/wnu/whats-new/" in u):
        return "Login"
        
    # Home pages
    # 1. Trailing slash modification: allow optional trailing slash
    # The JSON used equals for some ("https://cn.saxobank.com/", "https://www.home.saxo/en-mena", "https://www.home.saxo/", "https://www.home.saxo/ar-mena") and regex for "https://www.home.saxo/..-..$"
    # Rather than full regex, we can check basic matches for homepages.
    
    # Check absolute matches first (with or without trailing slash)
    exact_matches = [
        "https://cn.saxobank.com", "https://cn.saxobank.com/",
        "https://www.home.saxo/en-mena", "https://www.home.saxo/en-mena/",
        "https://www.home.saxo/ar-mena", "https://www.home.saxo/ar-mena/",
        "https://www.home.saxo", "https://www.home.saxo/"
    ]
    if u in exact_matches:
        return "Home pages"
        
    # Check the "https://www.home.saxo/..-.." regex equivalent (e.g. https://www.home.saxo/da-dk or https://www.home.saxo/da-dk/)
    if u.startswith("https://www.home.saxo/") and len(u) >= 27:
        # e.g., "https://www.home.saxo/da-dk" is 27 chars.
        parts = u.split("https://www.home.saxo/")[1].split("/")
        if len(parts) > 0 and len(parts[0]) == 5 and parts[0][2] == "-":
            # If it's JUST the locale or locale + slash
            if len(parts) == 1 or (len(parts) == 2 and parts[1] == ""):
                return "Home pages"
                
    # Juno Stocks
    if "/markets/stocks/" in u:
        return "Juno Stocks"
        
    # 3. Fallback
    return "Other"

def extract_market_language_combination(url):
    u = url.lower()
    if "bgsaxo.it" in u:
        return "it-it"
    
    import urllib.parse
    import re
    parsed = urllib.parse.urlparse(u)
    path = parsed.path
    if path.startswith("/"):
        parts = path.split("/")
        if len(parts) > 1:
            first_segment = parts[1]
            if re.match(r'^[a-z]{2}-[a-z]{2,4}$', first_segment):
                return first_segment
    return "global"

def extract_language_from_combination(combination):
    if "-" in combination:
        return combination.split("-")[0]
    return combination

def extract_url_slug(url):
    u = url.lower()
    import urllib.parse
    import re
    parsed = urllib.parse.urlparse(u)
    path = parsed.path
    
    if "bgsaxo.it" in u:
        return path if path else "/"
        
    if path.startswith("/"):
        parts = path.split("/")
        if len(parts) > 1:
            first_segment = parts[1]
            if re.match(r'^[a-z]{2}-[a-z]{2,4}$', first_segment):
                slug = "/" + "/".join(parts[2:])
                if slug == "//": return "/"
                return slug
                
    return path if path else "/"

def fetch_kpi_time_series(brand_id, tag, start_date, end_date, api_token):
    """
    Fetches historical KPI data (Visibility and Sentiment) from AccuRanker for a given brand and tag.
    Returns: A tuple (Pandas DataFrame indexed by Date, Pandas DataFrame of unique prompts).
    """
    if not brand_id or not api_token or not start_date or not end_date:
        return pd.DataFrame(), pd.DataFrame()

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "id,prompt,tags,results.created_at,results.brands.visibility,results.brands.sentiment,results.brands.is_own,results.brands.competitor.display_name,results.brands.competitor.pinned",
        "limit": 1000,
        "period_from": start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date,
        "period_to": end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next')
                if url: params = None
            
            all_prompts.extend(chunk)
            
            if len(all_prompts) > 10000: # Safety
                break

    except Exception as e:
        print(f"API Error fetching KPI series: {e}")
        return pd.DataFrame(), pd.DataFrame()

    tag_lower = tag.lower() if tag else None
    
    data_by_date = {}
    unique_prompts = []

    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        # Collect prompt
        prompt_text = p.get('prompt', '')
        if prompt_text and not any(up['Prompt'] == prompt_text for up in unique_prompts):
            tags_str = ", ".join(p.get('tags', []))
            unique_prompts.append({
                "Prompt": prompt_text,
                "Tags": tags_str
            })
            
        results = p.get('results', [])
        for r in results:
            date_str = r.get('created_at', '')[:10]
            if not date_str:
                continue
                
            if date_str not in data_by_date:
                data_by_date[date_str] = {'_total_prompts': 0}
                
            # Increment total prompts for this date
            data_by_date[date_str]['_total_prompts'] += 1
            
            brands = r.get('brands', [])
            for b in brands:
                is_own = b.get('is_own')
                comp = b.get('competitor')
                
                # Default is Own Brand
                entity_name = 'Own'
                
                # If not own, it might be a pinned competitor
                if not is_own and comp and comp.get('pinned'):
                     entity_name = comp.get('display_name')
                elif not is_own:
                     # We only care about own brand AND pinned competitors
                     continue
                
                visi = b.get('visibility')
                sent = b.get('sentiment')
                
                visi = float(visi) if visi is not None else 0.0
                sent = float(sent) if sent is not None else 0.0
                
                if date_str not in data_by_date:
                    data_by_date[date_str] = {}
                    
                if entity_name not in data_by_date[date_str]:
                     data_by_date[date_str][entity_name] = {'visi_sum': 0.0, 'visi_count': 0, 'sent_sum': 0.0, 'sent_count': 0}
                    
                data_by_date[date_str][entity_name]['visi_sum'] += visi
                data_by_date[date_str][entity_name]['visi_count'] += 1
                
                if visi > 0:
                    data_by_date[date_str][entity_name]['sent_sum'] += sent
                    data_by_date[date_str][entity_name]['sent_count'] += 1
                    
    records = []
    for date_str, entities in data_by_date.items():
        row = {'Date': date_str}
        total_p = entities.pop('_total_prompts', 1) # default to 1 to avoid div by zero if missing
        
        for entity_name, aggs in entities.items():
             v_count = aggs['visi_count']
             s_count = aggs['sent_count']
             
             if v_count > 0:
                 # Competitors only appear when they have a rank/visibility.
                 # Their average visibility for a day is the sum of their visibility across all prompts
                 # divided by the total number of prompts checked that day.
                 # Own brand might already have 0 visibility recorded in some prompts, so its v_count
                 # usually equals total_p, but to be safe and accurate for both, we divide by total_p.
                 avg_visi = aggs['visi_sum'] / total_p
                 avg_sent = (aggs['sent_sum'] / s_count) if s_count > 0 else 0.0
                 
                 if entity_name == 'Own':
                      row['Visibility'] = avg_visi
                      row['Sentiment'] = avg_sent
                 else:
                      row[f'Visibility - {entity_name}'] = avg_visi
                      row[f'Sentiment - {entity_name}'] = avg_sent
        
        # Only add if it has data
        if len(row) > 1:
            records.append(row)
            
    df = pd.DataFrame(records)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
    df_prompts = pd.DataFrame(unique_prompts)
    return df, df_prompts

def fetch_competitive_overview(brand_id, brand_name, tag, start_date, end_date, api_token):
    """
    Fetches raw historical KPI data for ALL competitors and the primary brand.
    Returns a long-form DataFrame with columns: Date | Competitor | Domain | Visibility | Sentiment.
    """
    if not brand_id or not api_token or not start_date or not end_date:
        return pd.DataFrame()

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "tags,results.created_at,results.brands.visibility,results.brands.sentiment,results.brands.is_own,results.brands.competitor.display_name,results.brands.competitor.domain",
        "limit": 1000,
        "period_from": start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date,
        "period_to": end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = data if isinstance(data, list) else data.get('results', [])
            all_prompts.extend(chunk)
            
            url = None if isinstance(data, list) else data.get('next')
            if url: params = None
            
            if len(all_prompts) > 10000:
                break
    except Exception as e:
        print(f"API Error fetching competitive overview: {e}")
        return pd.DataFrame()

    tag_lower = tag.lower() if tag else None
    data_by_date = {}

    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        results = p.get('results', [])
        for r in results:
            date_str = r.get('created_at', '')[:10]
            if not date_str: continue
            
            if date_str not in data_by_date:
                data_by_date[date_str] = {'_total_prompts': 0}
                
            data_by_date[date_str]['_total_prompts'] += 1
            
            brands = r.get('brands', [])
            for b in brands:
                is_own = b.get('is_own')
                comp = b.get('competitor')
                
                if is_own:
                    entity_name = brand_name
                    domain = "home.saxo" # Default Saxo domain
                elif comp:
                    entity_name = comp.get('display_name', 'Unknown')
                    domain = comp.get('domain', '')
                else:
                    continue
                    
                visi = b.get('visibility')
                sent = b.get('sentiment')
                visi = float(visi) if visi is not None else 0.0
                sent = float(sent) if sent is not None else 0.0
                
                key = (entity_name, domain)
                if key not in data_by_date[date_str]:
                     data_by_date[date_str][key] = {'visi_sum': 0.0, 'visi_count': 0, 'sent_sum': 0.0, 'sent_count': 0}
                    
                data_by_date[date_str][key]['visi_sum'] += visi
                data_by_date[date_str][key]['visi_count'] += 1
                if visi > 0:
                    data_by_date[date_str][key]['sent_sum'] += sent
                    data_by_date[date_str][key]['sent_count'] += 1
                    
    records = []
    for date_str, entities in data_by_date.items():
        total_p = entities.pop('_total_prompts', 1)
        for (entity_name, domain), aggs in entities.items():
             v_count = aggs['visi_count']
             s_count = aggs['sent_count']
             if v_count > 0:
                 avg_visi = aggs['visi_sum'] / total_p
                 avg_sent = (aggs['sent_sum'] / s_count) if s_count > 0 else 0.0
                 records.append({
                     'Date': date_str,
                     'Competitor': entity_name,
                     'Domain': domain,
                     'Visibility': avg_visi,
                     'Sentiment': avg_sent
                 })
                 
    df = pd.DataFrame(records)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def fetch_cross_market_data(brand_id, tag, start_date, end_date, api_token):
    """
    Fetches aggregate KPI data for a specific brand, broken down by LLM (search_engine).
    Returns a dictionary mapping search_engine -> {'Visibility': float, 'Sentiment': float, 'Web Search Rate': float}.
    """
    if not brand_id or not api_token or not start_date or not end_date:
        return {}

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "id,prompt,tags,results.created_at,results.search_engine,results.brands.visibility,results.brands.sentiment,results.brands.web_search_rate,results.brands.is_own",
        "limit": 1000,
        "period_from": start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date,
        "period_to": end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next')
                if url: params = None
            
            all_prompts.extend(chunk)
            
            if len(all_prompts) > 10000: # Safety
                break

    except Exception as e:
        print(f"API Error fetching cross market data: {e}")
        return {}

    tag_lower = tag.lower() if tag else None
    
    engine_aggs = {}

    for p in all_prompts:
        p_tags = [t.lower() for t in (p.get('tags') or [])]
        if tag_lower and tag_lower not in p_tags:
            continue
            
        results = p.get('results', [])
        
        # Group results by search engine to find the latest date per engine for this prompt
        results_by_engine = {}
        for r in results:
            engine = r.get('search_engine')
            if not engine: continue
            
            # Simple date comparison works for ISO 8601 strings
            if engine not in results_by_engine or r.get('created_at', '') > results_by_engine[engine].get('created_at', ''):
                results_by_engine[engine] = r
                
        # Now parse the latest result per engine
        for engine, r in results_by_engine.items():
            brands = r.get('brands', [])
            for b in brands:
                if b.get('is_own'):
                    visi = b.get('visibility')
                    sent = b.get('sentiment')
                    wsr = b.get('web_search_rate')
                    
                    visi = float(visi) if visi is not None else 0.0
                    sent = float(sent) if sent is not None else 0.0
                    wsr = float(wsr) if wsr is not None else 0.0
                    
                    if engine not in engine_aggs:
                        engine_aggs[engine] = {'visi_sum': 0.0, 'visi_count': 0, 'sent_sum': 0.0, 'sent_count': 0, 'wsr_sum': 0.0, 'wsr_count': 0}
                        
                    engine_aggs[engine]['visi_sum'] += visi
                    engine_aggs[engine]['visi_count'] += 1
                    
                    engine_aggs[engine]['wsr_sum'] += wsr
                    engine_aggs[engine]['wsr_count'] += 1
                    
                    if visi > 0:
                        engine_aggs[engine]['sent_sum'] += sent
                        engine_aggs[engine]['sent_count'] += 1
                        
                    break
                    
    final_data = {}
    for engine, aggs in engine_aggs.items():
        v_count = aggs['visi_count']
        s_count = aggs['sent_count']
        w_count = aggs['wsr_count']
        
        if v_count > 0:
            avg_visi = aggs['visi_sum'] / v_count
            avg_sent = (aggs['sent_sum'] / s_count) if s_count > 0 else 0.0
            avg_wsr = (aggs['wsr_sum'] / w_count) if w_count > 0 else 0.0
            
            final_data[engine] = {
                'Visibility': avg_visi,
                'Sentiment': avg_sent,
                'Web Search Rate': avg_wsr
            }
            
    return final_data

# -----------------------------------------------------------------------------
# SEMANTIC POSITIONING MAP FUNCTIONS
# -----------------------------------------------------------------------------

def fetch_competitor_names(brand_id, api_token):
    """
    Fetches unique competitor display names from AccuRanker for a given brand.
    Competitors are shared across all tags, so no tag/date filter is needed.
    Always appends 'Saxo Bank' to the list.
    Returns: List of unique brand/competitor name strings.
    """
    if not brand_id or not api_token:
        return ["Saxo Bank"]

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "results.brands.is_own,results.brands.competitor.display_name",
        "limit": 1000
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = data if isinstance(data, list) else data.get('results', [])
            all_prompts.extend(chunk)
            
            url = None if isinstance(data, list) else data.get('next')
            if url: params = None
            
            if len(all_prompts) > 10000:
                break
    except Exception as e:
        print(f"API Error fetching competitor names: {e}")
        return ["Saxo Bank"]

    competitor_names = set()

    for p in all_prompts:
        results = p.get('results', [])
        for r in results:
            brands = r.get('brands', [])
            for b in brands:
                is_own = b.get('is_own')
                comp = b.get('competitor')
                
                if not is_own and comp:
                    name = comp.get('display_name')
                    if name:
                        competitor_names.add(name)
    
    # Always include Saxo Bank
    competitor_names.add("Saxo Bank")
    
    return sorted(list(competitor_names))


def get_embeddings(text_list, model_choice, api_keys):
    """
    Gets embeddings for a list of texts using either OpenAI or Google.
    
    Args:
        text_list: List of strings to embed.
        model_choice: 'openai' or 'google'.
        api_keys: Dict with 'openai' and/or 'google' keys.
    
    Returns: List of embedding vectors (list of floats each).
    """
    if not text_list:
        return []
    
    if model_choice == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_keys.get('openai'))
        
        response = client.embeddings.create(
            input=text_list,
            model="text-embedding-3-small"
        )
        
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    
    else:  # google
        client = genai.Client(api_key=api_keys.get('google'))
        
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text_list,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY"
            )
        )
        
        return [emb.values for emb in response.embeddings]


def compute_similarity_matrix(brand_embeddings, feature_embeddings, brand_names, feature_names):
    """
    Computes cosine similarity matrix between brand and feature embeddings.
    
    Returns: pd.DataFrame with brands as rows and features as columns.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    brand_matrix = np.array(brand_embeddings)
    feature_matrix = np.array(feature_embeddings)
    
    sim_matrix = cosine_similarity(brand_matrix, feature_matrix)
    
    df = pd.DataFrame(sim_matrix, index=brand_names, columns=feature_names)
    df.index.name = "Brand"
    
    return df


def get_context_wrapped_embeddings(text_list, model_choice, api_keys):
    """
    Wraps each text item in a standardized financial context before embedding.
    This solves the 'entity trap' where brands and features cluster by type
    rather than by semantic similarity.
    
    Returns: List of embedding vectors.
    """
    wrapped = [
        f"The concept of {item} within the online trading, investing, and wealth management industry."
        for item in text_list
    ]
    return get_embeddings(wrapped, model_choice, api_keys)


def compute_quadrant_coordinates(brand_embeddings, anchor_embeddings, brand_names):
    """
    Computes Z-score standardized X/Y coordinates for brands based on 4 anchor concepts.
    The center (0,0) represents the market average for the cohort.
    
    Args:
        brand_embeddings: List of brand embedding vectors.
        anchor_embeddings: List of 4 anchor embedding vectors in order:
                          [x_left, x_right, y_bottom, y_top]
        brand_names: List of brand name strings.
    
    Returns: pd.DataFrame with columns: Brand, X, Y
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    brand_matrix = np.array(brand_embeddings)
    anchor_matrix = np.array(anchor_embeddings)
    
    # Similarity of each brand to each anchor: shape (n_brands, 4)
    sims = cosine_similarity(brand_matrix, anchor_matrix)
    
    # Raw coordinates: differential similarity
    raw_x = sims[:, 1] - sims[:, 0]  # sim_right - sim_left
    raw_y = sims[:, 3] - sims[:, 2]  # sim_top - sim_bottom
    
    # Z-score standardization: center on cohort mean, scale by std
    mean_x, std_x = np.mean(raw_x), np.std(raw_x)
    mean_y, std_y = np.mean(raw_y), np.std(raw_y)
    
    # Safety: avoid division by zero if std is 0 (e.g. single brand)
    x_coords = (raw_x - mean_x) / std_x if std_x > 0 else np.zeros_like(raw_x)
    y_coords = (raw_y - mean_y) / std_y if std_y > 0 else np.zeros_like(raw_y)
    
    df = pd.DataFrame({
        "Brand": brand_names,
        "X": x_coords,
        "Y": y_coords
    })
    
    return df

def discover_new_competitors(prompts_list, tracked_brands, api_keys):
    gemini_key = api_keys.get('google')
    openai_key = api_keys.get('openai')
    
    combined_text = ""
    for p in prompts_list:
        results = p.get('results', [])
        for r in results:
            response = r.get('prompt_response')
            if response:
                combined_text += response + "\n"
                
    if len(combined_text) > 40000:
        combined_text = combined_text[:40000]
                
    if not combined_text.strip():
        return []
        
    system_instruction = f"""You are a competitive intelligence analyst. Extract ALL competitor brand names mentioned in the following text (which contains AI search engine responses).
Include both already known brands and potential new ones.
Return EXACTLY a JSON object with a single key "competitors", which contains an array of objects with these keys:
- "Brand Name": The name of the competitor.
- "Website URL": Their official website URL (clean format without https://, www., or trailing slash. e.g., 'example.com'). Use your internal knowledge or web search to find the correct URL.
- "Alternative Names": Comma-separated alternative names or tickers for the brand (e.g., 'IBKR'). If none, use an empty string.

Output MUST be valid JSON, starting with {{ and ending with }}. Do not wrap in markdown or anything else."""
    
    raw_competitors = []

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": combined_text}
                ],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
            raw_competitors = data.get("competitors", [])
        except Exception as e:
            print(f"OpenAI failed for Competitor Scraper: {e}")
            
    if not raw_competitors:
        return []

    # Post-process to mark Tracked vs New
    # tracked_brands is now a dict: {"Name": {"domain": ..., "brand_list": ...}}
    tracked_lower_map = {str(k).lower().strip(): v for k, v in tracked_brands.items()}
    
    final_competitors = []
    seen = set()
    
    # First, add all the ones the AI found
    for comp in raw_competitors:
        name_orig = comp.get("Brand Name", "")
        name = str(name_orig).lower().strip()
        if not name or name in seen:
            continue
        seen.add(name)
        
        if name in tracked_lower_map:
            comp["Status"] = "Tracked (Mentioned) ✅"
            # Overwrite with AccuRanker API details to avoid AI hallucinations
            details = tracked_lower_map[name]
            if details.get("domain"):
                comp["Website URL"] = details["domain"]
            if details.get("brand_list"):
                comp["Alternative Names"] = ", ".join(details["brand_list"])
        else:
            comp["Status"] = "New ✨"
        final_competitors.append(comp)
        
    # Now, add all tracked brands that were NOT mentioned by AI
    for t_brand, details in tracked_brands.items():
        t_name_lower = str(t_brand).lower().strip()
        if t_name_lower and t_name_lower not in seen:
            seen.add(t_name_lower)
            alt_names = ", ".join(details.get("brand_list", []))
            final_competitors.append({
                "Brand Name": t_brand,
                "Website URL": details.get("domain", ""),
                "Alternative Names": alt_names,
                "Status": "Tracked (Not Mentioned) ➖"
            })
            
    return final_competitors

def fetch_competitor_details(brand_id, api_token):
    if not brand_id or not api_token:
        # Default fallback
        return {"Saxo Bank": {"domain": "home.saxo", "brand_list": ["Saxo Bank", "Saxo"]}}

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {
        "fields": "results.brands.is_own,results.brands.competitor.display_name,results.brands.competitor.domain,results.brands.competitor.brand_list",
        "limit": 1000
    }
    
    all_prompts = []
    
    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            chunk = data if isinstance(data, list) else data.get('results', [])
            all_prompts.extend(chunk)
            
            url = None if isinstance(data, list) else data.get('next')
            if url: params = None
            
            if len(all_prompts) > 10000:
                break
    except Exception as e:
        print(f"API Error fetching competitor details: {e}")
        return {"Saxo Bank": {"domain": "home.saxo", "brand_list": ["Saxo Bank", "Saxo"]}}

    competitor_details = {"Saxo Bank": {"domain": "home.saxo", "brand_list": ["Saxo Bank", "Saxo"]}}

    for p in all_prompts:
        results = p.get('results', [])
        for r in results:
            brands = r.get('brands', [])
            for b in brands:
                is_own = b.get('is_own')
                comp = b.get('competitor')
                
                if not is_own and comp:
                    name = comp.get('display_name')
                    if name and name not in competitor_details:
                        competitor_details[name] = {
                            "domain": comp.get('domain', ''),
                            "brand_list": comp.get('brand_list', [])
                        }
    
    return competitor_details


def fetch_prompts_lightweight(brand_id, api_token):
    """
    Fetches unique prompts (id, prompt, description, tags) from AccuRanker.
    Returns only prompt-level data without results to avoid duplicates per LLM engine.
    """
    if not brand_id or not api_token:
        return []

    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }

    params = {
        "fields": "id,prompt,description,tags",
        "limit": 1000
    }

    prompts = []

    try:
        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            chunk = []
            if isinstance(data, list):
                chunk = data
                url = None
            elif isinstance(data, dict):
                chunk = data.get('results', [])
                url = data.get('next')
                if url:
                    params = None

            prompts.extend(chunk)

            if len(prompts) > 10000:
                break

        return prompts

    except Exception as e:
        print(f"AccuRanker API Error (lightweight): {e}")
        return prompts


def generate_data_extractor_excel(brands_dict, api_token, progress_callback=None):
    """
    Generates an Excel file with two sheets:
      - 'Prompts': Brand, Prompt, Description, Tags
      - 'Competitors': Competitor Brand Name, Website URL, Alternative Spelling Versions

    Args:
        brands_dict: dict of {brand_name: brand_id} for selected brands
        api_token: AccuRanker API token
        progress_callback: optional callable(progress_float, status_text)

    Returns:
        BytesIO buffer containing the .xlsx file
    """
    import io

    all_prompts_rows = []
    all_competitors = {}  # name -> {domain, brand_list} — deduplicated across brands
    comp_per_market_rows = []  # per-brand competitor rows
    total_brands = len(brands_dict)

    for idx, (brand_name, brand_id) in enumerate(brands_dict.items()):
        if progress_callback:
            progress_callback(
                (idx / total_brands) * 0.9,
                f"Fetching data for {brand_name}..."
            )

        # --- Prompts ---
        raw_prompts = fetch_prompts_lightweight(brand_id, api_token)
        seen_prompts = set()
        for p in raw_prompts:
            prompt_text = p.get('prompt', '')
            if prompt_text in seen_prompts:
                continue
            seen_prompts.add(prompt_text)

            tags = p.get('tags') or []
            description = p.get('description') or ''
            all_prompts_rows.append({
                "Brand": brand_name,
                "Prompt": prompt_text,
                "Description": description,
                "Tags": ", ".join(tags)
            })

        # --- Competitors ---
        comp_details = fetch_competitor_details(brand_id, api_token)
        for comp_name, details in comp_details.items():
            domain = details.get("domain", "")
            brand_list = details.get("brand_list", [])

            # Per-market row
            comp_per_market_rows.append({
                "Brand": brand_name,
                "Competitor Brand Name": comp_name,
                "Website URL": domain,
                "Alternative Spelling Versions": ", ".join(brand_list)
            })

            # Aggregated (deduplicated)
            if comp_name not in all_competitors:
                all_competitors[comp_name] = {
                    "domain": domain,
                    "brand_list": brand_list
                }

    # Build DataFrames
    df_prompts = pd.DataFrame(all_prompts_rows, columns=["Brand", "Prompt", "Description", "Tags"])

    comp_agg_rows = []
    for comp_name, details in all_competitors.items():
        comp_agg_rows.append({
            "Competitor Brand Name": comp_name,
            "Website URL": details.get("domain", ""),
            "Alternative Spelling Versions": ", ".join(details.get("brand_list", []))
        })
    df_comp_aggregated = pd.DataFrame(comp_agg_rows, columns=["Competitor Brand Name", "Website URL", "Alternative Spelling Versions"])

    df_comp_per_market = pd.DataFrame(comp_per_market_rows, columns=["Brand", "Competitor Brand Name", "Website URL", "Alternative Spelling Versions"])

    # Write to Excel buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_prompts.to_excel(writer, sheet_name="Prompts", index=False)
        df_comp_aggregated.to_excel(writer, sheet_name="Competitors Aggregated", index=False)
        df_comp_per_market.to_excel(writer, sheet_name="Competitors per market", index=False)
    buffer.seek(0)

    if progress_callback:
        progress_callback(1.0, "Excel file ready!")

    return buffer
