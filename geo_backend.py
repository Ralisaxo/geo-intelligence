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
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
from sklearn.decomposition import PCA

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
    try:
        with open("prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
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

def get_semantic_similarity(text1, text2, api_key):
    """
    Computes semantic similarity between two texts using Gemini embeddings.
    Returns: (Score (0-100 float), vector1, vector2)
    """
    if not api_key:
        return 0.0, None, None

    genai.configure(api_key=api_key)
    
    try:
        # Get embeddings
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=[text1, text2],
            task_type="semantic_similarity"
        )
        
        if 'embedding' not in result:
             # Fallback
             v1 = genai.embed_content(model="models/text-embedding-004", content=text1)['embedding']
             v2 = genai.embed_content(model="models/text-embedding-004", content=text2)['embedding']
        else:
             v1 = result['embedding'][0]
             v2 = result['embedding'][1]

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
        return 0.0, None, None

def calculate_display_score(raw_score):
    """
    Normalizes raw cosine similarity (typically 0.35-0.65) to a 0-100% human-readable scale.
    """
    # Anchor points based on data analysis
    min_anchor = 0.32  # Scores below this become 0%
    max_anchor = 0.70  # Scores above this become 100%

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

    genai.configure(api_key=api_key)
    # Use specific version requested
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""
    You are a Brand Reputation Analyst for an investment bank. 
    Classify the sentiment of the following adjectives strictly as 'Positive', 'Negative', or 'Neutral' in the context of a financial trading platform. 
    Examples: 'Risky' is Negative. 'Expensive' is Negative. 'Complex' is Negative. 'Robust' is Positive.
    Input: {json.dumps(phrases)}
    Output JSON: {{phrase: sentiment}}
    Do not use markdown formatting. Return raw JSON.
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean potential markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        # LOGGING FOR DEBUGGING
        with open("sentiment_debug.txt", "w", encoding="utf-8") as f:
            f.write(f"Prompt Input: {json.dumps(phrases)}\n")
            f.write(f"Raw Response: {text}\n")

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
                                    {"role": "system", "content": "You are a helpful assistant. Follow constraints strictly."},
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
        genai.configure(api_key=gemini_key)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        for model_name in selected_gemini_models:
            model_instance = genai.GenerativeModel(model_name)
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
                        response = model_instance.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=100,
                                temperature=1.0
                            ),
                            safety_settings=safety_settings
                        )
                        if response.candidates and response.candidates[0].content.parts:
                             responses.append(response.candidates[0].content.parts[0].text.strip())
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
        "fields": "id,prompt,tags,results.created_at,results.sources.url,results.sources.title,results.sources.rank",
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
        tags = p.get('tags', [])
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
