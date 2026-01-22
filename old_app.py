import streamlit as st
import pandas as pd
import os
import io
import zipfile
import time
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import google.generativeai as genai
from openai import OpenAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- UI Configuration ---
st.set_page_config(page_title="Word Cloud Competitive Analysis", layout="wide")
st.title("Word Cloud Competitive Analysis Tool")

# --- Default Prompts from Notebook ---
DEFAULT_PROMPTS = """Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank is ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank is generally considered ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank is safe because ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank's reputation is ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank is best suited for ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: The SaxoTrader platform is designed for ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Users describe the SaxoTrader interface as ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: The SaxoInvestor platform is designed for ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Users describe the SaxoInvestor interface as ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank's pricing is considered ____.
Complete the sentence with exactly one word or adjective. Do NOT output a full sentence. Do NOT explain. Sentence: Saxo Bank's customer service is ____."""

# --- Helper Functions ---

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def normalize_brand(text):
    mapping = {
        "saxo bank": "saxo",
        "ig group": "ig",
        "interactive brokers": "ibkr",
        "degiro": "degiro",
    }
    return mapping.get(text, text)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("API Keys")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("Run Settings")
    iterations = st.number_input("Number of Iterations", min_value=1, max_value=100, value=1, help="How many times to run each prompt per model.")
    
    st.subheader("Model Selection")
    
    if st.button("Fetch Available Models"):
        # Fetch OpenAI Models
        if openai_api_key:
            try:
                client = OpenAI(api_key=openai_api_key)
                models = client.models.list()
                gpt_models = [m.id for m in models.data if 'gpt' in m.id]
                st.session_state['openai_models_list'] = sorted(gpt_models)
                st.success(f"Found {len(gpt_models)} OpenAI models.")
            except Exception as e:
                st.error(f"Error fetching OpenAI models: {e}")
        else:
             st.warning("Enter OpenAI API Key to fetch models.")

        # Fetch Gemini Models
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                models = genai.list_models()
                gemini_models = [m.name.replace('models/', '') for m in models if 'generateContent' in m.supported_generation_methods]
                st.session_state['gemini_models_list'] = sorted(gemini_models)
                st.success(f"Found {len(gemini_models)} Gemini models.")
            except Exception as e:
                st.error(f"Error fetching Gemini models: {e}")
        else:
             st.warning("Enter Gemini API Key to fetch models.")
    
    # Model Multiselects
    available_openai = st.session_state.get('openai_models_list', ["gpt-5.2", "gpt-5.2-pro", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    # Ensure default is in options
    if "gpt-5.2" not in available_openai:
        available_openai.insert(0, "gpt-5.2")
    
    selected_openai_models = st.multiselect("Select OpenAI Models", available_openai, default=["gpt-5.2"])
    
    available_gemini = st.session_state.get('gemini_models_list', ["gemini-3-flash-preview", "gemini-1.5-flash", "gemini-pro"])
    # Ensure default is in options
    if "gemini-3-flash-preview" not in available_gemini:
        available_gemini.insert(0, "gemini-3-flash-preview")

    selected_gemini_models = st.multiselect("Select Gemini Models", available_gemini, default=["gemini-3-flash-preview"])

    st.markdown("---")
    st.markdown("Created by Rasmus Lindbacke using Antigravity 2026")


# --- Main Area ---
st.subheader("Prompts")
prompts_text = st.text_area("Edit Prompts (One per line)", value=DEFAULT_PROMPTS, height=300)
prompts_list = [p.strip() for p in prompts_text.split('\n') if p.strip()]

def run_analysis():
    all_results = []
    
    # Progress Bar configuration
    total_steps = len(prompts_list) * (len(selected_openai_models) + len(selected_gemini_models))
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_count = 0

    # OpenAI Loop
    if openai_api_key and selected_openai_models:
        client = OpenAI(api_key=openai_api_key)
        for model in selected_openai_models:
            for prompt in prompts_list:
                status_text.text(f"Running OpenAI {model}: {prompt[:50]}...")
                
                # Determine clean question text
                if "Sentence:" in prompt:
                    question_text = prompt.split("Sentence:")[-1].strip()
                else:
                    question_text = prompt

                # Determine if reasoning model (simplified logic from notebook)
                is_reasoning = "o1-" in model or "gpt-5" in model
                
                responses = []
                # Batch logic simplified for Streamlit (sequential for clearer progress or small batches)
                # But notebook used batches. Let's do simple loop for iterations to populate list
                # Since 'iterations' might be small (e.g. 1), we loop.
                # Use notebook logic for reasoning vs standard if possible, but keep it robust.
                
                try:
                    for _ in range(iterations):
                        # Attempt Chat Completion first
                        try:
                            if is_reasoning:
                                # O1 and Preview models often use max_completion_tokens
                                completion = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_completion_tokens=500
                                )
                                responses.append(completion.choices[0].message.content.strip())
                            else:
                                # Standard Chat Models
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
                            # Catch specific error regarding "not a chat model"
                            error_str = str(e)
                            if "not a chat model" in error_str or "404" in error_str:
                                # Fallback to Legacy Completions Endpoint
                                status_text.text(f"Switching to Completions endpoint for {model}...")
                                completion = client.completions.create(
                                    model=model,
                                    prompt=prompt,
                                    max_tokens=500 if is_reasoning else 15,
                                    temperature=1.0
                                )
                                responses.append(completion.choices[0].text.strip())
                            else:
                                raise e # Re-raise other errors

                except Exception as e:
                    st.error(f"Error with OpenAI {model}: {e}")

                for ans in responses:
                    all_results.append({
                        "Prompt": prompt,
                        "Question": question_text,
                        "Model": f"OpenAI - {model}",
                        "Answer": ans
                    })
                
                step_count += 1
                if total_steps > 0:
                    progress_bar.progress(min(step_count / total_steps, 1.0))

    # Gemini Loop
    if gemini_api_key and selected_gemini_models:
        genai.configure(api_key=gemini_api_key)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        for model_name in selected_gemini_models:
            model_instance = genai.GenerativeModel(model_name)
            for prompt in prompts_list:
                status_text.text(f"Running Gemini {model_name}: {prompt[:50]}...")
                
                if "Sentence:" in prompt:
                    question_text = prompt.split("Sentence:")[-1].strip()
                else:
                    question_text = prompt
                
                responses = []
                for i in range(iterations):
                    try:
                        # Simple retry logic (simplified from notebook for brevity in app)
                        response = model_instance.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=100, # Reduced from 1500 as we expect short answers
                                temperature=1.0
                            ),
                            safety_settings=safety_settings
                        )
                        if response.candidates and response.candidates[0].content.parts:
                             responses.append(response.candidates[0].content.parts[0].text.strip())
                    except Exception as e:
                        # Log but don't stop everything
                        # print(f"Gemini error: {e}")
                        pass
                        time.sleep(1) # Basic backoff

                for ans in responses:
                    all_results.append({
                        "Prompt": prompt,
                        "Question": question_text,
                        "Model": f"Gemini - {model_name}",
                        "Answer": ans
                    })

                step_count += 1
                if total_steps > 0:
                    progress_bar.progress(min(step_count / total_steps, 1.0))

    progress_bar.empty()
    status_text.text("Analysis Complete!")
    return pd.DataFrame(all_results)

# --- Run Button ---
if st.button("Run Analysis", type="primary"):
    if not (openai_api_key or gemini_api_key):
        st.error("Please provide at least one API Key.")
    elif not (selected_openai_models or selected_gemini_models):
        st.error("Please select at least one model.")
    else:
        with st.spinner("Running analysis..."):
            df_results = run_analysis()
            if not df_results.empty:
                st.session_state['results_df'] = df_results
                # Trigger word cloud generation immediately
                st.session_state['wordclouds'] = {} # Reset
            else:
                st.warning("No results generated. Check API keys and limits.")

# --- Visualizations & Logic ---
if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
    df = st.session_state['results_df']
    
    st.divider()
    st.subheader("Results")
    st.dataframe(df)

    # Prepare Word Clouds
    # We need to compute them and store them to display
    
    st.subheader("Word Clouds")
    
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update([
        "output", "exactly", "one", "brand", "name", "do", "not", "more", "than", "only",
        "is", "a", "an", "the", "for", "in", "of", "question", "sentence", "complete"
    ])

    # Group by Question and Model
    grouped = df.groupby(['Question', 'Model'])
    
    # Structure for display: One row per Question, columns for Models?
    # Or just list them. A grid might be nice.
    # Let's organize by Question for easy comparison.
    
    unique_questions = df['Question'].unique()
    
    generated_images = {} # Key: (model, question_slug), Value: bytes
    
    for question in unique_questions:
        st.markdown(f"### {question}")
        models_for_q = df[df['Question'] == question]['Model'].unique()
        
        cols = st.columns(len(models_for_q)) if len(models_for_q) > 0 else [st.container()]
        
        for idx, model in enumerate(models_for_q):
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
                    wc = WordCloud(
                        width=400, height=300, background_color='white',
                        stopwords=custom_stopwords, max_words=50, normalize_plurals=False
                    ).generate_from_frequencies(phrase_counts)
                    
                    # Convert to image for st
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(f"{model}", fontsize=10)
                    
                    # Save to buffer for download
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    image_bytes = buf.getvalue()
                    
                    # Store for ZIP
                    safe_q = str(question)[:30].replace(" ", "_").replace("?", "")
                    safe_model = str(model).replace(" ", "_")
                    filename = f"cloud_{safe_model}_{safe_q}.png"
                    generated_images[filename] = image_bytes
                    
                    # Display
                    if idx < len(cols):
                        cols[idx].pyplot(fig)
                    plt.close(fig)
                else:
                    if idx < len(cols):
                        cols[idx].info(f"Not enough clean data for {model}")
            else:
                 if idx < len(cols):
                    cols[idx].info(f"No valid answers for {model}")

    # --- ZIP Download ---
    st.divider()
    
    def create_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add Main CSV
            zip_file.writestr("all_results.csv", df.to_csv(index=False))
            
            # Add Summary CSV (Question | Mentions)
            try:
                summary_dfs = []
                for q in unique_questions:
                    sub = df[df['Question'] == q]
                    answers = [normalize_brand(clean_text(a)) for a in sub['Answer'] if str(a).lower() not in ['nan', 'none']]
                    answers = [a for a in answers if a not in custom_stopwords and a]
                    counts = Counter(answers)
                    d_temp = pd.DataFrame(counts.most_common(), columns=[q, "Mentions"])
                    d_temp[" "] = "" # Spacer
                    summary_dfs.append(d_temp)
                
                if summary_dfs:
                    # Concatenate side-by-side like in notebook
                    df_summary = pd.concat(summary_dfs, axis=1)
                    zip_file.writestr("summary_results.csv", df_summary.to_csv(index=False))
            except Exception as e:
                # Fallback if concat fails or other issue
                zip_file.writestr("summary_error.txt", f"Could not generate summary csv: {str(e)}")

            # Add Images
            for fname, img_data in generated_images.items():
                zip_file.writestr(f"images/{fname}", img_data)
                
        return zip_buffer.getvalue()

    if st.button("Prepare Download"):
        zip_data = create_zip()
        st.download_button(
            label="Download All Results (.zip)",
            data=zip_data,
            file_name="competitive_analysis_results.zip",
            mime="application/zip"
        )
