
import requests
import json
import toml

# Load secrets
try:
    secrets = toml.load(".streamlit/secrets.toml")
    api_token = secrets["ACCURANKER_TOKEN"]
except Exception as e:
    print(f"Error loading secrets: {e}")
    exit()

brand_id = 10000084 # Saxo CH from previous debug logs

def test_fetch(fields_param):
    print(f"\n--- Testing fields='{fields_param}' ---")
    url = f"https://app.accuranker.com/api/v4/brands/{brand_id}/prompts/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Accept": "application/json",
    }
    
    params = {"limit": 5}
    if fields_param:
        params["fields"] = fields_param

    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            prompts = data if isinstance(data, list) else data.get('results', [])
            
            if not prompts:
                print("No prompts found.")
                return

            p = prompts[0]
            print(f"First Prompt Keys: {list(p.keys())}")
            print(f"Tags: {p.get('tags')}")
            
            results = p.get('results', [])
            if results:
                r = results[0]
                print(f"First Result Keys: {list(r.keys())}")
                sources = r.get('sources', [])
                if sources:
                    print(f"First Source: {sources[0]}")
                else:
                    print("No sources in first result.")
            else:
                print("No results in first prompt.")
                
    except Exception as e:
        print(f"Error: {e}")

# Test 1: No fields (Global)
test_fetch(None)

# Test 2: Explicit tags and results
test_fetch("id,tags,results")

# Test 3: Explicit tags and deep results
test_fetch("id,tags,results.created_at,results.sources")

# Test 5: Standard fields usually included
test_fetch("id,tags,results.created_at,results.sources.url,results.sources.competitor.domain")
