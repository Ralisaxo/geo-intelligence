Here is a comprehensive overview of the different sections and tools available within the Saxo GEO Command Center. 

## 📊 Overview
The Overview section is your home base. It provides a high-level summary of the data currently loaded into the system, including how many competitors are being tracked, Saxo's overall Authority Score, and a quick visual ranking of top competitors based on how often they appear in AI and search results.
<!-- ADVANCED -->
**How it works**: The Dashboard calculates metrics dynamically from the loaded competitor dataset.
**Metrics Calculated**: 
- **Saxo Auth Score**: Measures Saxo Bank's prominence in the data.
- **Top 5 Visual Authority**: Sorts the pandas DataFrame by the `Source_Authority_Score` property.

## 🔎 Knowledge Management
This section lets you deep-dive into the raw data backing our insights. It includes a Data Explorer to filter and view all underlying metrics, and an AI Strategist that automatically identifies semantic gaps between Saxo Bank and competitors. It also includes a Reality Check tool that simulates how an AI natively views a company before applying any internet search data.
<!-- ADVANCED -->
**APIs Used**: Wikidata Query Service, Google Knowledge Graph API, OpenAI API (GPT-4o/GPT-5.2).
**How it works**:
- **Data Explorer / Google KG**: Directly displays the merged data from Wikidata and the Google Knowledge Graph.
- **AI Strategist**: Feeds the aggregated dataframe directly to OpenAI models to identify structural and semantic gaps in Knowledge Graph coverage.
- **RAG Simulation**: Forces GPT-4o to write a company bio based explicitly *only* on the fetched structured data, completely bypassing its internal training memory.

## 📐 Semantic Triples
Semantic Triples helps us understand brand alignment. You can map out how closely Saxo Bank (or any competitor) aligns with specific brand adjectives like "Professional," "Innovative," or "Expensive." The AI Word Cloud and Semantic Positioning modules visually map these concepts so you can see where our brand sits in relation to market expectations.
<!-- ADVANCED -->
**APIs Used**: Google Gemini API.
**How it works**:
- **Semantic Alignment Lab**: Computes the semantic vector compatibility (cosine similarity) between a core statement and diagnostic concepts using Gemini's embeddings. Raw vectors are reduced dimensionally so they can be plotted onto an interactive 2D coordinate space (Altair chart). Sentiment analysis is also layered on top to color-code positive or negative brand associations.

## 💬 Reddit Analysis
This tool taps into community sentiment by analyzing discussions on Reddit. It gathers mentions and threads related to our brand or industry, then summarizes the overall market sentiment so you can hear directly from retail and institutional traders in their own words.
<!-- ADVANCED -->
**APIs Used**: Reddit Data / Apify Scraper (or standard fetch), OpenAI API (GPT-4o).
**How it works**: Raw threads and Reddit posts are aggregated and passed to GPT-4o with a custom-crafted prompt (`prompts/geo_analysis_system.txt`). The LLM extracts the prevailing market sentiment and condenses thousands of user comments into a digestible summary.

## 🤖 LLM Monitoring
See exactly how AI models like ChatGPT and Perplexity are forming their answers about the brokerage market. This section includes cross-market analysis and tracks which domains the LLMs cite most frequently (Source Extraction & Source Trends) when answering user prompts. It also allows you to track brands' competitive vector across visibility and sentiment over time.
- **Competitor Scraper**: Automatically identifies and tracks new competitors appearing in our semantic space.
<!-- ADVANCED -->
**APIs Used**: AccuRanker API (for accurate domain rank extraction and citation tracking).
**How it works**:
- **Source Extraction & Trends**: Pulls data using the AccuRanker API to analyze which domains are gaining or losing visibility in LLM-generated responses over custom time periods (e.g., rolling 7-day averages). Displays visual scorecards and bar charts to measure exact share-of-voice differences.
- **Competitor Scraper**: Queries the AccuRanker API to fetch currently tracked competitors and cross-references them against newly discovered domains.

## 🛠️ AI Powered Tools & Random Tools
A collection of supplementary utilities for deep analysis.
- **AI FlexSheet**: A flexible workspace that lets you run custom AI prompts across rows of data for bulk processing and categorization.
- **URL Classifier**: Categorizes lists of URLs quickly for SEO and tracking purposes.
- **GSC BigQuery Visualizer**: Analyzes your Google Search Console exports from BigQuery to track clicks and impressions, segmenting them by market and traffic origin.
<!-- ADVANCED -->
**How it works**:
- **AI FlexSheet**: Combines user-uploaded datasets with the OpenAI API, mapping a custom prompt row-by-row to automate large-scale text analysis tasks.
- **GSC BigQuery Visualizer**: Intakes a CSV export, processes it robustly to handle encodings, runs the URL classifier script, and generates interactive charts locally (ensuring data privacy by avoiding external servers).
