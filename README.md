# LinkedIn Content Creator AI

An AI agent to automate and optimize LinkedIn content creation for Archit Anand by scraping posts, analyzing trends, generating AI-powered posts, and refining based on user feedback.

## Overview

This tool fulfills the task to build a LinkedIn content creator AI agent by:
- **Scraping**: Collects posts from Archit Anand and competitors (Aaron Golbin, Robert Schöne, Jaspar Carmichael-Jack).
- **Analyzing**: Identifies engagement trends (topics, tones, CTAs, posting times).
- **Generating**: Creates 3 AI-powered posts using Llama (via Groq), tailored to Archit’s style with user keywords.
- **Refining**: Incorporates user feedback to improve future posts.
- **UI**: Provides a Streamlit interface for post generation, feedback, and trend visualization.
- **Bonus**: Suggests optimal posting times (Tuesday at 15:00) and includes a simple UI.

## Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/LinkedIn-Content-Creator-AI.git
   cd LinkedIn-Content-Creator-AI
   ```
2. **Create Virtual Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # Windows: env\Scripts\activate
   ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. Configure Environment
Copy .env.example to .env and add your LinkedIn credentials and Groq API key

## Usage

1. **Scrape Posts:**
    ```bash
    python scraper.py
    ```
    - **Outputs**: data/posts.csv
    - **Features**: Scrapes posts from specified profiles, Handles cookies, retries, and LinkedIn’s dynamic loading. Saves error pages for debugging 

2. **Analyze Trends:**
    ```bash
    python analyzer.py
    ```
    - **Features:** Analyzes posts for engagement, topics, hashtags, and posting times.
    - **Outputs:** data/trends.csv

3. **Generate Posts:**
    ```bash
    python generator.py
    ```
    - **Features:** Prompts for keywords (e.g., “AI marketing”) or uses trends.
    - **Outputs:** data/generated_posts.csv (trend-based) or data/generated_posts_user.csv (keyword-based)


## Streamlit UI
### Run App:
    ```bash
    streamlit run app.py
    ```
    Open http://localhost:8501.

# Features
    - **Create Posts:** Enter keywords (optional) and generate 3 posts optimized for Archit Anand’s audience.
    - **Submit Feedback:** Provide feedback on posts (tone, CTA, content) to improve future generations.
    - **Refine Posts:** Regenerate posts incorporating all feedback.
    - **View Trends:** Visualize engagement by day, hour, and hashtags.

# Workflow
1. **Scrape:** Run scraper.py to collect posts from profiles (data/posts.csv).
2. **Analyze:** Run analyzer.py to generate trends (data/trends.csv).
3. **Generate:** Use generator.py (CLI) or app.py (UI) to create posts, optionally with keywords.
4. **Feedback:** Submit feedback via UI to refine posts.
5. **Refine:** Regenerate posts with feedback for improved alignment.
6. **Schedule:** Review data/scheduled_posts.csv for suggested posting times.

# Results: 
Sample Trends
Top Hashtags (Archit): #PerformanceMarketing, #AI, #Marketing
Top Topics: AI, performance marketing, ad optimization
Best Posting Time: Tuesday at 15:00 (highest engagement)
Tone: Positive tone outperforms negative
CTA: Subtle CTAs (e.g., “Comment below”) drive more engagement