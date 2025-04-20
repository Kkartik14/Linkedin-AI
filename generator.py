import pandas as pd
from groq import Groq
import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Groq Client
try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=groq_api_key)
except ValueError as e:
    logger.error(f"Groq API Key error: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise


def load_trends(filename="data/trends.csv"):
    """Load trends from CSV, handling potential eval errors."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run analyzer.py first.")
    df = pd.read_csv(filename)
    trends = {}
    for _, row in df.iterrows():
        metric = row["metric"]
        value = row["value"]
        try:
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                 trends[metric] = eval(value)
            else:
                 trends[metric] = value
        except Exception as e:
            logger.warning(f"Could not evaluate trend value for {metric}: {value}. Keeping as string. Error: {e}")
            trends[metric] = value
    logger.info("Loaded trends from trends.csv")
    return trends

# Reverted get_prompt: Only uses trends and feedback
def get_prompt(trends, feedback=None):
    """Construct a prompt for post generation based ONLY on trends and feedback."""
    logger.info("Generating prompt based on trends and feedback.")

    if not trends:
        logger.warning("Trends data is missing or empty. Generating generic prompt.")
        # Fallback prompt if trends are missing
        prompt = """
        You are a professional LinkedIn content creator writing for Archit Anand, a marketing and AI expert at Fuelgrowth.
        Create a compelling LinkedIn post about AI in marketing or performance marketing.
        Use a professional and engaging tone. Include a subtle CTA. Keep it 500-1000 characters.
        Include relevant hashtags like #AI #Marketing #PerformanceMarketing.

        IMPORTANT: Generate the response as a JSON object ONLY with keys 'post_text' and 'explanation'.
        Example: {"post_text": "...", "explanation": "Generated generic post due to missing trend data."}
        """
        return prompt, None # No best days available

    # Extract trends safely
    top_hashtags = trends.get("archit_top_hashtags", {})
    top_topics = trends.get("archit_top_topics", {})
    avg_day_data = trends.get("avg_engagement_by_day", {})
    avg_tone_data = trends.get("avg_engagement_by_tone", {})
    avg_cta_data = trends.get("avg_engagement_by_has_cta", {})

    best_days = sorted(
        avg_day_data.items(), key=lambda item: item[1], reverse=True
    )[:2] if isinstance(avg_day_data, dict) else []

    tone = "positive"
    if isinstance(avg_tone_data, dict):
         positive_engagement = avg_tone_data.get("positive", 0)
         negative_engagement = avg_tone_data.get("negative", 0)
         tone = "positive" if positive_engagement >= negative_engagement else "negative"

    cta_preference = "subtle CTA (e.g., 'Comment your thoughts below')"
    if isinstance(avg_cta_data, dict):
         true_engagement = avg_cta_data.get(True, avg_cta_data.get('True', 0))
         false_engagement = avg_cta_data.get(False, avg_cta_data.get('False', 0))
         cta_preference = "explicit CTA (e.g., 'DM me for details')" if true_engagement > false_engagement else "subtle CTA (e.g., 'Comment your thoughts below')"

    # Prepare strings for prompt, handling empty dicts
    topics_str = ', '.join(list(top_topics.keys())) if isinstance(top_topics, dict) and top_topics else 'AI, marketing, performance'
    hashtags_str = ', '.join(list(top_hashtags.keys())) if isinstance(top_hashtags, dict) and top_hashtags else '#PerformanceMarketing, #AI, #Marketing'

    prompt = f"""
    You are a professional LinkedIn content creator writing for Archit Anand, a marketing and AI expert at Fuelgrowth.
    Analyze the provided trends and feedback (if any) to create a LinkedIn post that aligns with his style: professional, actionable, and focused on AI, marketing, and performance.

    Key insights from analysis:
    - High-engagement topics seem to be: {topics_str}.
    - Generally, a {tone} tone performs well.
    - Engagement suggests using a {cta_preference}.
    - Longer posts (500–1000 characters) tend to get more engagement.
    - Popular hashtags include: {hashtags_str}.
    - Archit’s style is insightful, data-driven, with a conversational touch.
    """

    # Adjust prompt based on feedback
    if feedback and isinstance(feedback, dict) and feedback:
         prompt += "\nIncorporate the following user feedback from previous generations:\n"
         all_feedback_str = ""
         for post_id, fb_items in feedback.items():
              if isinstance(fb_items, dict):
                  fb_details = "; ".join([f"{k}: {v}" for k, v in fb_items.items()])
                  all_feedback_str += f"- For posts similar to '{post_id[:10]}...': {fb_details}\n"
         if all_feedback_str:
            prompt += all_feedback_str
         else:
            logger.warning("Feedback provided but could not be processed into prompt string.")

    # Determine a focus topic for the explanation, default if needed
    focus_topic_for_explanation = list(top_topics.keys())[0] if isinstance(top_topics, dict) and top_topics else 'AI in Marketing'

    prompt += f"""
    Based on the analysis and feedback, generate ONE compelling LinkedIn post.

    Example style to emulate:
    "I turned ONE winning ad into 100 high-performing variations—without lifting a finger. Tired of wasting thousands testing creatives manually? At Fuelgrowth, our AI agent analyzes your ads, compares with competitors, and suggests optimizations on autopilot. Result? You scale fast, minus the chaos. Comment 'SUPERADS' below, and I’ll DM you a report! #PerformanceMarketing #MarketingAI #DTC"

    IMPORTANT: Generate the response as a JSON object ONLY. The JSON object must contain two keys:
    1.  `post_text`: A string containing the complete LinkedIn post text, including hashtags.
    2.  `explanation`: A string explaining the key choices made during generation or adjustments based on feedback (e.g., "Generated post focusing on {focus_topic_for_explanation} with a {tone} tone and {cta_preference}, incorporating feedback if provided.").

    Example JSON output format:
    {{
      "post_text": "The actual LinkedIn post content goes here...",
      "explanation": "Generated post focusing on AI in marketing with a positive tone and a subtle CTA, incorporating feedback about conversational style."
    }}

    Do not include any text before or after the JSON object.
    """
    return prompt, best_days


# Reverted generate_posts: No user_instructions parameter
def generate_posts(trends, num_posts=3, feedback=None): # Default back to 3 posts
    """Generate LinkedIn posts using Groq API based ONLY on trends and feedback."""
    num_posts = int(num_posts) # Ensure int
    logger.info(f"Generating {num_posts} post(s) based on trends and feedback.")
    prompt, best_days = get_prompt(trends, feedback) # Call reverted get_prompt

    suggested_day = "Tuesday" # Default
    if best_days and isinstance(best_days, list) and len(best_days) > 0 and isinstance(best_days[0], tuple):
         suggested_day = best_days[0][0]

    suggested_hour = 15 # Default hour
    avg_hour_data = trends.get("avg_engagement_by_hour", {}) if trends else {}
    if isinstance(avg_hour_data, dict) and avg_hour_data:
        try:
             valid_hours = {}
             for k, v in avg_hour_data.items():
                 try: valid_hours[float(k)] = v
                 except (ValueError, TypeError): logger.warning(f"Skipping invalid hour key '{k}' in trends.")
             if valid_hours: suggested_hour = int(max(valid_hours, key=valid_hours.get))
        except Exception as e:
             logger.warning(f"Could not determine best hour from trends: {e}. Using default {suggested_hour}.")
    elif not trends:
         logger.warning("Trends data unavailable for determining best hour.")

    if not prompt:
         logger.error("Failed to generate a valid prompt.")
         return []

    posts = []
    for i in range(num_posts):
        # Reverted post ID prefix
        post_id_prefix = 'trend'
        post_data = {
            "post_id": f"post_{post_id_prefix}_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "post_text": None,
            "explanation": None,
            "generated_at": datetime.now().isoformat(),
            "suggested_posting_day": suggested_day,
            "suggested_posting_hour": suggested_hour,
            "error": None
        }
        try:
            logger.info(f"Sending request to Groq API for post {i+1}/{num_posts} ({post_id_prefix})...")
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that strictly outputs JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            raw_response = chat_completion.choices[0].message.content.strip()
            logger.debug(f"Raw LLM Response {i+1}: {raw_response[:500]}...")

            # JSON Parsing Logic (remains the same)
            try:
                cleaned_response = re.sub(r"^```json\s*|\s*```$", "", raw_response, flags=re.MULTILINE | re.DOTALL).strip()
                json_start = cleaned_response.find('{')
                json_end = cleaned_response.rfind('}')
                if json_start != -1 and json_end != -1: json_string = cleaned_response[json_start:json_end+1]
                else: json_string = cleaned_response
                parsed_json = json.loads(json_string)

                if isinstance(parsed_json, dict) and "post_text" in parsed_json and "explanation" in parsed_json:
                    post_data["post_text"] = parsed_json.get("post_text")
                    post_data["explanation"] = parsed_json.get("explanation")
                    logger.info(f"Successfully parsed JSON and generated post {i+1} ({post_id_prefix})")
                else:
                     raise ValueError("Parsed JSON missing required keys ('post_text', 'explanation')")

            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON response for post {i+1}: {json_e}")
                logger.error(f"Cleaned response was: {cleaned_response[:500]}...")
                post_data["error"] = f"JSON Parsing Error: {json_e}"
                post_data["post_text"] = raw_response
                post_data["explanation"] = "Error: LLM did not return valid JSON conforming to structure."
            except ValueError as val_e:
                logger.error(f"Parsed JSON validation error for post {i+1}: {val_e}")
                post_data["error"] = str(val_e)
                post_data["post_text"] = raw_response
                post_data["explanation"] = "Error: JSON structure invalid (missing keys)."

        except Exception as e:
            logger.error(f"Error generating post {i+1} via Groq API: {e}", exc_info=True)
            post_data["error"] = f"API Error: {e}"
            post_data["explanation"] = f"Error during generation: {e}"

        posts.append(post_data)

    return posts

# --- save_posts, load_feedback, save_feedback, apply_feedback ---
# These utility functions remain the same as the previous version
# (Ensure they are present and correct as provided in the previous full code)

def save_posts(posts, filename="data/generated_posts.csv"):
    """Save generated posts to CSV."""
    if not posts:
        logger.warning("No posts to save.")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        df = pd.DataFrame(posts)
        expected_cols = ["post_id", "post_text", "explanation", "generated_at", "suggested_posting_day", "suggested_posting_hour", "error"]
        for col in expected_cols:
            if col not in df.columns: df[col] = None
        df = df[expected_cols]
        df.to_csv(filename, index=False, na_rep='NULL')
        logger.info(f"Saved {len(posts)} posts to {filename}")
    except Exception as e:
        logger.error(f"Failed to save posts to CSV {filename}: {e}", exc_info=True)


def load_feedback(filename="data/feedback.json"):
    """Load feedback from JSON."""
    feedback = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f: feedback = json.load(f)
            if not isinstance(feedback, dict):
                 logger.warning(f"Feedback file {filename} does not contain a valid JSON object. Resetting feedback.")
                 return {}
        except json.JSONDecodeError:
             logger.error(f"Error reading or parsing feedback file {filename}. Returning empty feedback.", exc_info=True)
             return {}
        except Exception as e:
             logger.error(f"Unexpected error loading feedback file {filename}: {e}", exc_info=True)
             return {}
    return feedback

def save_feedback(feedback, filename="data/feedback.json"):
    """Save feedback to JSON."""
    if not isinstance(feedback, dict):
        logger.error("Attempted to save invalid feedback (not a dictionary). Aborting save.")
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "w") as f: json.dump(feedback, f, indent=2)
        logger.info(f"Saved feedback to {filename}")
    except Exception as e:
        logger.error(f"Failed to save feedback to {filename}: {e}", exc_info=True)


def apply_feedback(post_id, user_feedback, feedback_file="data/feedback.json"):
    """Load existing feedback, add/update feedback for a post_id, and save."""
    if not isinstance(user_feedback, dict) or not post_id:
        logger.error("Invalid input for applying feedback.")
        return None
    feedback = load_feedback(feedback_file)
    feedback[post_id] = user_feedback
    save_feedback(feedback, feedback_file)
    logger.info(f"Applied and saved feedback for post {post_id}.")
    return feedback


# (Optional) CLI testing function - remains similar but calls the reverted generate_posts
def main_cli():
    logger.info("Running Generator in CLI mode (Reverted - Trend/Feedback Only).")
    os.makedirs("data", exist_ok=True)
    try:
        trends = load_trends()
    except FileNotFoundError:
        logger.error("trends.csv not found. Run analyzer.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading trends: {e}")
        return

    feedback = load_feedback()

    print("\nGenerating posts based on trends & feedback...")
    trend_posts = generate_posts(trends=trends, num_posts=3, feedback=feedback) # Call reverted function
    if trend_posts:
        save_posts(trend_posts, "data/generated_posts_cli_reverted.csv")
        # CLI Feedback collection can still be done here if needed
        # from generator import collect_feedback_cli
        # fb = collect_feedback_cli(trend_posts)
        # if fb:
        #      for post_id, feedback_data in fb.items():
        #          apply_feedback(post_id, feedback_data)


if __name__ == "__main__":
    main_cli()