# app.py
import streamlit as st
import pandas as pd
import json
import os
import logging
from datetime import datetime
from groq import Groq
import plotly.express as px
# Make sure generator functions needed by app are importable
from generator import load_trends, generate_posts, save_posts, load_feedback, save_feedback, apply_feedback

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Groq client
try:
    # Ensure the API key is loaded (e.g., from .env via load_dotenv or Streamlit secrets)
    # from dotenv import load_dotenv
    # load_dotenv() # Uncomment if you use .env file and run app locally
    groq_api_key = os.environ.get("GROQ_API_KEY") # Use Streamlit secrets in deployment
    if not groq_api_key:
        # Fallback for local testing if needed
        try:
             from dotenv import load_dotenv
             load_dotenv()
             groq_api_key = os.getenv("GROQ_API_KEY")
        except ImportError:
             pass

    if not groq_api_key:
         raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your environment or Streamlit secrets.")
    client = Groq(api_key=groq_api_key)

except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Streamlit app configuration
st.set_page_config(page_title="LinkedIn Content Creator AI", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .post-box {
        border: 2px solid #e6e6e6; /* Light gray border */
        border-radius: 10px;
        padding: 15px; /* Add padding inside the box */
        margin-bottom: 15px; /* Space below the box */
        background-color: #2b2b2b; /* Darker background for contrast */
    }
    .post-title {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
        color: #FFFFFF;  /* White title */
    }
    .post-content {
        color: #FFFFFF;  /* White text for post content */
        white-space: pre-wrap; /* Preserve line breaks from the post */
        margin-bottom: 10px; /* Space above copy button if needed */
    }
    .post-meta {
        color: #CCCCCC; /* Lighter gray for metadata */
        font-size: 0.9em;
        margin-top: 10px; /* Space above metadata */
    }
    .explanation-section {
        color: #B0B0B0; /* Slightly dimmer gray for explanation */
        font-size: 0.95em;
        margin-top: 5px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 2px solid #555555; /* Add a left border for visual separation */
    }
    .error-message {
        color: #FF6B6B; /* Reddish color for errors */
        font-weight: bold;
        margin-top: 5px;
    }
    .copy-button-container {
         margin-bottom: 10px; /* Space below copy button */
    }
    /* Style the button itself if needed */
    .stButton>button {
        /* Example: background-color: #4CAF50; color: white; */
    }
    </style>
""", unsafe_allow_html=True)

def get_descriptive_post_name(post_text, index):
    """Generate a descriptive name for the post based on content."""
    if not isinstance(post_text, str):
        return f"Post {index+1}: Invalid Content"
    keywords = ["AI", "Marketing", "Scaling", "Ads", "Creative", "Performance", "Growth", "DTC"]
    for keyword in keywords:
        if keyword.lower() in post_text.lower():
            return f"Post {index+1}: {keyword} Focus"
    # Fallback if no common keywords found
    first_words = " ".join(post_text.split()[:5])
    return f"Post {index+1}: {first_words}..." if first_words else f"Post {index+1}: Generated Insight"

def display_trends(trends_file="data/trends.csv"):
    # ... (Keep existing code, maybe add error handling for eval failures) ...
    try:
        trends = load_trends(trends_file) # Assumes load_trends handles eval safely
        st.subheader("Key Trends")

        # Raw trends
        with st.expander("View Raw Trends Data"):
            st.json(trends) # Display raw trends as JSON for clarity

        # Visualizations
        st.subheader("Trend Visualizations")
        col1, col2 = st.columns(2)

        # Engagement by Hour
        avg_hour_data = trends.get("avg_engagement_by_hour")
        if isinstance(avg_hour_data, dict) and avg_hour_data:
            with col1:
                df_hour = pd.DataFrame({"Hour": avg_hour_data.keys(), "Avg Engagement": avg_hour_data.values()})
                # Ensure Hour is treated correctly (e.g., as category or number)
                df_hour["Hour"] = df_hour["Hour"].astype(float).astype(int)
                df_hour = df_hour.sort_values("Hour")
                fig_hour = px.bar(df_hour, x="Hour", y="Avg Engagement", title="Avg Engagement by Hour")
                st.plotly_chart(fig_hour, use_container_width=True)
        else:
             with col1: st.warning("Engagement by Hour data missing or invalid.")

        # Engagement by Day
        avg_day_data = trends.get("avg_engagement_by_day")
        if isinstance(avg_day_data, dict) and avg_day_data:
            with col2:
                # Define order for days of the week
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                df_day = pd.DataFrame({"Day": avg_day_data.keys(), "Avg Engagement": avg_day_data.values()})
                # Convert Day to categorical type with defined order
                df_day['Day'] = pd.Categorical(df_day['Day'], categories=day_order, ordered=True)
                df_day = df_day.sort_values('Day')
                fig_day = px.bar(df_day, x="Day", y="Avg Engagement", title="Avg Engagement by Day")
                st.plotly_chart(fig_day, use_container_width=True)
        else:
             with col2: st.warning("Engagement by Day data missing or invalid.")

        # Top Hashtags (Overall)
        top_hashtags_data = trends.get("top_hashtags")
        if isinstance(top_hashtags_data, dict) and top_hashtags_data:
            with col1:
                df_hashtags = pd.DataFrame({"Hashtag": top_hashtags_data.keys(), "Count": top_hashtags_data.values()})
                fig_hashtags = px.pie(df_hashtags, names="Hashtag", values="Count", title="Overall Top Hashtags")
                st.plotly_chart(fig_hashtags, use_container_width=True)
        else:
            with col1: st.warning("Top Hashtags data missing or invalid.")

        # Archit’s Top Hashtags
        archit_hashtags_data = trends.get("archit_top_hashtags")
        if isinstance(archit_hashtags_data, dict) and archit_hashtags_data:
            with col2:
                df_archit_hashtags = pd.DataFrame({"Hashtag": archit_hashtags_data.keys(), "Count": archit_hashtags_data.values()})
                fig_archit_hashtags = px.pie(df_archit_hashtags, names="Hashtag", values="Count", title="Archit’s Top Hashtags")
                st.plotly_chart(fig_archit_hashtags, use_container_width=True)
        else:
            with col2: st.warning("Archit's Top Hashtags data missing or invalid.")

    except FileNotFoundError:
        st.error(f"{trends_file} not found. Please run `analyzer.py` or upload a `trends.csv` file via the sidebar.")
    except Exception as e:
        st.error(f"Error loading or displaying trends: {e}")
        logger.exception("Error during trend display:") # Log full traceback


def generate_and_display_posts(trends_file="data/trends.csv", regenerate_with_feedback=False):
    """Generate and display posts using structured JSON format."""
    try:
        trends = load_trends(trends_file)
        feedback = load_feedback() if regenerate_with_feedback else None # Load feedback only if regenerating

        button_label = "Regenerate Posts with Feedback" if regenerate_with_feedback else "Generate Posts"
        if st.button(button_label):
            st.session_state['generate_clicked'] = True # Flag that button was clicked

        # Proceed only if the button was clicked in this run
        if st.session_state.get('generate_clicked', False):
            with st.spinner("Generating posts... Please wait."):
                posts = generate_posts(trends, num_posts=3, feedback=feedback)

            # Reset the flag after generation
            st.session_state['generate_clicked'] = False

            if not posts:
                st.error("No posts were generated. Check the logs or Groq API status.")
                return None # Return None if generation failed

            st.subheader("Generated Posts" + (" (with feedback)" if regenerate_with_feedback else ""))

            valid_posts = []
            for i, post in enumerate(posts):
                post_text = post.get("post_text")
                explanation = post.get("explanation", "No explanation provided.")
                error = post.get("error")
                post_id = post.get("post_id", f"unknown_{i}")

                if error:
                    st.error(f"Error generating Post {i+1} ({post_id}): {error}")
                    # Optionally display raw text if available despite error
                    if post_text:
                         st.warning(f"Raw content received for Post {i+1}:\n```\n{post_text}\n```")
                    st.markdown("---")
                    continue # Skip to next post if there was a critical error

                if not post_text:
                    st.warning(f"Post {i+1} ({post_id}) has no content. LLM might have failed.")
                    st.markdown(f"<div class='explanation-section'>Explanation: {explanation}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    continue # Skip if no post text

                valid_posts.append(post) # Add to list of valid posts for feedback/download
                post_name = get_descriptive_post_name(post_text, i)

                # Prepare post text for display and copying (escape backticks for JS)
                # Use json.dumps for robust escaping
                escaped_post_text_for_js = json.dumps(post_text)

                # --- Display Structure ---
                st.markdown(f"""
                    <div class="post-box">
                        <div class="post-title">{post_name} (Suggested: {post.get('suggested_posting_day','N/A')} at {post.get('suggested_posting_hour','N/A')}:00)</div>
                        <div class="post-content">{post_text}</div>
                        <div class="copy-button-container">
                            <button onclick="copyToClipboard_{i}()">Copy Post Text</button>
                        </div>
                    </div>
                    <script>
                    function copyToClipboard_{i}() {{
                        navigator.clipboard.writeText({escaped_post_text_for_js})
                            .then(() => {{
                                alert('Post text copied to clipboard!');
                            }})
                            .catch(err => {{
                                console.error('Failed to copy text: ', err);
                                alert('Failed to copy text. See console for details.');
                            }});
                    }}
                    </script>
                """, unsafe_allow_html=True)

                # Display Explanation and Metadata BELOW the box
                st.markdown(f"<div class='explanation-section'><b>LLM Explanation:</b> {explanation}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='post-meta'>Generated at: {post.get('generated_at', 'N/A')} | ID: {post_id}</div>", unsafe_allow_html=True)
                st.markdown("---") # Separator between posts
            # --- End Display Structure ---

            if valid_posts:
                # Save posts and provide download link
                output_filename = f"data/generated_posts{'_with_feedback' if regenerate_with_feedback else ''}.csv"
                save_posts(valid_posts, output_filename)
                try:
                    with open(output_filename, "rb") as f:
                        st.download_button(
                            label=f"Download {'Regenerated' if regenerate_with_feedback else 'Generated'} Posts (CSV)",
                            data=f,
                            file_name=os.path.basename(output_filename),
                            mime="text/csv",
                        )
                except FileNotFoundError:
                    st.error(f"Could not find file {output_filename} for download.")
                return valid_posts # Return the successfully generated posts
            else:
                st.warning("No valid posts were generated successfully.")
                return None
        return None # Return None if button wasn't clicked
    except FileNotFoundError:
        st.error(f"{trends_file} not found. Please run `analyzer.py` or upload a `trends.csv` file via the sidebar.")
        logger.error(f"Trends file {trends_file} not found.")

def collect_and_apply_feedback(posts_for_feedback):
    """Collect feedback via Streamlit interface and trigger regeneration."""
    st.subheader("Provide Feedback to Refine Posts")

    if not posts_for_feedback:
        st.warning("No posts available to provide feedback on. Please generate posts first.")
        # Check if there are previous posts in session state
        if 'latest_posts' in st.session_state and st.session_state['latest_posts']:
             st.info("Showing feedback options for the last generated set of posts.")
             posts_for_feedback = st.session_state['latest_posts']
        else:
             return # Exit if truly no posts available

    # Ensure posts are in the expected format (list of dicts)
    if not isinstance(posts_for_feedback, list) or not all(isinstance(p, dict) for p in posts_for_feedback):
        st.error("Invalid format for posts data in session state.")
        logger.error(f"Invalid posts_for_feedback type: {type(posts_for_feedback)}")
        return

    # Filter out posts that had errors or no text
    valid_posts_for_feedback = [p for p in posts_for_feedback if p.get("post_text") and not p.get("error")]

    if not valid_posts_for_feedback:
        st.warning("No valid posts from the last generation to provide feedback on.")
        return

    post_options = {get_descriptive_post_name(p["post_text"], i): p["post_id"] for i, p in enumerate(valid_posts_for_feedback)}
    selected_post_name = st.selectbox("Select Post to Provide Feedback On", options=list(post_options.keys()))

    if selected_post_name:
        post_id = post_options[selected_post_name]
        st.markdown(f"**Providing feedback for:** {selected_post_name} (`{post_id}`)")

        # Display the selected post for context
        selected_post_data = next((p for p in valid_posts_for_feedback if p["post_id"] == post_id), None)
        if selected_post_data:
             with st.expander("Show selected post content"):
                 st.markdown(f"```\n{selected_post_data['post_text']}\n```")
                 st.markdown(f"**LLM Explanation:** {selected_post_data.get('explanation', 'N/A')}")


        # Feedback fields
        tone = st.text_input("Tone Feedback (e.g., 'make it more urgent', 'more empathetic')", key=f"tone_{post_id}")
        cta = st.text_input("CTA Feedback (e.g., 'change CTA to link to signup', 'remove CTA')", key=f"cta_{post_id}")
        content = st.text_area("Content Feedback (e.g., 'add a stat about X', 'mention Y feature', 'make it shorter')", key=f"content_{post_id}")
        other = st.text_input("Other Feedback (e.g., 'use fewer emojis', 'target audience is Z')", key=f"other_{post_id}")

        if st.button("Submit Feedback & Regenerate All Posts", key=f"submit_feedback_{post_id}"):
            user_feedback = {k: v for k, v in {"tone": tone, "cta": cta, "content": content, "other": other}.items() if v}

            if user_feedback:
                # Apply feedback using the function from generator.py
                updated_global_feedback = apply_feedback(post_id, user_feedback) # This saves to feedback.json
                st.success(f"Feedback saved for {selected_post_name}! Regenerating posts with all accumulated feedback...")
                st.session_state['feedback_submitted'] = True # Flag feedback submission

                # Trigger regeneration by calling the main display function with flag
                # The actual regeneration happens in the 'Generate Posts' section logic now
                # We just need to ensure the feedback file is updated before the next generation cycle.

            else:
                st.warning("No feedback provided in the fields.")

    # This section will now run if the "Regenerate" button in the main generation function is clicked
    # and the `regenerate_with_feedback` flag is true.
    # No separate regeneration button needed here anymore.


def main():
    st.title("LinkedIn Content Creator AI")
    st.markdown("Generate engaging LinkedIn posts for Archit Anand using AI, tailored to audience trends and refined with your feedback.")

    # Initialize session state variables if they don't exist
    if 'latest_posts' not in st.session_state:
        st.session_state['latest_posts'] = []
    if 'generate_clicked' not in st.session_state:
        st.session_state['generate_clicked'] = False
    if 'feedback_submitted' not in st.session_state:
        st.session_state['feedback_submitted'] = False


    # Sidebar
    st.sidebar.title("Navigation & Upload")
    page = st.sidebar.radio("Go to", ["Generate & Refine", "View Trends"])
    uploaded_file = st.sidebar.file_uploader("Upload new `trends.csv` (Optional)", type="csv")

    # Handle uploaded trends.csv
    trends_file = "data/trends.csv"
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        try:
            with open(trends_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("Uploaded and using new `trends.csv`")
            # Clear potentially outdated trends from cache/memory if needed
        except Exception as e:
            st.sidebar.error(f"Failed to save uploaded file: {e}")


    # Page Routing
    if page == "View Trends":
        display_trends(trends_file)

    elif page == "Generate & Refine":
        st.header("1. Generate Initial Posts")
        st.markdown("Click the button below to generate posts based on the latest trends and any saved feedback.")
        # Generate posts (button is inside the function)
        generated_posts = generate_and_display_posts(trends_file, regenerate_with_feedback=False)
        if generated_posts:
            st.session_state['latest_posts'] = generated_posts # Store the latest valid posts

        st.header("2. Provide Feedback (Optional)")
        st.markdown("Select a post from the last generation, provide feedback, and submit it. The feedback will be used the *next* time you click 'Generate Posts' or 'Regenerate Posts'.")
        collect_and_apply_feedback(st.session_state.get('latest_posts', []))

        st.header("3. Regenerate with Feedback (Optional)")
        st.markdown("Click this button to generate a new set of posts, incorporating *all* feedback submitted so far.")
        regenerated_posts = generate_and_display_posts(trends_file, regenerate_with_feedback=True)
        if regenerated_posts:
            st.session_state['latest_posts'] = regenerated_posts # Update latest posts


if __name__ == "__main__":
    # Ensure data directory exists at startup
    os.makedirs("data", exist_ok=True)
    main()