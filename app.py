import streamlit as st
import pandas as pd
import json
import os
import logging
from datetime import datetime
from groq import Groq
import plotly.express as px
from generator import load_trends, generate_posts, save_posts, load_feedback, save_feedback, apply_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        from dotenv import load_dotenv
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

st.set_page_config(page_title="LinkedIn Content Creator AI", layout="wide")

st.markdown("""
    <style>
    .post-box {
        border: 2px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #2b2b2b;
    }
    .post-title {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
        color: #FFFFFF;
    }
    .post-content {
        color: #FFFFFF;
        white-space: pre-wrap;
        margin-bottom: 10px;
    }
    .post-meta {
        color: #CCCCCC;
        font-size: 0.9em;
        margin-top: 10px;
    }
    .explanation-section {
        color: #B0B0B0;
        font-size: 0.95em;
        margin-top: 5px;
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 2px solid #555555;
    }
    .error-message {
        color: #FF6B6B;
        font-weight: bold;
        margin-top: 5px;
    }
    .copy-button-container {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def get_descriptive_post_name(post_text, index):
    if not isinstance(post_text, str):
        return f"Post {index+1}: Invalid Content"
    keywords = ["AI", "Marketing", "Scaling", "Ads", "Creative", "Performance", "Growth", "DTC"]
    for keyword in keywords:
        if keyword.lower() in post_text.lower():
            return f"Post {index+1}: {keyword} Focus"
    first_words = " ".join(post_text.split()[:5])
    return f"Post {index+1}: {first_words}..." if first_words else f"Post {index+1}: Generated Insight"

def display_trends(trends_file="data/trends.csv"):
    try:
        trends = load_trends(trends_file)
        st.subheader("Audience Engagement Trends")
        with st.expander("View Raw Trends Data"):
            st.json(trends)
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        avg_hour_data = trends.get("avg_engagement_by_hour")
        if isinstance(avg_hour_data, dict) and avg_hour_data:
            with col1:
                df_hour = pd.DataFrame({"Hour": avg_hour_data.keys(), "Avg Engagement": avg_hour_data.values()})
                df_hour["Hour"] = df_hour["Hour"].astype(float).astype(int)
                df_hour = df_hour.sort_values("Hour")
                fig_hour = px.bar(df_hour, x="Hour", y="Avg Engagement", title="Engagement by Hour")
                st.plotly_chart(fig_hour, use_container_width=True)
        else:
            with col1:
                st.warning("Engagement by Hour data unavailable.")
        avg_day_data = trends.get("avg_engagement_by_day")
        if isinstance(avg_day_data, dict) and avg_day_data:
            with col2:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                df_day = pd.DataFrame({"Day": avg_day_data.keys(), "Avg Engagement": avg_day_data.values()})
                df_day['Day'] = pd.Categorical(df_day['Day'], categories=day_order, ordered=True)
                df_day = df_day.sort_values('Day')
                fig_day = px.bar(df_day, x="Day", y="Avg Engagement", title="Engagement by Day")
                st.plotly_chart(fig_day, use_container_width=True)
        else:
            with col2:
                st.warning("Engagement by Day data unavailable.")
        top_hashtags_data = trends.get("top_hashtags")
        if isinstance(top_hashtags_data, dict) and top_hashtags_data:
            with col1:
                df_hashtags = pd.DataFrame({"Hashtag": top_hashtags_data.keys(), "Count": top_hashtags_data.values()})
                fig_hashtags = px.pie(df_hashtags, names="Hashtag", values="Count", title="Overall Top Hashtags")
                st.plotly_chart(fig_hashtags, use_container_width=True)
        else:
            with col1:
                st.warning("Top Hashtags data unavailable.")
        archit_hashtags_data = trends.get("archit_top_hashtags")
        if isinstance(archit_hashtags_data, dict) and archit_hashtags_data:
            with col2:
                df_archit_hashtags = pd.DataFrame({"Hashtag": archit_hashtags_data.keys(), "Count": archit_hashtags_data.values()})
                fig_archit_hashtags = px.pie(df_archit_hashtags, names="Hashtag", values="Count", title="Archit’s Top Hashtags")
                st.plotly_chart(fig_archit_hashtags, use_container_width=True)
        else:
            with col2:
                st.warning("Archit's Top Hashtags data unavailable.")
    except FileNotFoundError:
        st.error(f"{trends_file} not found. Run `analyzer.py` or upload a `trends.csv` file via the sidebar.")
    except Exception as e:
        st.error(f"Error displaying trends: {e}")
        logger.exception("Error during trend display")

def generate_and_display_posts(trends_file="data/trends.csv", regenerate_with_feedback=False, key_prefix="initial"):
    try:
        trends = load_trends(trends_file)
        feedback = load_feedback() if regenerate_with_feedback else None
        button_label = "Generate New Posts" if not regenerate_with_feedback else "Refine Posts with Feedback"
        if st.button(button_label, key=f"{key_prefix}_generate_button"):
            st.session_state['generate_clicked'] = True
            st.session_state['current_key_prefix'] = key_prefix
        if st.session_state.get('generate_clicked', False) and st.session_state.get('current_key_prefix') == key_prefix:
            with st.spinner("Generating posts..."):
                user_keywords = st.session_state.get('user_keywords')
                posts = generate_posts(trends, num_posts=3, feedback=feedback, user_keywords=user_keywords)
            st.session_state['generate_clicked'] = False
            if not posts:
                st.error("No posts generated. Check logs or Groq API status.")
                return None
            st.subheader(f"{'Refined' if regenerate_with_feedback else 'New'} Posts")
            if user_keywords:
                st.markdown(f"**Focus Keywords:** {user_keywords}")
            valid_posts = []
            for i, post in enumerate(posts):
                post_text = post.get("post_text")
                explanation = post.get("explanation", "No explanation provided.")
                error = post.get("error")
                post_id = post.get("post_id", f"unknown_{i}")
                if error:
                    st.error(f"Error generating Post {i+1} ({post_id}): {error}")
                    if post_text:
                        st.warning(f"Raw content: {post_text}")
                    st.markdown("---")
                    continue
                if not post_text:
                    st.warning(f"Post {i+1} ({post_id}) has no content.")
                    st.markdown(f"<div class='explanation-section'>Explanation: {explanation}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    continue
                valid_posts.append(post)
                post_name = get_descriptive_post_name(post_text, i)
                escaped_post_text_for_js = json.dumps(post_text)
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
                            .then(() => {{ alert('Post text copied to clipboard!'); }})
                            .catch(err => {{ console.error('Failed to copy text: ', err); alert('Failed to copy text.'); }});
                    }}
                    </script>
                """, unsafe_allow_html=True)
                st.markdown(f"<div class='explanation-section'><b>Why this post?</b> {explanation}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='post-meta'>Generated: {post.get('generated_at', 'N/A')} | ID: {post_id}</div>", unsafe_allow_html=True)
                st.markdown("---")
            if valid_posts:
                output_filename = f"data/generated_posts{'_with_feedback' if regenerate_with_feedback else ''}{'_user' if user_keywords else ''}.csv"
                save_posts(valid_posts, output_filename)
                try:
                    with open(output_filename, "rb") as f:
                        st.download_button(
                            label=f"Download {'Refined' if regenerate_with_feedback else 'Generated'} Posts (CSV)",
                            data=f,
                            file_name=os.path.basename(output_filename),
                            mime="text/csv",
                            key=f"{key_prefix}_download_button"
                        )
                except FileNotFoundError:
                    st.error(f"Could not find file {output_filename} for download.")
                return valid_posts
            else:
                st.warning("No valid posts generated.")
                return None
        return None
    except FileNotFoundError:
        st.error(f"{trends_file} not found. Run `analyzer.py` or upload a `trends.csv` file.")
        logger.error(f"Trends file {trends_file} not found.")
        return None

def collect_and_apply_feedback(posts_for_feedback):
    st.subheader("Submit Feedback to Improve Posts")
    if not posts_for_feedback:
        st.warning("No posts available. Generate posts first.")
        if 'latest_posts' in st.session_state and st.session_state['latest_posts']:
            st.info("Using last generated posts for feedback.")
            posts_for_feedback = st.session_state['latest_posts']
        else:
            return
    if not isinstance(posts_for_feedback, list) or not all(isinstance(p, dict) for p in posts_for_feedback):
        st.error("Invalid posts data format.")
        logger.error(f"Invalid posts_for_feedback type: {type(posts_for_feedback)}")
        return
    valid_posts_for_feedback = [p for p in posts_for_feedback if p.get("post_text") and not p.get("error")]
    if not valid_posts_for_feedback:
        st.warning("No valid posts to provide feedback on.")
        return
    post_options = {get_descriptive_post_name(p["post_text"], i): p["post_id"] for i, p in enumerate(valid_posts_for_feedback)}
    selected_post_name = st.selectbox("Select a Post", options=list(post_options.keys()), key="feedback_post_select")
    if selected_post_name:
        post_id = post_options[selected_post_name]
        st.markdown(f"**Feedback for:** {selected_post_name}")
        selected_post_data = next((p for p in valid_posts_for_feedback if p["post_id"] == post_id), None)
        if selected_post_data:
            with st.expander("View Post Content"):
                st.markdown(f"```\n{selected_post_data['post_text']}\n```")
                st.markdown(f"**Why this post?** {selected_post_data.get('explanation', 'N/A')}")
        tone = st.text_input("Tone (e.g., more urgent, more empathetic)", key=f"tone_{post_id}")
        cta = st.text_input("CTA (e.g., link to signup, remove CTA)", key=f"cta_{post_id}")
        content = st.text_area("Content (e.g., add a stat, mention a feature, shorter)", key=f"content_{post_id}")
        other = st.text_input("Other (e.g., fewer emojis, target audience)", key=f"other_{post_id}")
        if st.button("Submit Feedback", key=f"submit_feedback_{post_id}"):
            user_feedback = {k: v for k, v in {"tone": tone, "cta": cta, "content": content, "other": other}.items() if v}
            if user_feedback:
                updated_global_feedback = apply_feedback(post_id, user_feedback)
                st.success(f"Feedback saved for {selected_post_name}. Use 'Refine Posts with Feedback' to generate improved posts.")
                st.session_state['feedback_submitted'] = True
            else:
                st.warning("Please provide at least one feedback field.")

def main():
    st.title("LinkedIn Content Creator AI")
    st.markdown("Create and refine LinkedIn posts for Archit Anand, optimized for audience engagement using AI.")
    if 'latest_posts' not in st.session_state:
        st.session_state['latest_posts'] = []
    if 'generate_clicked' not in st.session_state:
        st.session_state['generate_clicked'] = False
    if 'feedback_submitted' not in st.session_state:
        st.session_state['feedback_submitted'] = False
    if 'current_key_prefix' not in st.session_state:
        st.session_state['current_key_prefix'] = None
    if 'user_keywords' not in st.session_state:
        st.session_state['user_keywords'] = None
    st.sidebar.title("Navigation & Data")
    page = st.sidebar.radio("Go to", ["Create Posts", "View Trends"])
    uploaded_file = st.sidebar.file_uploader("Upload trends.csv (optional)", type="csv")
    trends_file = "data/trends.csv"
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        try:
            with open(trends_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("Uploaded new trends.csv")
        except Exception as e:
            st.sidebar.error(f"Failed to save uploaded file: {e}")
    if page == "View Trends":
        display_trends(trends_file)
    elif page == "Create Posts":
        st.header("Customize Posts")
        st.markdown("Enter keywords to focus your posts (optional), then generate posts optimized for Archit Anand’s audience trends.")
        user_keywords = st.text_input(
            "Keywords or Topic (e.g., new product launch, AI analytics)",
            help="Leave blank for trend-based posts.",
            key="keywords_input"
        )
        st.session_state['user_keywords'] = user_keywords.strip() if user_keywords.strip() else None
        st.header("Create New Posts")
        st.markdown("Generate posts based on trends and your keywords (if provided).")
        generated_posts = generate_and_display_posts(trends_file, regenerate_with_feedback=False, key_prefix="initial")
        if generated_posts:
            st.session_state['latest_posts'] = generated_posts
        st.header("Submit Feedback")
        st.markdown("Provide feedback on a post to improve future generations.")
        collect_and_apply_feedback(st.session_state.get('latest_posts', []))
        st.header("Refine Posts with Feedback")
        st.markdown("Generate new posts incorporating all feedback for better alignment with your preferences.")
        regenerated_posts = generate_and_display_posts(trends_file, regenerate_with_feedback=True, key_prefix="regenerate")
        if regenerated_posts:
            st.session_state['latest_posts'] = regenerated_posts

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()