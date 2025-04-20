from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# LinkedIn credentials (store securely in production)
USERNAME = os.getenv('LinkedIn_USERNAME') 
PASSWORD = os.getenv('LinkedIn_PASSWORD') 

# Profiles to scrape
PROFILES = [
    "https://www.linkedin.com/in/archit-anand/",
    "https://www.linkedin.com/in/aarongolbin/",
    "https://www.linkedin.com/in/robertschöne/",
    "https://www.linkedin.com/in/jaspar-carmichael-jack/"
]

def initialize_driver():
    """Initialize Selenium WebDriver with Chrome."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def login_to_linkedin(driver):
    """Log in to LinkedIn."""
    logger.info("Logging in to LinkedIn")
    driver.get("https://www.linkedin.com/login")
    time.sleep(3)
    try:
        driver.find_element(By.ID, "username").send_keys(USERNAME)
        driver.find_element(By.ID, "password").send_keys(PASSWORD)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(5)
        if "checkpoint" in driver.current_url:
            logger.warning("Login checkpoint detected. Manual intervention required.")
            raise Exception("LinkedIn login checkpoint")
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise

def scroll_page(driver, max_scrolls=10, scroll_delay=3):
    """Scroll to load all posts with enhanced logic."""
    logger.info("Scrolling page to load posts")
    scrolls = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    while scrolls < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_delay)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logger.info("No more posts to load")
            break
        last_height = new_height
        scrolls += 1
        logger.info(f"Scroll {scrolls}/{max_scrolls}")
    return scrolls

def clean_number(text):
    """Clean number strings by removing commas and non-numeric characters."""
    if not text:
        return 0
    try:
        return int(re.sub(r"[^\d]", "", text))
    except ValueError:
        return 0

def extract_posts(driver, profile_url):
    """Extract posts from a LinkedIn profile."""
    logger.info(f"Extracting posts from {profile_url}")
    posts_url = profile_url + "recent-activity/all/"
    driver.get(posts_url)
    time.sleep(3)
    scroll_page(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    posts = soup.find_all("div", class_="feed-shared-update-v2")

    post_data = []
    skipped_posts = 0
    for i, post in enumerate(posts):
        try:
            # Extract post text
            text_elem = post.find("span", class_="break-words")
            text = text_elem.get_text(strip=True) if text_elem else ""

            # Extract hashtags
            hashtags = re.findall(r"#\w+", text)

            # Extract date
            date_elem = post.find("span", class_="update-components-actor__sub-description")
            date = date_elem.get_text(strip=True) if date_elem else ""

            # Extract likes, comments, shares
            likes_elem = post.find("span", class_="social-details-social-counts__reactions-count")
            likes = clean_number(likes_elem.get_text(strip=True)) if likes_elem else 0

            comments_elem = post.find("li", class_="social-details-social-counts__comments")
            comments = clean_number(comments_elem.get_text(strip=True).split()[0]) if comments_elem else 0

            shares_elem = post.find("li", class_="social-details-social-counts__shares")
            shares = clean_number(shares_elem.get_text(strip=True).split()[0]) if shares_elem else 0

            post_data.append({
                "profile": profile_url,
                "text": text,
                "hashtags": hashtags,
                "date": date,
                "likes": likes,
                "comments": comments,
                "shares": shares
            })
            logger.info(f"Extracted post {i+1}: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error parsing post {i+1}: {e}")
            skipped_posts += 1
            continue

    logger.info(f"Extracted {len(post_data)} posts, skipped {skipped_posts}")
    return post_data

def save_to_csv(data, filename="data/posts.csv"):
    """Save post data to CSV."""
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(data)} posts to {filename}")

def main():
    driver = initialize_driver()
    try:
        login_to_linkedin(driver)
        all_posts = []
        for profile in PROFILES:
            posts = extract_posts(driver, profile)
            all_posts.extend(posts)
            time.sleep(5)  # Increased delay to avoid rate limiting
        save_to_csv(all_posts)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()