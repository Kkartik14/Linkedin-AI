from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import logging
import pickle
import random
from fake_useragent import UserAgent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USERNAME = os.getenv('LINKEDIN_USERNAME')
PASSWORD = os.getenv('LINKEDIN_PASSWORD')
COOKIE_FILE = "data/linkedin_cookies.pkl"

PROFILES = [
    "https://www.linkedin.com/in/archit-anand/",
    "https://www.linkedin.com/in/aarongolbin/",
    "https://www.linkedin.com/in/robertschöne/",
    "https://www.linkedin.com/in/jaspar-carmichael-jack/"
]

def validate_credentials():
    """Validate that credentials are set."""
    if not USERNAME or not PASSWORD:
        logger.error("Missing LinkedIn credentials. Ensure LINKEDIN_USERNAME and LINKEDIN_PASSWORD are set in .env")
        raise ValueError("LINKEDIN_USERNAME and LINKEDIN_PASSWORD must be set in .env")

def initialize_driver(headless=True):
    """Initialize Selenium WebDriver with Chrome."""
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")  
    options.add_argument("--disable-blink-features=AutomationControlled")  
    ua = UserAgent()
    options.add_argument(f"user-agent={ua.random}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})") 
    return driver

def save_cookies(driver, path):
    """Save cookies to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(driver.get_cookies(), file)
    logger.info(f"Saved cookies to {path}")

def load_cookies(driver, path):
    """Load cookies from a file."""
    if os.path.exists(path):
        with open(path, 'rb') as file:
            cookies = pickle.load(file)
        driver.get("https://www.linkedin.com")  
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                logger.warning(f"Failed to add cookie: {e}")
        logger.info(f"Loaded cookies from {path}")
        return True
    return False

def login_to_linkedin(driver, retries=3):
    """Log in to LinkedIn with cookie persistence and retries."""
    logger.info("Attempting to log in to LinkedIn")
    for attempt in range(retries):
        try:
            driver.get("https://www.linkedin.com/login")
            time.sleep(random.uniform(2, 4))
            if load_cookies(driver, COOKIE_FILE):
                driver.get("https://www.linkedin.com/feed/")
                time.sleep(random.uniform(2, 4))
                if "feed" in driver.current_url:
                    logger.info("Successfully logged in using cookies")
                    return True
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "username")))
            username_field = driver.find_element(By.ID, "username")
            password_field = driver.find_element(By.ID, "password")
            submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")

            username_field.send_keys(USERNAME)
            password_field.send_keys(PASSWORD)
            submit_button.click()
            time.sleep(random.uniform(3, 5))

            if "checkpoint" in driver.current_url or "verify" in driver.current_url:
                logger.warning("Login checkpoint detected. Please complete manual verification.")
                input("Press Enter after completing verification...")
                time.sleep(random.uniform(2, 4))

            if "feed" not in driver.current_url:
                raise Exception("Login failed: Not redirected to feed")

            save_cookies(driver, COOKIE_FILE)
            logger.info("Login successful, cookies saved")
            return True

        except Exception as e:
            logger.error(f"Login attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                logger.info("Retrying...")
                time.sleep(random.uniform(5, 10))
            else:
                logger.error("All login attempts failed")
                return False
    return False

def scroll_page(driver, scroll_delay=3, max_attempts=50, min_posts=10):
    """Scroll until all posts are loaded or max attempts reached."""
    logger.info("Scrolling page to load all posts")
    attempts = 0
    last_post_count = 0
    max_stale_attempts = 3
    stale_attempts = 0

    while attempts < max_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(scroll_delay, scroll_delay + 2))

        soup = BeautifulSoup(driver.page_source, "html.parser")
        posts = soup.find_all("div", class_="feed-shared-update-v2")
        current_post_count = len(posts)
        logger.info(f"Attempt {attempts + 1}/{max_attempts}: Found {current_post_count} posts")

        if current_post_count == last_post_count:
            stale_attempts += 1
            logger.info(f"No new posts loaded (stale attempt {stale_attempts}/{max_stale_attempts})")
            if stale_attempts >= max_stale_attempts:
                logger.info("No more posts to load (stale limit reached)")
                if current_post_count < min_posts:
                    logger.warning(f"Found only {current_post_count} posts, which is less than min_posts={min_posts}")
                break
        else:
            stale_attempts = 0
            last_post_count = current_post_count

        attempts += 1

    if attempts >= max_attempts:
        logger.warning(f"Reached max attempts ({max_attempts}). May not have loaded all posts.")
    return last_post_count

def clean_number(text):
    """Clean number strings."""
    if not text:
        return 0
    try:
        return int(re.sub(r"[^\d]", "", text))
    except ValueError:
        return 0

def extract_posts(driver, profile_url, retries=3):
    """Extract all posts from a LinkedIn profile with retries."""
    logger.info(f"Extracting posts from {profile_url}")
    posts_url = profile_url.rstrip("/") + "/recent-activity/all/"

    for attempt in range(retries):
        try:
            driver.get(posts_url)
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "feed-shared-update-v2"))
                )
            except TimeoutException as e:
                logger.error(f"Timeout loading posts page: {e}")
                with open(f"data/error_page_{profile_url.split('/')[-2]}.html", "w") as f:
                    f.write(driver.page_source)
                logger.info(f"Saved page source to data/error_page_{profile_url.split('/')[-2]}.html")
                raise

            post_count = scroll_page(driver)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            posts = soup.find_all("div", class_="feed-shared-update-v2")

            post_data = []
            skipped_posts = 0
            for i, post in enumerate(posts):
                try:
                    text_elem = post.find("span", class_="break-words")
                    text = text_elem.get_text(strip=True) if text_elem else ""

                    if not text or len(text) < 10:
                        skipped_posts += 1
                        continue

                    hashtags = re.findall(r"#\w+", text)
                    date_elem = post.find("span", class_="update-components-actor__sub-description")
                    date = date_elem.get_text(strip=True) if date_elem else ""

                    likes_elem = post.find("span", class_="social-details-social-counts__reactions-count")
                    likes = clean_number(likes_elem.get_text(strip=True) if likes_elem else "0")

                    comments_elem = post.find("li", class_="social-details-social-counts__comments")
                    comments = clean_number(comments_elem.get_text(strip=True) if comments_elem else "0")

                    shares_elem = post.find("li", class_="social-details-social-counts__shares")
                    shares = clean_number(shares_elem.get_text(strip=True) if shares_elem else "0")

                    post_data.append({
                        "profile": profile_url,
                        "text": text,
                        "hashtags": hashtags,
                        "date": date,
                        "likes": likes,
                        "comments": comments,
                        "shares": shares
                    })
                    logger.info(f"Extracted post {i + 1}: {text[:50]}...")
                except Exception as e:
                    logger.error(f"Error parsing post {i + 1}: {e}")
                    skipped_posts += 1
                    continue

            logger.info(f"Extracted {len(post_data)} posts, skipped {skipped_posts} (Attempt {attempt + 1})")
            if len(post_data) > 0:
                return post_data
            logger.warning(f"No posts extracted on attempt {attempt + 1}. Retrying...")
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            logger.error(f"Failed to load posts page on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                logger.info("Retrying...")
                time.sleep(random.uniform(5, 10))
            continue

    logger.error(f"Failed to extract posts from {profile_url} after {retries} attempts")
    return []

def save_to_csv(data, filename="data/posts.csv"):
    """Save post data to CSV."""
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(data)} posts to {filename}")

def main():
    validate_credentials()
    driver = initialize_driver(headless=True)
    all_posts = []
    try:
        if not login_to_linkedin(driver):
            logger.error("Login failed. Exiting.")
            return
        for profile in PROFILES:
            posts = extract_posts(driver, profile)
            all_posts.extend(posts)
            time.sleep(random.uniform(5, 10))
        save_to_csv(all_posts)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        if all_posts:
            save_to_csv(all_posts, "data/posts_partial.csv")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()