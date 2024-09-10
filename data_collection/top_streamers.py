from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

def setup_driver():
    firefox_options = Options()
    # firefox_options.add_argument("-headless")  # Run in headless mode, remove if you want to see the browser
    return webdriver.Firefox(options=firefox_options)

def scrape_twitch_streamers(url, max_streamers=1000):
    driver = setup_driver()
    driver.get(url)

    streamers = []
    page = 1

    while len(streamers) < max_streamers:
        print(f"Scraping page {page}...")
        
        try:
            # Wait for the specific table to load
            table = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//div[2]/div[2]/div[2]/table"))
            )
            print("Table found. Proceeding to scrape...")
        except TimeoutException:
            print("Timed out waiting for table to load. Trying to proceed anyway...")
        
        # Find all rows in the table
        try:
            rows = driver.find_elements(By.XPATH, "//div[2]/div[2]/div[2]/table/tbody/tr")
            print(f"Found {len(rows)} rows")
        except NoSuchElementException:
            print("Could not find table rows. Exiting...")
            break

        for i, row in enumerate(rows, 1):
            try:
                # Extract the streamer name using the provided XPath structure
                name_element = row.find_element(By.XPATH, "./td[3]/a")
                streamer_name = name_element.text.strip()
                streamers.append(streamer_name)
                print(f"Scraped streamer: {streamer_name}")
                
                if len(streamers) >= max_streamers:
                    break
            except NoSuchElementException:
                print(f"Could not find name element in row {i}")

        if len(streamers) >= max_streamers:
            break

        # Try to click the "NEXT" button
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'NEXT')]"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)  # Wait for the page to load
            page += 1
        except (NoSuchElementException, TimeoutException):
            print("No more pages to load or couldn't find the NEXT button.")
            break

    driver.quit()
    return streamers

def save_to_file(streamers, filename="top_streamers.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, streamer in enumerate(streamers, 1):
            f.write(f"{i}. {streamer}\n")

# Main execution
url = "https://twitchtracker.com/channels/ranking"
top_streamers = scrape_twitch_streamers(url)
save_to_file(top_streamers)

print(f"Scraped {len(top_streamers)} streamers and saved to top_streamers.txt")