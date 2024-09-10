import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def setup_driver():
    firefox_options = Options()
    # firefox_options.add_argument("-headless")  # Run in headless mode, remove if you want to see the browser
    return webdriver.Firefox(options=firefox_options)

def is_valid_name(name):
    return name.isascii() and name.replace('_', '').isalnum()

def download_gif(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        return True
    return False

def scrape_emotes(driver, streamer, output_folder):
    base_url = "https://twitchemotes.com/"
    driver.get(base_url)

    try:
        # Wait for the search bar to be visible and interact with it
        search_bar = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Channel Name']"))
        )
        search_bar.send_keys(streamer)
        search_bar.send_keys(Keys.RETURN)

        # Wait for the page to load
        time.sleep(5)  # Adjust this value if needed

        # Find all img elements
        img_elements = driver.find_elements(By.TAG_NAME, "img")

        emote_count = 0
        for img in img_elements:
            src = img.get_attribute('src')
            if src and 'animated' in src:
                emote_count += 1
                filename = os.path.join(output_folder, f"{streamer}_{emote_count}.gif")
                if download_gif(src, filename):
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Failed to download: {src}")

        if emote_count == 0:
            print(f"No animated emotes found for {streamer}")
        else:
            print(f"Downloaded {emote_count} emotes for {streamer}")

    except TimeoutException:
        print(f"Timeout occurred while processing {streamer}. The page might not have loaded completely.")
    except NoSuchElementException:
        print(f"Could not find elements for {streamer}. The streamer might not have any emotes.")
    except Exception as e:
        print(f"An error occurred while processing {streamer}: {str(e)}")

def main():
    driver = setup_driver()
    driver.install_addon('ublock_origin-1.59.0.xpi')
    output_folder = "twitch_emotes"
    os.makedirs(output_folder, exist_ok=True)

    with open("top_streamers.txt", "r", encoding="utf-8") as file:
        streamers = [line.split(". ", 1)[1].strip() for line in file if ". " in line]

    for streamer in streamers:
        if is_valid_name(streamer):
            print(f"Processing {streamer}...")
            scrape_emotes(driver, streamer, output_folder)
            time.sleep(3)  # Wait between streamers
        else:
            print(f"Skipping invalid name: {streamer}")

    driver.quit()
    print("Emote download complete!")

if __name__ == "__main__":
    main()