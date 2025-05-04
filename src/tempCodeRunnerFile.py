from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

options = Options()
options.add_argument("--start-maximized")  # atau opsional lainnya

# Setup driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Login manual
print("Silakan login ke Twitter di browser...")
driver.get("https://twitter.com/login")
input("Setelah login selesai, tekan ENTER di terminal ini...")

# Daftar keyword
queries = [
    "oriental circus indonesia",
    "eksploitasi ex oci"
]

# Inisialisasi
data = []
scroll_pause = 4
max_scrolls = 1000
for query in queries:
    print(f"\n🔍 Mulai scraping keyword: {query}")
    search_url = f"https://twitter.com/search?q={query.replace(' ', '%20')}&src=typed_query&f=live"
    driver.get(search_url)
    time.sleep(5)

    last_height = driver.execute_script("return document.body.scrollHeight")
    tweet_found = False

    for scroll in range(max_scrolls):
        print(f"📜 Scroll ke-{scroll+1} untuk keyword: {query}")

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//article[@data-testid="tweet"]'))
            )
        except:
            print("⏳ Tidak ada tweet yang ketemu, tunggu sebentar...")
            time.sleep(5)
            continue

        tweet_blocks = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')

        for tweet in tweet_blocks:
            try:
                text_elem = tweet.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
                text = text_elem.text

                time_elem = tweet.find_element(By.XPATH, './/time')
                timestamp = time_elem.get_attribute("datetime")

                data.append({
                    "platform": "twitter",
                    "keyword": query,
                    "komentar": text,
                    "timestamp": timestamp,
                    "sentimen": ""  # kosong dulu
                })
                tweet_found = True
            except:
                continue

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print(f"✅ Scroll selesai untuk keyword: {query}")
            break
        last_height = new_height

    if not tweet_found:
        print(f"⚠ Tidak ada tweet yang ditemukan untuk keyword: {query}")

# Simpan ke CSV
df = pd.DataFrame(data)
df.to_csv("komentar_twitter.csv", index=False)
print(f"\n✅ Berhasil simpan {len(df)} komentar ke komentar_twitter.csv")

driver.quit()