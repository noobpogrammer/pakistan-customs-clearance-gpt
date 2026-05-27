print(">>> INITIALIZING: CustomSPK Spider Bot <<<")

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import urllib3
import re

# 1. IGNORE SSL ERRORS (Crucial for customspk.com)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
START_URL = "http://customspk.com/"
DOMAIN = "customspk.com"
DOWNLOAD_FOLDER = "uploads"
MAX_PAGES = 50  # Safety limit. Increase this to 500 or 1000 if you want EVERYTHING.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Create folder
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Global tracking
visited_urls = set()
urls_to_visit = [START_URL]

def clean_html_text(soup):
    """Extracts readable text from the webpage, ignoring menus and ads."""
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer"]):
        script.extract()
    return soup.get_text(separator='\n').strip()

def spider_crawl():
    # Lazy load the AI engine
    print("2. 🧠 Loading AI Brain (Wait 10s)...")
    try:
        from chatbot_server import add_documents_to_index
        print("   -> AI Engine Ready!")
    except ImportError:
        print("❌ Error: Could not find 'app.py'.")
        return

    count = 0
    
    while urls_to_visit and count < MAX_PAGES:
        # Get next URL
        current_url = urls_to_visit.pop(0)
        
        if current_url in visited_urls:
            continue
            
        visited_urls.add(current_url)
        count += 1
        
        print(f"\n🕷️ Visiting [{count}/{MAX_PAGES}]: {current_url}")
        
        try:
            # Request the page
            response = requests.get(current_url, headers={"User-Agent": USER_AGENT}, verify=False, timeout=20)
            if response.status_code != 200:
                print(f"   ❌ Failed to load (Status {response.status_code})")
                continue

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- ACTION 1: INDEX THE WEBPAGE TEXT ---
            page_text = clean_html_text(soup)
            if len(page_text) > 500: # Only index if it has substantial text
                # We save the text as a temporary file to feed the ingestor
                temp_filename = f"webpage_{count}.txt"
                temp_path = os.path.join(DOWNLOAD_FOLDER, temp_filename)
                
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: {current_url}\n\n{page_text}")
                
                # Feed to AI
                add_documents_to_index(temp_path, f"Webpage: {soup.title.string if soup.title else 'Untitled'}")
                print(f"   📝 Read & Indexed page text.")
                
                # Cleanup temp file
                os.remove(temp_path)

            # --- ACTION 2: FIND PDFS ---
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                full_url = urljoin(current_url, href)
                
                # If it's a PDF, download it
                if full_url.lower().endswith('.pdf'):
                    if full_url not in visited_urls:
                        visited_urls.add(full_url) # Mark downloaded
                        
                        pdf_name = full_url.split('/')[-1]
                        # Fix weird filenames
                        pdf_name = re.sub(r'[^\w\-_\.]', '_', pdf_name)
                        local_pdf_path = os.path.join(DOWNLOAD_FOLDER, pdf_name)
                        
                        print(f"   ⬇️  Found PDF: {pdf_name}")
                        try:
                            pdf_resp = requests.get(full_url, stream=True, verify=False, headers={"User-Agent": USER_AGENT})
                            with open(local_pdf_path, 'wb') as f:
                                for chunk in pdf_resp.iter_content(8192):
                                    f.write(chunk)
                            
                            # Index the PDF
                            add_documents_to_index(local_pdf_path, pdf_name)
                            print(f"      -> Indexed PDF successfully.")
                        except Exception as e:
                            print(f"      -> Failed to download PDF: {e}")

                # --- ACTION 3: FIND MORE PAGES (RECURSION) ---
                # Only follow links that are part of customspk.com
                elif DOMAIN in full_url:
                    if full_url not in visited_urls and full_url not in urls_to_visit:
                        # Avoid duplicates
                        urls_to_visit.append(full_url)

        except Exception as e:
            print(f"   ❌ Error crawling page: {e}")
        
        # Be polite, don't hammer the server
        time.sleep(1)

    print(f"\n✅ CRAWL FINISHED. Visited {count} pages.")
    print("   Run 'python app.py' to use your new knowledge!")

if __name__ == "__main__":
    spider_crawl()