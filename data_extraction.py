from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse, urlunparse
import re
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import base64

def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def get_unique_main_links(links):
    normalized = set()
    unique_links = []
    for link in links:
        norm = normalize_url(link)
        if norm not in normalized:
            normalized.add(norm)
            unique_links.append(norm)
    return unique_links

def get_subpage_links(url, base_url="https://www.wpi.edu/student-experience/", visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html5lib')
    links = []
    for a in soup.find_all('a', href=True):
        absolute_link = urljoin(url, a['href'])
        if absolute_link.startswith(base_url):
            links.append(absolute_link)
            links.extend(get_subpage_links(absolute_link, base_url, visited))
    return list(set(links))

def save_page_as_pdf(url, output_path):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--allow-running-insecure-content')
    
    driver = webdriver.Chrome(options=chrome_options)
    try:
        print(f"Loading {url} and saving as {output_path}")
        driver.get(url)
        time.sleep(5)  
        
        
        result = driver.execute_cdp_cmd('Page.printToPDF', {
            'printBackground': True,
            'preferCSSPageSize': True,
        })
        
    
        
        pdf_data = base64.b64decode(result['data'])
        
        # Save the PDF
        with open(output_path, 'wb') as f:
            f.write(pdf_data)
            
        print(f"Successfully saved {output_path}")
        
    except Exception as e:
        print(f"Error saving {url}: {e}")
        
        try:
            print(f"Result keys: {result.keys() if 'result' in locals() else 'No result'}")
        except:
            pass
    finally:
        driver.quit()

def main():
    main_url = "https://www.wpi.edu/student-experience/"
    print("Fetching all subpage links...")
    all_links = get_subpage_links(main_url)
    print(f"Found {len(all_links)} links before deduplication.")
    unique_links = get_unique_main_links(all_links)
    print(f"{len(unique_links)} unique main links after normalization.")

   
    output_dir = 'site_data'
    os.makedirs(output_dir, exist_ok=True)

    for idx, link in enumerate(unique_links):
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', urlparse(link).path.strip('/')) or f'page_{idx}'
        pdf_path = os.path.join(output_dir, f'{safe_filename}.pdf')
        print(f"[{idx+1}/{len(unique_links)}] Saving {link} as {pdf_path}")
        try:
            save_page_as_pdf(link, pdf_path)
        except Exception as e:
            print(f"Failed to save {link}: {e}")

if __name__ == '__main__':
    main()

