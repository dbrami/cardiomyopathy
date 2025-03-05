import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from urllib.parse import urljoin
import logging
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAIN_URL = "https://web.mit.edu/sirna/sirnas-human.html"
KNOCKDOWN_COLUMN = "mRNA knockdown"

def scrape_main_page(url, session):
    """Scrape the main page and extract table data."""
    response = session.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to retrieve main page: {response.status_code}")
        raise Exception("Main page retrieval failed")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'width': '400', 'cellpadding': '4', 'bgcolor': '#000000'})
    if not table:
        logging.error("No table found on the main page")
        raise Exception("No table found")
    
    header_row = table.find('tr', class_='cellhead')
    if not header_row:
        logging.error("No header row found in the table")
        raise Exception("No header row found")
    
    headers = []
    for td in header_row.find_all('td'):
        h4 = td.find('h4')
        strong = td.find('strong')
        if h4:
            headers.append(h4.get_text().strip())
        elif strong:
            headers.append(strong.get_text().strip())
        else:
            headers.append(td.get_text().strip())
    
    logging.info(f"Headers extracted: {headers}")
    
    expected_headers = [
        "Target Gene", "siRNA ID#", "siRNA", "shRNA", "Human",
        "NCBI Probe #", "mRNA knockdown", "Protein knockdown"
    ]
    if len(headers) != len(expected_headers):
        logging.error("Number of headers does not match expected")
        raise Exception("Unexpected number of table headers")
    
    rows = []
    for tr in table.find_all('tr')[1:]:
        cells = tr.find_all('td')
        if len(cells) != len(headers):
            logging.warning(f"Skipping row with {len(cells)} cells, expected {len(headers)}")
            continue
        
        row = {}
        for i, cell in enumerate(cells):
            if headers[i] == "siRNA ID#":
                a_tag = cell.find('a')
                if a_tag:
                    row[headers[i]] = a_tag.get_text().strip()
                    relative_url = a_tag['href']
                    row["siRNA ID# URL"] = urljoin(url, relative_url)
                else:
                    row[headers[i]] = cell.get_text().strip()
                    row["siRNA ID# URL"] = None
            else:
                row[headers[i]] = cell.get_text().strip()
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def scrape_detail_page(url, session):
    """Scrape a detail page for additional sequence data."""
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Failed to retrieve detail page: {url} - Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request exception for {url}: {e}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    # Locate the inner table containing the detail data (width 500)
    table = soup.find("table", {"width": "500"})
    if not table:
        logging.warning(f"No inner table found on detail page: {url}")
        return None

    details = {}
    rows = table.find_all("tr")
    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            continue
        
        # Use the text of the first cell to decide what data is in the row
        cell_text = tds[0].get_text(" ", strip=True)
        # Skip rows that contain unwanted data
        if "Probe type:" in cell_text:
            continue

        if "Target gene:" in cell_text:
            # Remove the heading and extra whitespace
            target_text = cell_text.replace("Target gene:", "").strip()
            details["Target gene (detailed)"] = target_text

        if "Sense sequence" in cell_text:
            sense_text = cell_text.replace("Sense sequence", "").replace(":", "").strip()
            details["Sense sequence"] = sense_text
            # Also check second cell for NCBI Database ID# if present
            if len(tds) > 1:
                ncbi_text = tds[1].get_text(" ", strip=True)
                if "NCBI Database ID#" in ncbi_text:
                    ncbi_text = ncbi_text.replace("NCBI Database ID#:", "").strip()
                    details["NCBI Database ID#"] = ncbi_text

        if "Anti-sense sequence" in cell_text:
            antisense_text = cell_text.replace("Anti-sense sequence", "").replace(":", "").strip()
            details["Anti-sense sequence"] = antisense_text

    if not details:
        logging.warning(f"No matching fields found on detail page: {url}")
        return None
    
    return details

def parse_knockdown(value):
    """Parse knockdown values, handling ranges and percentages."""
    if pd.isna(value) or not value.strip():
        return np.nan
    value = value.strip()
    if '-' in value:
        try:
            parts = value.split('-')
            # Remove any '%' and extra whitespace from each part
            low_str = parts[0].replace('%', '').strip()
            high_str = parts[1].replace('%', '').strip()
            low = float(low_str)
            high = float(high_str)
            result = (low + high) / 2
            if value.endswith('%'):
                result /= 100
            return result
        except ValueError:
            logging.warning(f"Could not parse knockdown range: {value}")
            return np.nan
    elif value.endswith('%'):
        try:
            return float(value.replace('%', '').strip()) / 100
        except ValueError:
            logging.warning(f"Could not parse knockdown percentage: {value}")
            return np.nan
    else:
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Could not parse knockdown value: {value}")
            return np.nan

if __name__ == "__main__":
    logging.info("Starting the scraping process")
    try:
        with requests.Session() as session:
            # Scrape the main page
            main_df = scrape_main_page(MAIN_URL, session)
            logging.info(f"Scraped main table with {len(main_df)} rows")
            
            # Scrape detail pages with progress bar
            detail_data = []
            for index, row in tqdm(main_df.iterrows(), total=len(main_df), desc="Processing detail pages"):
                url = row["siRNA ID# URL"]
                if url:
                    details = scrape_detail_page(url, session)
                    detail_data.append(details if details else {})
                    time.sleep(1)  # Be polite to the server
                else:
                    detail_data.append({})
            
            # Combine main and detail data
            detail_df = pd.DataFrame(detail_data)
            enriched_df = pd.concat([main_df, detail_df], axis=1)
            
            # Parse knockdown values and compute average
            enriched_df[f"{KNOCKDOWN_COLUMN} numeric"] = enriched_df[KNOCKDOWN_COLUMN].apply(parse_knockdown)
            average_knockdown = enriched_df[f"{KNOCKDOWN_COLUMN} numeric"].mean()
            
            # Ensure output directory exists
            output_file = "../data/mit_sirna/enriched_sirna_data.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            enriched_df.to_csv(output_file, index=False)
            logging.info(f"Enriched data saved to {output_file}")
            logging.info(f"Average {KNOCKDOWN_COLUMN}: {average_knockdown}")
            print(f"Average {KNOCKDOWN_COLUMN}: {average_knockdown}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise