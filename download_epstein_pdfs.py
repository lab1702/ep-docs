#!/usr/bin/env python3
"""
Download PDF Files from DOJ Epstein Library
https://www.justice.gov/epstein

Downloads all PDF files from:
1. Court Records
2. DOJ Disclosures
3. FOIA Records

Preserves folder structure to handle files with identical names.
"""

import os
import re
import csv
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://www.justice.gov"
OUTPUT_DIR = Path("./epstein_pdfs")

SECTIONS = {
    "court_records": "/epstein/court-records",
    "doj_disclosures": "/epstein/doj-disclosures",
    "foia": "/epstein/foia",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

COOKIES = {
    'doj_disclosures': {
        'justiceGovAgeVerified': 'true',
    },
}

def safe_read_html(url, max_retries=3, delay=2):
    """Safely read a webpage with retry logic."""
    for i in range(max_retries):
        try:
            time.sleep(delay)
            response = requests.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(f"  Attempt {i+1} failed for {url}: {e}")
    return None


def extract_pdf_links(page, base_url=BASE_URL):
    """Extract all PDF links from a page."""
    if page is None:
        return []

    pdf_links = []
    for link in page.find_all("a", href=True):
        href = link["href"]
        if href.lower().endswith(".pdf"):
            if href.startswith("http"):
                pdf_links.append(href)
            elif href.startswith("/"):
                pdf_links.append(base_url + href)
            else:
                pdf_links.append(base_url + "/" + href)

    return list(set(pdf_links))


def extract_subpage_links(page, section_path, base_url=BASE_URL):
    """Extract links to sub-pages within a section."""
    if page is None:
        return []

    subpage_links = []
    for link in page.find_all("a", href=True):
        href = link["href"]
        if section_path in href:
            if href.startswith("http"):
                subpage_links.append(href)
            elif href.startswith("/"):
                subpage_links.append(base_url + href)
            else:
                subpage_links.append(base_url + "/" + href)

    return list(set(subpage_links))


def url_to_local_path(url, base_output_dir=OUTPUT_DIR):
    """Convert URL to local file path, preserving folder structure."""
    decoded_url = unquote(url)
    parsed = urlparse(decoded_url)
    url_path = parsed.path.lstrip("/")

    # Sanitize path components
    parts = url_path.split("/")
    sanitized_parts = []
    for part in parts:
        sanitized = re.sub(r"[^a-zA-Z0-9._() -]", "_", part)
        sanitized_parts.append(sanitized)

    return base_output_dir / Path(*sanitized_parts)


def download_pdf(url, base_output_dir=OUTPUT_DIR, overwrite=False):
    """Download a single PDF file."""
    output_path = url_to_local_path(url, base_output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"  Skipping (exists): {output_path}")
        return {"success": True, "skipped": True}

    try:
        time.sleep(0.5)
        response = requests.get(
            url, headers=HEADERS, timeout=120, **({
                'cookies': COOKIES["doj_disclosures"]
            } if 'DataSet' in url else {}
        )) # Kludge, but trying to keep mods simple
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            print(f"  Downloaded: {output_path}")
            return {"success": True, "skipped": False}
        else:
            print(f"  Failed (HTTP {response.status_code}): {output_path.name}")
            return {"success": False, "skipped": False}
    except Exception as e:
        print(f"  Error downloading {output_path.name}: {e}")
        return {"success": False, "skipped": False}


def scrape_court_records():
    """Scrape Court Records section."""
    print("\n========================================")
    print("Scraping Court Records section...")
    print("========================================\n")

    all_pdf_links = []
    visited_pages = set()

    main_url = BASE_URL + SECTIONS["court_records"]
    main_page = safe_read_html(main_url)
    visited_pages.add(main_url)

    if main_page is None:
        print("Failed to load court records main page")
        return all_pdf_links

    main_pdfs = extract_pdf_links(main_page)
    all_pdf_links.extend(main_pdfs)
    print(f"Found {len(main_pdfs)} PDFs on main page")

    # Find sub-pages
    subpages = extract_subpage_links(main_page, "/epstein/court-records/")
    subpages = [s for s in subpages if s not in visited_pages]
    print(f"Found {len(subpages)} sub-pages to scrape")

    for subpage_url in subpages:
        if subpage_url in visited_pages:
            continue

        print(f"\nScraping: {subpage_url}")
        visited_pages.add(subpage_url)

        subpage = safe_read_html(subpage_url)
        if subpage:
            subpage_pdfs = extract_pdf_links(subpage)
            all_pdf_links.extend(subpage_pdfs)
            print(f"  Found {len(subpage_pdfs)} PDFs")

            # Check for pagination
            for link in subpage.find_all("a", href=True):
                href = link["href"]
                if "?page=" in href:
                    if not href.startswith("http"):
                        href = BASE_URL + href
                    if href not in visited_pages:
                        print(f"  Scraping paginated: {href}")
                        visited_pages.add(href)
                        paged = safe_read_html(href)
                        if paged:
                            paged_pdfs = extract_pdf_links(paged)
                            all_pdf_links.extend(paged_pdfs)
                            print(f"    Found {len(paged_pdfs)} PDFs")

    all_pdf_links = list(set(all_pdf_links))
    print(f"\nTotal unique PDFs found in Court Records: {len(all_pdf_links)}")
    return all_pdf_links


def scrape_doj_disclosures():
    """Scrape DOJ Disclosures section."""
    print("\n========================================")
    print("Scraping DOJ Disclosures section...")
    print("========================================\n")

    all_pdf_links = []
    visited_pages = set()

    main_url = BASE_URL + SECTIONS["doj_disclosures"]
    main_page = safe_read_html(main_url)
    visited_pages.add(main_url)

    if main_page is None:
        print("Failed to load DOJ disclosures main page")
        return all_pdf_links

    main_pdfs = extract_pdf_links(main_page)
    all_pdf_links.extend(main_pdfs)
    print(f"Found {len(main_pdfs)} PDFs on main page")

    # Check data set sub-pages
    for i in range(1, 13):
        ds_url = f"{BASE_URL}/epstein/doj-disclosures/data-set-{i}-files"
        print(f"\nChecking Data Set: {ds_url}")

        ds_page = safe_read_html(ds_url)
        if ds_page is None:
            print("  Data set not found or empty, skipping...")
            continue

        visited_pages.add(ds_url)
        ds_pdfs = extract_pdf_links(ds_page)

        if not ds_pdfs:
            print("  No PDFs found, skipping...")
            continue

        all_pdf_links.extend(ds_pdfs)
        print(f"  Page 0: Found {len(ds_pdfs)} PDFs")

        # Check pagination
        page_num = 1
        while True:
            page_url = f"{ds_url}?page={page_num}"
            if page_url in visited_pages:
                continue

            paged = safe_read_html(page_url)
            if paged is None:
                break

            visited_pages.add(page_url)
            paged_pdfs = extract_pdf_links(paged)

            if not paged_pdfs:
                break

            new_pdfs = [p for p in paged_pdfs if p not in all_pdf_links]
            if not new_pdfs:
                break

            all_pdf_links.extend(paged_pdfs)
            print(f"  Page {page_num}: Found {len(new_pdfs)} new PDFs")
            page_num += 1

    all_pdf_links = list(set(all_pdf_links))
    print(f"\nTotal unique PDFs found in DOJ Disclosures: {len(all_pdf_links)}")
    return all_pdf_links


def scrape_foia():
    """Scrape FOIA section."""
    print("\n========================================")
    print("Scraping FOIA section...")
    print("========================================\n")

    main_url = BASE_URL + SECTIONS["foia"]
    main_page = safe_read_html(main_url)

    if main_page is None:
        print("Failed to load FOIA main page")
        return []

    main_pdfs = extract_pdf_links(main_page)
    print(f"Found {len(main_pdfs)} PDFs on main page")

    all_pdf_links = list(set(main_pdfs))
    print(f"\nTotal unique PDFs found in FOIA: {len(all_pdf_links)}")
    return all_pdf_links


def download_all_pdfs(pdf_links, base_output_dir=OUTPUT_DIR):
    """Download all PDFs."""
    print("\n========================================")
    print(f"Downloading {len(pdf_links)} PDFs")
    print(f"Preserving folder structure under: {base_output_dir}")
    print("========================================\n")

    base_output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, url in enumerate(pdf_links, 1):
        print(f"[{i}/{len(pdf_links)}] {url}")
        result = download_pdf(url, base_output_dir)

        if result["success"]:
            if result["skipped"]:
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1

        if i % 100 == 0:
            print(f"\n--- Progress: {i}/{len(pdf_links)} ({100*i/len(pdf_links):.1f}%) ---\n")

    print("\n----------------------------------------")
    print("Download Summary:")
    print(f"  New downloads: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Failed: {fail_count}")
    print("----------------------------------------\n")

    return {"success": success_count, "skipped": skip_count, "failed": fail_count}


def main(skip_confirmation=False):
    """Main execution function."""
    print("==============================================")
    print("DOJ Epstein Library PDF Downloader")
    print("==============================================\n")
    print("This script will download all PDF files from:")
    print("  1. Court Records")
    print("  2. DOJ Disclosures")
    print("  3. FOIA Records")
    print("\nFolder structure will be preserved to handle")
    print("files with identical names (e.g., 001.pdf)")
    print("\nNote: The House Disclosures section links to")
    print("an external site and is not included.")
    print("\n==============================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scrape all sections
    court_pdfs = scrape_court_records()
    doj_pdfs = scrape_doj_disclosures()
    foia_pdfs = scrape_foia()

    # Combine all PDFs
    all_pdfs = list(set(court_pdfs + doj_pdfs + foia_pdfs))

    print("\n==============================================")
    print("SCRAPING COMPLETE - Summary")
    print("==============================================")
    print(f"Court Records: {len(court_pdfs)} PDFs")
    print(f"DOJ Disclosures: {len(doj_pdfs)} PDFs")
    print(f"FOIA: {len(foia_pdfs)} PDFs")
    print(f"Total unique: {len(all_pdfs)} PDFs")
    print("==============================================\n")

    # Save URLs to JSON
    url_data = {
        "court_records": court_pdfs,
        "doj_disclosures": doj_pdfs,
        "foia": foia_pdfs,
        "all": all_pdfs,
    }
    json_path = OUTPUT_DIR / "pdf_urls.json"
    with open(json_path, "w") as f:
        json.dump(url_data, f, indent=2)
    print(f"PDF URLs saved to: {json_path}")

    # Save as CSV
    csv_path = OUTPUT_DIR / "pdf_urls.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "local_path"])
        for url in all_pdfs:
            writer.writerow([url, str(url_to_local_path(url))])
    print(f"PDF URLs also saved to: {csv_path}")

    if not skip_confirmation:
        response = input("\nProceed with download? (y/n): ")
        if response.lower() != "y":
            print("\nDownload cancelled.")
            print("Run with --download-saved to download later.")
            return

    results = download_all_pdfs(all_pdfs, OUTPUT_DIR)

    print("\n==============================================")
    print("DOWNLOAD COMPLETE - Final Summary")
    print("==============================================")
    print(f"New downloads: {results['success']}")
    print(f"Skipped (already exist): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")
    print("==============================================\n")


def download_from_saved(json_file=None):
    """Download from previously saved URLs."""
    json_file = json_file or OUTPUT_DIR / "pdf_urls.json"
    if not Path(json_file).exists():
        print(f"URL file not found: {json_file}")
        print("Run main() first to scrape URLs.")
        return

    with open(json_file) as f:
        url_data = json.load(f)

    print(f"Loaded {len(url_data['all'])} URLs from {json_file}")
    download_all_pdfs(url_data["all"], OUTPUT_DIR)


if __name__ == "__main__":
    import sys

    if "--download-saved" in sys.argv:
        download_from_saved()
    elif "--yes" in sys.argv or "-y" in sys.argv:
        main(skip_confirmation=True)
    else:
        main()
