#!/usr/bin/env python3
"""
Timeline Extraction Script - Extract dates and build chronology from documents.

Uses spaCy NER and regex patterns to extract dates, normalize them, and
build a timeline of events/mentions across the document corpus.

Setup:
    python -m spacy download en_core_web_sm
    pip install python-dateutil

Usage:
    python extract_timeline.py
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
import multiprocessing
import json
import re

import spacy
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as date_parser
from datetime import datetime

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./timeline_extraction_output")

# Date range filter (to exclude obviously wrong dates)
MIN_YEAR = 1970
MAX_YEAR = 2025

# Global variable for worker processes
_worker_nlp = None


def _init_worker():
    """Initialize spaCy model for each worker process."""
    global _worker_nlp
    _worker_nlp = spacy.load('en_core_web_sm')
    _worker_nlp.max_length = 2_000_000


# Regex patterns for common date formats
DATE_PATTERNS = [
    # MM/DD/YYYY or MM-DD-YYYY
    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
    # YYYY/MM/DD or YYYY-MM-DD
    r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
    # Month DD, YYYY
    r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
    # DD Month YYYY
    r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
    # Month YYYY
    r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
    # Abbreviated months: Jan 15, 2005
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4})\b',
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]


def extract_dates_regex(text):
    """Extract dates using regex patterns."""
    dates = []
    for pattern in COMPILED_PATTERNS:
        matches = pattern.findall(text)
        dates.extend(matches)
    return dates


def extract_dates_spacy(text, nlp):
    """Extract DATE entities using spaCy NER."""
    max_chunk = 100000
    dates = []

    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == 'DATE':
                    dates.append(ent.text.strip())
        except Exception:
            continue

    return dates


def parse_date(date_str):
    """Try to parse a date string into a datetime object."""
    try:
        # Clean up the string
        date_str = ' '.join(date_str.split())

        # Try parsing with dateutil
        parsed = date_parser.parse(date_str, fuzzy=True, default=datetime(2000, 1, 1))

        # Validate year range
        if MIN_YEAR <= parsed.year <= MAX_YEAR:
            return parsed
    except (ValueError, OverflowError, TypeError):
        pass

    return None


def extract_context(text, date_str, context_chars=100):
    """Extract text context around a date mention."""
    idx = text.find(date_str)
    if idx == -1:
        return ""

    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(date_str) + context_chars)

    context = text[start:end].strip()
    # Clean up
    context = ' '.join(context.split())

    return context


def sanitize_text(text):
    """Remove control characters and non-printable characters."""
    return ''.join(c for c in text if c.isprintable() or c in ' \t\n')


def process_file(txt_file):
    """Process a single file - designed for parallel execution."""
    global _worker_nlp
    try:
        text = txt_file.read_text(encoding='utf-8', errors='ignore')
        text = sanitize_text(text)

        # Extract dates using both methods
        regex_dates = extract_dates_regex(text)
        spacy_dates = extract_dates_spacy(text, _worker_nlp)

        # Combine and deduplicate
        all_date_strs = list(set(regex_dates + spacy_dates))

        # Parse dates
        parsed_dates = []
        for date_str in all_date_strs:
            parsed = parse_date(date_str)
            if parsed:
                context = extract_context(text, date_str)
                parsed_dates.append({
                    'original': date_str,
                    'parsed': parsed.isoformat(),
                    'year': parsed.year,
                    'month': parsed.month,
                    'context': context[:200]  # Limit context length
                })

        if parsed_dates:
            return (str(txt_file), parsed_dates)
    except Exception:
        pass
    return None


def main():
    print("=== TIMELINE EXTRACTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all .txt files
    print("Scanning for .txt files...")
    txt_files = list(BASE_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files\n")

    if not txt_files:
        print(f"No .txt files found in {BASE_DIR}")
        return

    # Process all files in parallel
    print("Extracting dates from files...")
    file_dates = {}
    total_files = len(txt_files)

    num_workers = min(multiprocessing.cpu_count(), total_files) if total_files > 0 else 1
    print(f"Using {num_workers} parallel workers\n")

    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(process_file, f): f for f in txt_files}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                file_path, dates = result
                file_dates[file_path] = dates

            completed += 1
            if completed % 100 == 0 or completed == total_files:
                print(f"Progress: {completed}/{total_files} files ({100*completed/total_files:.1f}%)")

    print(f"\nFiles with dates found: {len(file_dates)}")

    # Aggregate date counts
    year_counts = Counter()
    month_counts = Counter()  # (year, month) tuples
    all_dates = []

    for file_path, dates in file_dates.items():
        for date_info in dates:
            year_counts[date_info['year']] += 1
            month_counts[(date_info['year'], date_info['month'])] += 1
            all_dates.append({
                'file': file_path,
                **date_info
            })

    total_dates = len(all_dates)
    print(f"Total date mentions: {total_dates:,}")
    print(f"Unique years: {len(year_counts)}")

    # Display timeline summary
    print("\n" + "="*60)
    print("TIMELINE SUMMARY")
    print("="*60)

    print("\n--- Date Mentions by Year ---")
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        bar = '#' * min(50, count // 10)
        print(f"{year}: {count:5d} {bar}")

    # Find peak periods
    print("\n--- Top 10 Years by Mentions ---")
    for i, (year, count) in enumerate(year_counts.most_common(10), 1):
        print(f"{i:2d}. {year}: {count} mentions")

    print("\n--- Top 10 Months by Mentions ---")
    for i, ((year, month), count) in enumerate(month_counts.most_common(10), 1):
        month_name = datetime(year, month, 1).strftime('%B %Y')
        print(f"{i:2d}. {month_name}: {count} mentions")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save raw extraction data
    with open(OUTPUT_DIR / "extracted_dates.json", 'w', encoding='utf-8') as f:
        json.dump(file_dates, f, indent=2, default=str)
    print(f"  Raw extractions: {OUTPUT_DIR / 'extracted_dates.json'}")

    # Save all dates as CSV
    dates_df = pd.DataFrame(all_dates)
    dates_df.to_csv(OUTPUT_DIR / "all_dates.csv", index=False)
    print(f"  All dates: {OUTPUT_DIR / 'all_dates.csv'}")

    # Save year counts
    year_df = pd.DataFrame([
        {'year': year, 'mentions': count}
        for year, count in sorted(year_counts.items())
    ])
    year_df.to_csv(OUTPUT_DIR / "year_counts.csv", index=False)
    print(f"  Year counts: {OUTPUT_DIR / 'year_counts.csv'}")

    # Save month counts
    month_df = pd.DataFrame([
        {'year': year, 'month': month, 'mentions': count}
        for (year, month), count in sorted(month_counts.items())
    ])
    month_df.to_csv(OUTPUT_DIR / "month_counts.csv", index=False)
    print(f"  Month counts: {OUTPUT_DIR / 'month_counts.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Year timeline bar chart
    if year_counts:
        fig, ax = plt.subplots(figsize=(14, 6))

        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]

        bars = ax.bar(years, counts, color='steelblue', edgecolor='navy', alpha=0.8)

        # Highlight peak years
        max_count = max(counts)
        for bar, count in zip(bars, counts):
            if count > max_count * 0.7:
                bar.set_color('#e74c3c')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Mentions', fontsize=12)
        ax.set_title('Date Mentions by Year', fontsize=14)

        # Rotate x labels if many years
        if len(years) > 20:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "timeline_by_year.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Year timeline: {OUTPUT_DIR / 'timeline_by_year.pdf'}")

    # Monthly heatmap for top years
    if month_counts:
        # Get top 10 years
        top_years = [y for y, _ in year_counts.most_common(15)]
        top_years = sorted(top_years)

        # Create matrix
        months = list(range(1, 13))
        matrix = []
        for year in top_years:
            row = [month_counts.get((year, m), 0) for m in months]
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(top_years)))
        ax.set_yticklabels(top_years)

        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title('Date Mentions Heatmap (Top 15 Years)', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Mentions')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "monthly_heatmap.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Monthly heatmap: {OUTPUT_DIR / 'monthly_heatmap.pdf'}")

    # Cumulative timeline
    if year_counts:
        fig, ax = plt.subplots(figsize=(12, 6))

        years = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years]
        cumulative = []
        total = 0
        for c in counts:
            total += c
            cumulative.append(total)

        ax.fill_between(years, cumulative, alpha=0.3, color='steelblue')
        ax.plot(years, cumulative, color='steelblue', linewidth=2)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Mentions', fontsize=12)
        ax.set_title('Cumulative Date Mentions Over Time', fontsize=14)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cumulative_timeline.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Cumulative timeline: {OUTPUT_DIR / 'cumulative_timeline.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed: {len(txt_files)}")
    print(f"Files with dates: {len(file_dates)}")
    print(f"Total date mentions: {total_dates:,}")
    print(f"Year range: {min(year_counts.keys())} - {max(year_counts.keys())}")
    print(f"Peak year: {year_counts.most_common(1)[0][0]} ({year_counts.most_common(1)[0][1]} mentions)")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nTimeline extraction complete!")


if __name__ == "__main__":
    main()
