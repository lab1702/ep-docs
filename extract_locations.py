#!/usr/bin/env python3
"""
Location Extraction Script - Extract geographic entities from text files using spaCy NER.

Extracts GPE (countries, cities, states), LOC (non-GPE locations), and FAC (facilities)
from all documents and generates location frequency analysis.

Setup:
    python -m spacy download en_core_web_sm

Usage:
    python extract_locations.py
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import multiprocessing
import json

import spacy
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./location_extraction_output")

# Global variable for worker processes
_worker_nlp = None


def _init_worker():
    """Initialize spaCy model for each worker process."""
    global _worker_nlp
    _worker_nlp = spacy.load('en_core_web_sm')
    _worker_nlp.max_length = 2_000_000


def extract_locations(text, nlp):
    """Extract location entities from text using spaCy NER."""
    max_chunk = 100000
    locations = {
        'GPE': set(),   # Countries, cities, states
        'LOC': set(),   # Non-GPE locations (mountains, bodies of water)
        'FAC': set()    # Facilities (buildings, airports, highways)
    }

    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in locations:
                    # Clean up the location name
                    name = ' '.join(ent.text.split()).strip()
                    if len(name) >= 2:
                        locations[ent.label_].add(name)
        except Exception:
            continue

    return locations


def filter_locations(locations):
    """Filter out likely false positives."""
    filtered = {
        'GPE': set(),
        'LOC': set(),
        'FAC': set()
    }

    # Common false positives to exclude
    exclude_terms = {
        'i', 'we', 'you', 'he', 'she', 'it', 'they', 'mr', 'mrs', 'ms', 'dr',
        'yes', 'no', 'ok', 'okay', 'see', 'page', 'exhibit', 'document',
        'plaintiff', 'defendant', 'court', 'case', 'matter', 'pursuant',
        'section', 'chapter', 'article', 'item', 'number', 'date'
    }

    for loc_type, names in locations.items():
        for name in names:
            name_lower = name.lower()

            # Skip if it's a common false positive
            if name_lower in exclude_terms:
                continue

            # Skip if all uppercase and very short (likely acronym)
            if name.isupper() and len(name) <= 3:
                continue

            # Skip if contains only digits
            if name.replace(' ', '').isdigit():
                continue

            # Skip very long strings (likely parsing errors)
            if len(name) > 100:
                continue

            filtered[loc_type].add(name)

    return filtered


def process_file(txt_file):
    """Process a single file - designed for parallel execution."""
    global _worker_nlp
    try:
        text = txt_file.read_text(encoding='utf-8', errors='ignore')
        locations = extract_locations(text, _worker_nlp)
        locations = filter_locations(locations)

        # Combine all location types for this file
        all_locations = {
            'GPE': list(locations['GPE']),
            'LOC': list(locations['LOC']),
            'FAC': list(locations['FAC'])
        }

        if any(all_locations.values()):
            return (str(txt_file), all_locations)
    except Exception:
        pass
    return None


def main():
    print("=== LOCATION EXTRACTION (spaCy NER) ===\n")

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
    print("Extracting locations from files...")
    file_locations = {}
    total_files = len(txt_files)

    num_workers = min(multiprocessing.cpu_count(), total_files) if total_files > 0 else 1
    print(f"Using {num_workers} parallel workers\n")

    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(process_file, f): f for f in txt_files}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                file_path, locations = result
                file_locations[file_path] = locations

            completed += 1
            if completed % 100 == 0 or completed == total_files:
                print(f"Progress: {completed}/{total_files} files ({100*completed/total_files:.1f}%)")

    print(f"\nFiles with locations found: {len(file_locations)}")

    # Aggregate location counts by type
    gpe_counts = Counter()
    loc_counts = Counter()
    fac_counts = Counter()

    for locations in file_locations.values():
        gpe_counts.update(locations.get('GPE', []))
        loc_counts.update(locations.get('LOC', []))
        fac_counts.update(locations.get('FAC', []))

    print(f"\nUnique GPE (countries/cities/states): {len(gpe_counts)}")
    print(f"Unique LOC (geographic features): {len(loc_counts)}")
    print(f"Unique FAC (facilities): {len(fac_counts)}")

    # Display top locations
    print("\n" + "="*60)
    print("TOP LOCATIONS BY CATEGORY")
    print("="*60)

    print("\n--- Top 30 GPE (Countries/Cities/States) ---")
    for i, (loc, count) in enumerate(gpe_counts.most_common(30), 1):
        print(f"{i:2d}. {loc}: {count} mentions")

    print("\n--- Top 20 LOC (Geographic Features) ---")
    for i, (loc, count) in enumerate(loc_counts.most_common(20), 1):
        print(f"{i:2d}. {loc}: {count} mentions")

    print("\n--- Top 20 FAC (Facilities) ---")
    for i, (loc, count) in enumerate(fac_counts.most_common(20), 1):
        print(f"{i:2d}. {loc}: {count} mentions")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save raw extraction data
    with open(OUTPUT_DIR / "extracted_locations.json", 'w', encoding='utf-8') as f:
        json.dump(file_locations, f, indent=2)
    print(f"  Raw extractions: {OUTPUT_DIR / 'extracted_locations.json'}")

    # Save GPE frequency table
    gpe_df = pd.DataFrame([
        {'location': loc, 'type': 'GPE', 'mentions': count}
        for loc, count in gpe_counts.most_common()
    ])
    gpe_df.to_csv(OUTPUT_DIR / "gpe_frequencies.csv", index=False)
    print(f"  GPE frequencies: {OUTPUT_DIR / 'gpe_frequencies.csv'}")

    # Save LOC frequency table
    loc_df = pd.DataFrame([
        {'location': loc, 'type': 'LOC', 'mentions': count}
        for loc, count in loc_counts.most_common()
    ])
    loc_df.to_csv(OUTPUT_DIR / "loc_frequencies.csv", index=False)
    print(f"  LOC frequencies: {OUTPUT_DIR / 'loc_frequencies.csv'}")

    # Save FAC frequency table
    fac_df = pd.DataFrame([
        {'location': loc, 'type': 'FAC', 'mentions': count}
        for loc, count in fac_counts.most_common()
    ])
    fac_df.to_csv(OUTPUT_DIR / "fac_frequencies.csv", index=False)
    print(f"  FAC frequencies: {OUTPUT_DIR / 'fac_frequencies.csv'}")

    # Save combined frequency table
    all_locations_df = pd.concat([gpe_df, loc_df, fac_df], ignore_index=True)
    all_locations_df = all_locations_df.sort_values('mentions', ascending=False)
    all_locations_df.to_csv(OUTPUT_DIR / "all_locations.csv", index=False)
    print(f"  All locations: {OUTPUT_DIR / 'all_locations.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Top GPE bar chart
    if len(gpe_counts) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_gpe = gpe_counts.most_common(25)
        locations = [loc for loc, _ in top_gpe]
        counts = [count for _, count in top_gpe]

        bars = ax.barh(range(len(locations)), counts, color='steelblue')
        ax.set_yticks(range(len(locations)))
        ax.set_yticklabels(locations)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Mentions')
        ax.set_title('Top 25 Geographic Locations (Countries/Cities/States)')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_gpe_chart.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  GPE chart: {OUTPUT_DIR / 'top_gpe_chart.pdf'}")

    # Location type distribution pie chart
    type_totals = {
        'GPE (Countries/Cities)': sum(gpe_counts.values()),
        'LOC (Geographic)': sum(loc_counts.values()),
        'FAC (Facilities)': sum(fac_counts.values())
    }

    if sum(type_totals.values()) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = [f"{k}\n({v:,})" for k, v in type_totals.items()]
        sizes = list(type_totals.values())
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title('Location Types Distribution')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "location_types_pie.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Type distribution: {OUTPUT_DIR / 'location_types_pie.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed: {len(txt_files)}")
    print(f"Files with locations: {len(file_locations)}")
    print(f"Total GPE mentions: {sum(gpe_counts.values()):,}")
    print(f"Total LOC mentions: {sum(loc_counts.values()):,}")
    print(f"Total FAC mentions: {sum(fac_counts.values()):,}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nLocation extraction complete!")


if __name__ == "__main__":
    main()
