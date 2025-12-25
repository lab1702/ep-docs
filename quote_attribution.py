#!/usr/bin/env python3
"""
Quote Attribution Script - Extract "who said what" from documents.

Identifies quotes, testimony, and attributed statements in depositions
and other legal documents.

Usage:
    python quote_attribution.py                  # Extract quotes
    python quote_attribution.py --min-length 50  # Minimum quote length
    python quote_attribution.py --search "EPSTEIN"  # Search specific speaker
"""

from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import re

import pandas as pd

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./quote_attribution_output")
NAMES_FILE = Path("./extracted_names.json")

# Quote patterns
DEPOSITION_QA = re.compile(
    r'(?:^|\n)\s*Q[.:\s]+(.{20,500}?)\s*(?:\n\s*)?A[.:\s]+(.{20,500}?)(?=\n\s*Q[.:\s]|\n\n|\Z)',
    re.MULTILINE | re.DOTALL
)

# "PERSON said" patterns
SAID_PATTERN = re.compile(
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\s+(?:said|stated|testified|replied|answered|explained|noted|added|claimed|alleged|admitted|denied|recalled|remembered|mentioned|indicated|suggested|asked|questioned|inquired)[,:]?\s*["\'](.{20,300}?)["\']',
    re.DOTALL
)

# According to PERSON patterns
ACCORDING_PATTERN = re.compile(
    r'[Aa]ccording to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})[,:]?\s*["\']?(.{20,300}?)["\']?(?:\.|$)',
    re.DOTALL
)

# Direct quotes with attribution after
QUOTE_THEN_ATTR = re.compile(
    r'["\'](.{20,300}?)["\']\s*(?:said|stated|testified|according to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
    re.DOTALL
)

# THE WITNESS / THE COURT patterns (common in depositions)
WITNESS_PATTERN = re.compile(
    r'(?:THE WITNESS|THE DEPONENT|MR\.|MS\.|MRS\.)\s*([A-Z][A-Z\s]+)?[:\s]+(.{20,500}?)(?=\n\s*(?:Q\.|THE |MR\.|MS\.|BY )|$)',
    re.MULTILINE | re.DOTALL
)

# BY ATTORNEY patterns
BY_ATTORNEY_PATTERN = re.compile(
    r'BY\s+(MR\.|MS\.|MRS\.)\s*([A-Z][A-Z]+)[:\s]*\n?\s*Q[.:\s]+(.{20,500}?)(?=\n\s*A[.:\s])',
    re.MULTILINE | re.DOTALL
)


def load_known_names():
    """Load known person names for validation."""
    if not NAMES_FILE.exists():
        return set()

    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        file_names = json.load(f)

    all_names = set()
    for names in file_names.values():
        all_names.update(names)

    return all_names


def clean_quote(text):
    """Clean up extracted quote text."""
    # Normalize whitespace
    text = ' '.join(text.split())
    # Remove leading/trailing punctuation
    text = text.strip('.,;:"\' ')
    return text


def extract_deposition_qa(text):
    """Extract Q&A pairs from deposition format."""
    quotes = []

    for match in DEPOSITION_QA.finditer(text):
        question = clean_quote(match.group(1))
        answer = clean_quote(match.group(2))

        if len(question) >= 20 and len(answer) >= 20:
            quotes.append({
                'type': 'deposition_qa',
                'speaker': 'THE WITNESS',
                'question': question,
                'answer': answer,
                'quote': answer,  # The answer is the main quote
                'context': f"Q: {question[:100]}..."
            })

    return quotes


def extract_said_quotes(text):
    """Extract quotes with 'said/stated/testified' attribution."""
    quotes = []

    for match in SAID_PATTERN.finditer(text):
        speaker = match.group(1).strip()
        quote = clean_quote(match.group(2))

        if len(quote) >= 20:
            quotes.append({
                'type': 'said_attribution',
                'speaker': speaker.upper(),
                'quote': quote,
                'context': match.group(0)[:200]
            })

    return quotes


def extract_according_to(text):
    """Extract 'according to PERSON' quotes."""
    quotes = []

    for match in ACCORDING_PATTERN.finditer(text):
        speaker = match.group(1).strip()
        quote = clean_quote(match.group(2))

        if len(quote) >= 20:
            quotes.append({
                'type': 'according_to',
                'speaker': speaker.upper(),
                'quote': quote,
                'context': match.group(0)[:200]
            })

    return quotes


def extract_witness_statements(text):
    """Extract THE WITNESS statements from depositions."""
    quotes = []

    for match in WITNESS_PATTERN.finditer(text):
        speaker_name = match.group(1)
        statement = clean_quote(match.group(2))

        if len(statement) >= 20:
            speaker = f"THE WITNESS ({speaker_name.strip()})" if speaker_name else "THE WITNESS"
            quotes.append({
                'type': 'witness_statement',
                'speaker': speaker,
                'quote': statement,
                'context': match.group(0)[:200]
            })

    return quotes


def extract_attorney_questions(text):
    """Extract questions by named attorneys."""
    quotes = []

    for match in BY_ATTORNEY_PATTERN.finditer(text):
        title = match.group(1)
        name = match.group(2).strip()
        question = clean_quote(match.group(3))

        if len(question) >= 20:
            quotes.append({
                'type': 'attorney_question',
                'speaker': f"{title} {name}",
                'quote': question,
                'context': match.group(0)[:200]
            })

    return quotes


def extract_all_quotes(text, min_length=30):
    """Extract all quotes from a document."""
    all_quotes = []

    # Try different extraction methods
    all_quotes.extend(extract_deposition_qa(text))
    all_quotes.extend(extract_said_quotes(text))
    all_quotes.extend(extract_according_to(text))
    all_quotes.extend(extract_witness_statements(text))
    all_quotes.extend(extract_attorney_questions(text))

    # Filter by minimum length
    all_quotes = [q for q in all_quotes if len(q.get('quote', '')) >= min_length]

    return all_quotes


def process_file(file_path, min_length):
    """Process a single file for quotes."""
    try:
        text = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        quotes = extract_all_quotes(text, min_length)

        # Add file info
        for q in quotes:
            q['file'] = str(file_path)
            q['filename'] = Path(file_path).name

        return quotes
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract attributed quotes")
    parser.add_argument("--min-length", type=int, default=30,
                        help="Minimum quote length (default: 30)")
    parser.add_argument("--search", type=str, default=None,
                        help="Search for quotes by specific speaker")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top speakers to show (default: 50)")
    args = parser.parse_args()

    print("=== QUOTE ATTRIBUTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load known names
    print("Loading known names...")
    known_names = load_known_names()
    print(f"  Known names: {len(known_names)}")

    # Find all .txt files
    print("\nScanning for .txt files...")
    txt_files = list(BASE_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files\n")

    if not txt_files:
        print(f"No .txt files found in {BASE_DIR}")
        return

    # Process all documents
    print(f"Extracting quotes (min length: {args.min_length})...")
    all_quotes = []

    for i, txt_file in enumerate(txt_files):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(txt_files)}")

        quotes = process_file(txt_file, args.min_length)
        all_quotes.extend(quotes)

    print(f"\nTotal quotes extracted: {len(all_quotes)}")

    # Aggregate by speaker
    speaker_quotes = defaultdict(list)
    for q in all_quotes:
        speaker_quotes[q['speaker']].append(q)

    print(f"Unique speakers: {len(speaker_quotes)}")

    # Count quote types
    type_counts = Counter(q['type'] for q in all_quotes)

    # Handle search mode
    if args.search:
        search_upper = args.search.upper()
        matching_speakers = [s for s in speaker_quotes if search_upper in s]

        print(f"\n--- Quotes matching '{args.search}' ---")

        for speaker in matching_speakers:
            quotes = speaker_quotes[speaker]
            print(f"\n{speaker} ({len(quotes)} quotes):")
            for q in quotes[:10]:
                print(f"\n  [{q['type']}] {q['filename']}")
                print(f"  \"{q['quote'][:200]}...\"" if len(q['quote']) > 200 else f"  \"{q['quote']}\"")

        return

    # Display results
    print("\n" + "="*60)
    print("QUOTE ANALYSIS")
    print("="*60)

    print("\n--- Quote Types ---")
    for qtype, count in type_counts.most_common():
        print(f"  {qtype}: {count}")

    # Top speakers by quote count
    speaker_counts = [(s, len(qs)) for s, qs in speaker_quotes.items()]
    speaker_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Top {args.top} Speakers by Quote Count ---")
    for i, (speaker, count) in enumerate(speaker_counts[:args.top], 1):
        print(f"{i:3d}. {speaker}: {count} quotes")

    # Sample quotes from top speakers
    print("\n--- Sample Quotes from Top Speakers ---")
    for speaker, count in speaker_counts[:10]:
        quotes = speaker_quotes[speaker]
        print(f"\n{speaker} ({count} quotes):")
        for q in quotes[:3]:
            quote_preview = q['quote'][:150] + "..." if len(q['quote']) > 150 else q['quote']
            print(f"  - \"{quote_preview}\"")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save all quotes
    quotes_df = pd.DataFrame(all_quotes)
    if not quotes_df.empty:
        quotes_df.to_csv(OUTPUT_DIR / "all_quotes.csv", index=False)
        print(f"  All quotes: {OUTPUT_DIR / 'all_quotes.csv'}")

    # Save speaker summary
    speaker_summary = []
    for speaker, quotes in speaker_quotes.items():
        quote_types = Counter(q['type'] for q in quotes)
        speaker_summary.append({
            'speaker': speaker,
            'quote_count': len(quotes),
            'files': len(set(q['file'] for q in quotes)),
            'primary_type': quote_types.most_common(1)[0][0] if quote_types else '',
            'sample_quote': quotes[0]['quote'][:200] if quotes else ''
        })

    speaker_df = pd.DataFrame(speaker_summary)
    speaker_df = speaker_df.sort_values('quote_count', ascending=False)
    speaker_df.to_csv(OUTPUT_DIR / "speaker_summary.csv", index=False)
    print(f"  Speaker summary: {OUTPUT_DIR / 'speaker_summary.csv'}")

    # Save quotes grouped by speaker
    speaker_data = {}
    for speaker, quotes in speaker_quotes.items():
        speaker_data[speaker] = {
            'quote_count': len(quotes),
            'quotes': [
                {
                    'quote': q['quote'],
                    'type': q['type'],
                    'file': q['filename']
                }
                for q in quotes[:100]  # Limit per speaker
            ]
        }

    with open(OUTPUT_DIR / "quotes_by_speaker.json", 'w', encoding='utf-8') as f:
        json.dump(speaker_data, f, indent=2)
    print(f"  Quotes by speaker: {OUTPUT_DIR / 'quotes_by_speaker.json'}")

    # Save type breakdown
    type_df = pd.DataFrame([
        {'type': t, 'count': c}
        for t, c in type_counts.most_common()
    ])
    type_df.to_csv(OUTPUT_DIR / "quote_types.csv", index=False)
    print(f"  Quote types: {OUTPUT_DIR / 'quote_types.csv'}")

    # Match speakers to known names
    matched_speakers = []
    for speaker in speaker_quotes:
        # Check if any known name matches
        for name in known_names:
            if name in speaker or speaker in name:
                matched_speakers.append({
                    'speaker': speaker,
                    'matched_name': name,
                    'quote_count': len(speaker_quotes[speaker])
                })
                break

    if matched_speakers:
        matched_df = pd.DataFrame(matched_speakers)
        matched_df = matched_df.sort_values('quote_count', ascending=False)
        matched_df.to_csv(OUTPUT_DIR / "matched_speakers.csv", index=False)
        print(f"  Matched speakers: {OUTPUT_DIR / 'matched_speakers.csv'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents processed: {len(txt_files)}")
    print(f"Total quotes extracted: {len(all_quotes)}")
    print(f"Unique speakers: {len(speaker_quotes)}")
    print(f"Speakers matched to known names: {len(matched_speakers)}")

    if speaker_counts:
        print(f"\nMost quoted: {speaker_counts[0][0]} ({speaker_counts[0][1]} quotes)")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nTip: Use --search 'NAME' to find quotes by a specific speaker")
    print("\nQuote attribution complete!")


if __name__ == "__main__":
    main()
