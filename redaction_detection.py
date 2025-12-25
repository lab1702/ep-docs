#!/usr/bin/env python3
"""
Redaction Detection Script - Identify redacted sections in documents.

Scans documents for redaction patterns such as blacked-out text, [REDACTED]
markers, long underscore sequences, and other common redaction indicators.

Usage:
    python redaction_detection.py                # Scan all documents
    python redaction_detection.py --top 50       # Show top 50 redacted docs
    python redaction_detection.py --show-samples # Show redaction samples
"""

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import re

import pandas as pd
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./redaction_detection_output")

# Redaction patterns
REDACTION_PATTERNS = {
    # Explicit redaction markers
    'redacted_marker': r'\[REDACTED\]|\(REDACTED\)|REDACTED',
    'sealed_marker': r'\[SEALED\]|\(SEALED\)|SEALED',
    'classified_marker': r'\[CLASSIFIED\]|\(CLASSIFIED\)|CLASSIFIED',
    'withheld_marker': r'\[WITHHELD\]|\(WITHHELD\)|WITHHELD',
    'confidential_marker': r'\[CONFIDENTIAL\]|\(CONFIDENTIAL\)',

    # Black box characters (common in PDF conversions)
    'black_boxes': r'[█▓▒░■]{3,}',

    # Underscore sequences (often used for redactions)
    'underscores': r'_{5,}',

    # X sequences
    'x_sequence': r'[Xx]{5,}',

    # Asterisk sequences
    'asterisks': r'\*{5,}',

    # Bracketed blanks
    'bracketed_blank': r'\[[\s_\.]{3,}\]|\([\s_\.]{3,}\)',

    # Dash sequences
    'dash_sequence': r'-{10,}|—{5,}',

    # Deleted/removed markers
    'deleted_marker': r'\[DELETED\]|\(DELETED\)|DELETED',

    # Exemption markers (FOIA)
    'exemption_marker': r'\(b\)\([1-9]\)|\[b\]\[[1-9]\]|Exemption \d',

    # Blacked out with description
    'blacked_out': r'blacked[\s-]?out|redaction|obscured',
}

# Compile patterns
COMPILED_PATTERNS = {name: re.compile(pattern, re.IGNORECASE)
                     for name, pattern in REDACTION_PATTERNS.items()}


def detect_redactions(text):
    """Detect redaction patterns in text."""
    redactions = {
        'counts': Counter(),
        'samples': defaultdict(list),
        'positions': []
    }

    for pattern_name, pattern in COMPILED_PATTERNS.items():
        matches = list(pattern.finditer(text))
        redactions['counts'][pattern_name] = len(matches)

        # Store sample matches (first 5)
        for match in matches[:5]:
            sample = match.group()
            if len(sample) > 50:
                sample = sample[:50] + "..."
            redactions['samples'][pattern_name].append(sample)

        # Store positions for analysis
        for match in matches:
            redactions['positions'].append({
                'type': pattern_name,
                'start': match.start(),
                'end': match.end(),
                'length': match.end() - match.start()
            })

    return redactions


def analyze_document(file_path):
    """Analyze a document for redactions."""
    try:
        text = Path(file_path).read_text(encoding='utf-8', errors='ignore')

        # Basic stats
        doc_length = len(text)
        line_count = text.count('\n') + 1

        # Detect redactions
        redactions = detect_redactions(text)

        # Calculate redaction metrics
        total_redactions = sum(redactions['counts'].values())
        total_redaction_chars = sum(p['length'] for p in redactions['positions'])
        redaction_percentage = (total_redaction_chars / doc_length * 100) if doc_length > 0 else 0

        return {
            'file': str(file_path),
            'doc_length': doc_length,
            'line_count': line_count,
            'total_redactions': total_redactions,
            'redaction_chars': total_redaction_chars,
            'redaction_percentage': redaction_percentage,
            'pattern_counts': dict(redactions['counts']),
            'samples': dict(redactions['samples']),
            'has_redactions': total_redactions > 0
        }

    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'has_redactions': False
        }


def main():
    parser = argparse.ArgumentParser(description="Detect redactions in documents")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top redacted documents to show (default: 30)")
    parser.add_argument("--show-samples", action="store_true",
                        help="Show sample redaction text")
    parser.add_argument("--min-redactions", type=int, default=1,
                        help="Minimum redactions to include in output (default: 1)")
    args = parser.parse_args()

    print("=== REDACTION DETECTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all .txt files
    print("Scanning for .txt files...")
    txt_files = list(BASE_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} .txt files\n")

    if not txt_files:
        print(f"No .txt files found in {BASE_DIR}")
        return

    # Analyze all documents
    print("Analyzing documents for redactions...")
    results = []

    for i, txt_file in enumerate(txt_files):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(txt_files)}")
        result = analyze_document(txt_file)
        results.append(result)

    # Aggregate statistics
    docs_with_redactions = [r for r in results if r.get('has_redactions', False)]
    total_redactions = sum(r.get('total_redactions', 0) for r in results)

    # Count by pattern type
    pattern_totals = Counter()
    for r in results:
        pattern_counts = r.get('pattern_counts', {})
        pattern_totals.update(pattern_counts)

    print(f"\nDocuments analyzed: {len(results)}")
    print(f"Documents with redactions: {len(docs_with_redactions)}")
    print(f"Total redaction instances: {total_redactions}")

    # Display results
    print("\n" + "="*60)
    print("REDACTION ANALYSIS")
    print("="*60)

    print("\n--- Redaction Patterns Found ---")
    for pattern, count in pattern_totals.most_common():
        if count > 0:
            pattern_label = pattern.replace('_', ' ').title()
            print(f"  {pattern_label}: {count:,}")

    # Sort by most redacted
    redacted_docs = sorted(
        [r for r in results if r.get('total_redactions', 0) >= args.min_redactions],
        key=lambda x: x.get('total_redactions', 0),
        reverse=True
    )

    print(f"\n--- Top {args.top} Most Redacted Documents ---")
    for i, doc in enumerate(redacted_docs[:args.top], 1):
        filename = Path(doc['file']).name
        count = doc.get('total_redactions', 0)
        pct = doc.get('redaction_percentage', 0)
        print(f"{i:3d}. {filename[:50]:<50} {count:5d} redactions ({pct:.1f}%)")

        if args.show_samples and doc.get('samples'):
            for pattern, samples in doc['samples'].items():
                if samples:
                    print(f"       [{pattern}]: {samples[0]}")

    # Redaction severity breakdown
    print("\n--- Documents by Redaction Severity ---")
    severity_counts = {
        'None (0)': sum(1 for r in results if r.get('total_redactions', 0) == 0),
        'Light (1-5)': sum(1 for r in results if 1 <= r.get('total_redactions', 0) <= 5),
        'Moderate (6-20)': sum(1 for r in results if 6 <= r.get('total_redactions', 0) <= 20),
        'Heavy (21-50)': sum(1 for r in results if 21 <= r.get('total_redactions', 0) <= 50),
        'Severe (50+)': sum(1 for r in results if r.get('total_redactions', 0) > 50),
    }
    for severity, count in severity_counts.items():
        pct = count / len(results) * 100
        print(f"  {severity}: {count} documents ({pct:.1f}%)")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save summary for all redacted documents
    redacted_summary = []
    for r in redacted_docs:
        summary = {
            'file': r['file'],
            'filename': Path(r['file']).name,
            'total_redactions': r.get('total_redactions', 0),
            'redaction_chars': r.get('redaction_chars', 0),
            'redaction_percentage': r.get('redaction_percentage', 0),
            'doc_length': r.get('doc_length', 0),
        }
        # Add pattern counts as columns
        for pattern in REDACTION_PATTERNS.keys():
            summary[pattern] = r.get('pattern_counts', {}).get(pattern, 0)
        redacted_summary.append(summary)

    summary_df = pd.DataFrame(redacted_summary)
    summary_df.to_csv(OUTPUT_DIR / "redacted_documents.csv", index=False)
    print(f"  Redacted documents: {OUTPUT_DIR / 'redacted_documents.csv'}")

    # Save pattern statistics
    pattern_df = pd.DataFrame([
        {'pattern': p, 'count': c}
        for p, c in pattern_totals.most_common()
    ])
    pattern_df.to_csv(OUTPUT_DIR / "pattern_counts.csv", index=False)
    print(f"  Pattern counts: {OUTPUT_DIR / 'pattern_counts.csv'}")

    # Save severity breakdown
    severity_df = pd.DataFrame([
        {'severity': s, 'count': c, 'percentage': c/len(results)*100}
        for s, c in severity_counts.items()
    ])
    severity_df.to_csv(OUTPUT_DIR / "severity_breakdown.csv", index=False)
    print(f"  Severity breakdown: {OUTPUT_DIR / 'severity_breakdown.csv'}")

    # Save detailed results as JSON
    detailed_results = [r for r in results if r.get('has_redactions', False)]
    with open(OUTPUT_DIR / "detailed_redactions.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"  Detailed results: {OUTPUT_DIR / 'detailed_redactions.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Pattern distribution bar chart
    if pattern_totals:
        fig, ax = plt.subplots(figsize=(12, 6))

        patterns = [p.replace('_', ' ').title() for p, c in pattern_totals.most_common() if c > 0]
        counts = [c for p, c in pattern_totals.most_common() if c > 0]

        bars = ax.bar(range(len(patterns)), counts, color='steelblue', edgecolor='navy')
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels(patterns, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Redaction Patterns Found Across All Documents', fontsize=14)

        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{count:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pattern_distribution.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Pattern distribution: {OUTPUT_DIR / 'pattern_distribution.pdf'}")

    # Severity pie chart
    fig, ax = plt.subplots(figsize=(10, 10))

    severities = list(severity_counts.keys())
    counts = list(severity_counts.values())
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']

    # Only show non-zero slices
    non_zero = [(s, c, col) for s, c, col in zip(severities, counts, colors) if c > 0]
    if non_zero:
        labels, values, cols = zip(*non_zero)
        ax.pie(values, labels=labels, colors=cols, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title('Documents by Redaction Severity', fontsize=14)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "severity_pie_chart.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Severity pie chart: {OUTPUT_DIR / 'severity_pie_chart.pdf'}")

    # Top redacted documents bar chart
    if redacted_docs:
        fig, ax = plt.subplots(figsize=(14, 8))

        top_docs = redacted_docs[:20]
        filenames = [Path(d['file']).name[:40] for d in top_docs]
        redaction_counts = [d.get('total_redactions', 0) for d in top_docs]

        bars = ax.barh(range(len(filenames)), redaction_counts, color='#e74c3c', edgecolor='darkred')
        ax.set_yticks(range(len(filenames)))
        ax.set_yticklabels(filenames, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Redactions', fontsize=12)
        ax.set_title('Top 20 Most Redacted Documents', fontsize=14)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_redacted_docs.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Top redacted docs: {OUTPUT_DIR / 'top_redacted_docs.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents analyzed: {len(results)}")
    print(f"Documents with redactions: {len(docs_with_redactions)} ({len(docs_with_redactions)/len(results)*100:.1f}%)")
    print(f"Total redaction instances: {total_redactions:,}")

    if pattern_totals:
        most_common = pattern_totals.most_common(1)[0]
        print(f"Most common pattern: {most_common[0].replace('_', ' ')} ({most_common[1]:,} instances)")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nRedaction detection complete!")


if __name__ == "__main__":
    main()
