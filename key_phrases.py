#!/usr/bin/env python3
"""
Key Phrase Extraction Script - Extract significant phrases from documents.

Uses TF-IDF to identify important n-grams and phrases across the corpus.
Optionally uses RAKE (Rapid Automatic Keyword Extraction) for comparison.

Usage:
    python key_phrases.py                    # Extract key phrases
    python key_phrases.py --top 200          # Top 200 phrases
    python key_phrases.py --ngram-max 4      # Up to 4-grams
    python key_phrases.py --per-doc          # Also extract per-document phrases
"""

from pathlib import Path
from collections import Counter
import argparse
import json
import re

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./key_phrases_output")

# Stop words to filter out
LEGAL_STOP_WORDS = {
    'plaintiff', 'defendant', 'court', 'case', 'matter', 'pursuant',
    'herein', 'thereof', 'hereby', 'wherein', 'therefore', 'wherefore',
    'exhibit', 'page', 'document', 'file', 'record', 'evidence',
    'testimony', 'deposition', 'witness', 'counsel', 'attorney',
    'motion', 'order', 'judgment', 'ruling', 'appeal', 'trial',
    'said', 'also', 'would', 'could', 'shall', 'may', 'must',
    'united states', 'state', 'county', 'district'
}


def load_documents(base_dir, max_docs=None):
    """Load all text files from the base directory."""
    print("Loading documents...")
    documents = []
    file_paths = []

    txt_files = sorted(base_dir.rglob("*.txt"))
    if max_docs:
        txt_files = txt_files[:max_docs]

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            if len(text) >= 100:
                documents.append(text)
                file_paths.append(str(txt_file))
        except Exception:
            continue

    print(f"Loaded {len(documents)} documents\n")
    return documents, file_paths


def sanitize_text(text):
    """Clean text for processing."""
    # Remove control characters
    text = ''.join(c for c in text if c.isprintable() or c in ' \t\n')
    # Normalize whitespace
    text = ' '.join(text.split())
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def is_valid_phrase(phrase):
    """Check if a phrase is valid (not just stopwords or noise)."""
    words = phrase.split()

    # Must have at least one word with 3+ characters
    if not any(len(w) >= 3 for w in words):
        return False

    # Skip if all numbers
    if all(w.isdigit() for w in words):
        return False

    # Skip if starts/ends with common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its'}
    if words[0] in stopwords or words[-1] in stopwords:
        return False

    # Skip legal stopwords
    if phrase in LEGAL_STOP_WORDS:
        return False

    return True


def extract_tfidf_phrases(documents, ngram_range=(2, 3), top_n=500, min_df=3):
    """Extract key phrases using TF-IDF."""
    print(f"Extracting phrases with TF-IDF (n-grams: {ngram_range})...")

    # Clean documents
    clean_docs = [sanitize_text(doc) for doc in documents]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=10000,
        min_df=min_df,
        max_df=0.7,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z][a-zA-Z\'-]*[a-zA-Z]\b|\b[a-zA-Z]{2,}\b'
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(clean_docs)
    except ValueError as e:
        print(f"  Error: {e}")
        return []

    feature_names = vectorizer.get_feature_names_out()

    # Get average TF-IDF scores across all documents
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    # Also count document frequency
    doc_freq = np.asarray((tfidf_matrix > 0).sum(axis=0)).flatten()

    # Combine scores
    phrase_scores = []
    for i, phrase in enumerate(feature_names):
        if is_valid_phrase(phrase):
            phrase_scores.append({
                'phrase': phrase,
                'tfidf_score': avg_scores[i],
                'doc_frequency': doc_freq[i],
                'combined_score': avg_scores[i] * np.log1p(doc_freq[i])
            })

    # Sort by combined score
    phrase_scores.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"  Found {len(phrase_scores)} valid phrases")

    return phrase_scores[:top_n], vectorizer, tfidf_matrix


def extract_document_phrases(doc_idx, tfidf_matrix, vectorizer, top_n=10):
    """Extract top phrases for a specific document."""
    feature_names = vectorizer.get_feature_names_out()

    # Get TF-IDF scores for this document
    doc_vector = tfidf_matrix[doc_idx].toarray().flatten()

    # Get top phrases
    top_indices = doc_vector.argsort()[-top_n*2:][::-1]

    phrases = []
    for idx in top_indices:
        if doc_vector[idx] > 0:
            phrase = feature_names[idx]
            if is_valid_phrase(phrase):
                phrases.append({
                    'phrase': phrase,
                    'score': doc_vector[idx]
                })
                if len(phrases) >= top_n:
                    break

    return phrases


def categorize_phrase(phrase):
    """Attempt to categorize a phrase by topic."""
    phrase_lower = phrase.lower()

    categories = {
        'Legal': ['attorney', 'lawyer', 'court', 'judge', 'trial', 'lawsuit', 'legal', 'law firm', 'prosecution', 'defense'],
        'Financial': ['bank', 'account', 'money', 'payment', 'financial', 'fund', 'investment', 'million', 'dollar'],
        'Travel': ['flight', 'plane', 'airport', 'travel', 'trip', 'flew', 'passenger', 'pilot'],
        'Property': ['house', 'home', 'property', 'island', 'estate', 'residence', 'apartment', 'mansion'],
        'Communication': ['phone', 'call', 'email', 'message', 'contact', 'conversation', 'told', 'said'],
        'Time': ['year', 'month', 'day', 'time', 'date', 'period', 'during', 'between'],
        'People': ['man', 'woman', 'girl', 'boy', 'person', 'people', 'friend', 'employee', 'staff'],
        'Location': ['new york', 'palm beach', 'florida', 'virgin islands', 'london', 'paris'],
    }

    for category, keywords in categories.items():
        if any(kw in phrase_lower for kw in keywords):
            return category

    return 'Other'


def main():
    parser = argparse.ArgumentParser(description="Extract key phrases from documents")
    parser.add_argument("--top", type=int, default=100,
                        help="Number of top phrases to extract (default: 100)")
    parser.add_argument("--ngram-min", type=int, default=2,
                        help="Minimum n-gram size (default: 2)")
    parser.add_argument("--ngram-max", type=int, default=3,
                        help="Maximum n-gram size (default: 3)")
    parser.add_argument("--min-docs", type=int, default=3,
                        help="Minimum document frequency (default: 3)")
    parser.add_argument("--per-doc", action="store_true",
                        help="Also extract per-document key phrases")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to process (for testing)")
    args = parser.parse_args()

    print("=== KEY PHRASE EXTRACTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load documents
    documents, file_paths = load_documents(BASE_DIR, args.max_docs)

    if len(documents) < 10:
        print("Error: Need at least 10 documents for key phrase extraction")
        return

    # Extract phrases
    ngram_range = (args.ngram_min, args.ngram_max)
    phrases, vectorizer, tfidf_matrix = extract_tfidf_phrases(
        documents,
        ngram_range=ngram_range,
        top_n=args.top,
        min_df=args.min_docs
    )

    if not phrases:
        print("No phrases extracted. Try adjusting parameters.")
        return

    # Categorize phrases
    for p in phrases:
        p['category'] = categorize_phrase(p['phrase'])

    # Display results
    print("\n" + "="*60)
    print("TOP KEY PHRASES")
    print("="*60)

    print(f"\n--- Top {min(50, len(phrases))} Phrases ---")
    for i, p in enumerate(phrases[:50], 1):
        print(f"{i:3d}. {p['phrase']:<40} (docs: {p['doc_frequency']:4d}, score: {p['combined_score']:.4f}) [{p['category']}]")

    # Category breakdown
    category_counts = Counter(p['category'] for p in phrases)
    print("\n--- Phrases by Category ---")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save all phrases
    phrases_df = pd.DataFrame(phrases)
    phrases_df.to_csv(OUTPUT_DIR / "key_phrases.csv", index=False)
    print(f"  Key phrases: {OUTPUT_DIR / 'key_phrases.csv'}")

    # Save phrases by category
    for category in category_counts:
        cat_phrases = [p for p in phrases if p['category'] == category]
        cat_df = pd.DataFrame(cat_phrases)
        safe_cat = category.lower().replace(' ', '_')
        cat_df.to_csv(OUTPUT_DIR / f"phrases_{safe_cat}.csv", index=False)
    print(f"  Category files: {OUTPUT_DIR / 'phrases_*.csv'}")

    # Extract per-document phrases if requested
    if args.per_doc:
        print("\nExtracting per-document phrases...")
        doc_phrases = {}

        for i, file_path in enumerate(file_paths):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(file_paths)}")

            top_doc_phrases = extract_document_phrases(i, tfidf_matrix, vectorizer, top_n=10)
            if top_doc_phrases:
                doc_phrases[file_path] = top_doc_phrases

        with open(OUTPUT_DIR / "document_phrases.json", 'w', encoding='utf-8') as f:
            json.dump(doc_phrases, f, indent=2)
        print(f"  Document phrases: {OUTPUT_DIR / 'document_phrases.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Top phrases bar chart
    fig, ax = plt.subplots(figsize=(12, 10))

    top_30 = phrases[:30]
    phrase_labels = [p['phrase'] for p in top_30]
    scores = [p['combined_score'] for p in top_30]
    colors = [plt.cm.tab10(hash(p['category']) % 10) for p in top_30]

    bars = ax.barh(range(len(phrase_labels)), scores, color=colors)
    ax.set_yticks(range(len(phrase_labels)))
    ax.set_yticklabels(phrase_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Combined Score (TF-IDF × log(doc_freq))', fontsize=11)
    ax.set_title('Top 30 Key Phrases', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_phrases_chart.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Top phrases chart: {OUTPUT_DIR / 'top_phrases_chart.pdf'}")

    # Category distribution pie chart
    fig, ax = plt.subplots(figsize=(10, 10))

    categories = [cat for cat, _ in category_counts.most_common()]
    counts = [count for _, count in category_counts.most_common()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('Key Phrases by Category', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Category distribution: {OUTPUT_DIR / 'category_distribution.pdf'}")

    # Document frequency distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    doc_freqs = [p['doc_frequency'] for p in phrases]
    ax.hist(doc_freqs, bins=30, color='steelblue', edgecolor='navy', alpha=0.8)

    ax.set_xlabel('Document Frequency', fontsize=12)
    ax.set_ylabel('Number of Phrases', fontsize=12)
    ax.set_title('Distribution of Phrase Document Frequencies', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "doc_freq_distribution.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Doc freq distribution: {OUTPUT_DIR / 'doc_freq_distribution.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents processed: {len(documents)}")
    print(f"Key phrases extracted: {len(phrases)}")
    print(f"N-gram range: {ngram_range}")
    print(f"Min document frequency: {args.min_docs}")
    print(f"\nTop 5 phrases:")
    for i, p in enumerate(phrases[:5], 1):
        print(f"  {i}. {p['phrase']} ({p['doc_frequency']} docs)")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nKey phrase extraction complete!")


if __name__ == "__main__":
    main()
