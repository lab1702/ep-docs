#!/usr/bin/env python3
"""
Word Frequency Analysis Script - Analyze vocabulary patterns across documents.

Counts word frequencies, identifies distinctive terms, and generates
word clouds and vocabulary statistics.

Usage:
    python word_frequency.py                  # Basic frequency analysis
    python word_frequency.py --top 500        # Top 500 words
    python word_frequency.py --min-length 4   # Words with 4+ characters
    python word_frequency.py --by-folder      # Analyze by subfolder
"""

from pathlib import Path
from collections import Counter
import argparse
import json
import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./word_frequency_output")

# Extended stopwords (English + legal terms)
STOPWORDS = {
    # Common English
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'did', 'does',
    'doing', 'should', 'could', 'would', 'might', 'must', 'shall', 'may', 'am',
    'very', 'just', 'more', 'such', 'those', 'own', 'same', 'much', 'both', 'each',

    # Legal/document terms to filter
    'page', 'document', 'exhibit', 'file', 'case', 'court', 'pursuant',
    'herein', 'thereof', 'hereby', 'wherein', 'therefore', 'wherefore',
    'said', 'such', 'shall', 'upon', 'under', 'above', 'below',
    'plaintiff', 'defendant', 'witness', 'testimony', 'deposition',
    'attorney', 'counsel', 'motion', 'order', 'judgment', 'ruling',
    'matter', 'action', 'proceeding', 'hearing', 'trial',
    'united', 'states', 'state', 'county', 'district',
    'mr', 'ms', 'mrs', 'dr', 'jr', 'sr',
    'yes', 'no', 'okay', 'right', 'correct',
}


def load_documents(base_dir):
    """Load all text files from the base directory."""
    print("Loading documents...")
    documents = []

    txt_files = sorted(base_dir.rglob("*.txt"))

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore')
            if len(text) >= 100:
                # Get relative folder for grouping
                rel_path = txt_file.relative_to(base_dir)
                folder = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'

                documents.append({
                    'path': str(txt_file),
                    'filename': txt_file.name,
                    'folder': folder,
                    'text': text
                })
        except Exception:
            continue

    print(f"Loaded {len(documents)} documents\n")
    return documents


def tokenize(text, min_length=3):
    """Tokenize text into words."""
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    # Split into words
    words = text.split()

    # Filter
    filtered = []
    for word in words:
        # Skip short words
        if len(word) < min_length:
            continue
        # Skip numbers
        if word.isdigit():
            continue
        # Skip words that are mostly numbers
        if sum(c.isdigit() for c in word) > len(word) / 2:
            continue
        # Skip stopwords
        if word in STOPWORDS:
            continue

        filtered.append(word)

    return filtered


def calculate_frequencies(documents, min_length=3):
    """Calculate word frequencies across all documents."""
    print("Calculating word frequencies...")

    # Overall frequency
    total_freq = Counter()

    # Frequency by folder
    folder_freq = {}

    # Document frequency (how many docs contain each word)
    doc_freq = Counter()

    for doc in documents:
        words = tokenize(doc['text'], min_length)
        word_set = set(words)

        # Update total frequency
        total_freq.update(words)

        # Update document frequency
        doc_freq.update(word_set)

        # Update folder frequency
        folder = doc['folder']
        if folder not in folder_freq:
            folder_freq[folder] = Counter()
        folder_freq[folder].update(words)

    return total_freq, folder_freq, doc_freq


def calculate_tfidf_words(total_freq, doc_freq, num_docs, top_n=100):
    """Find words with high TF-IDF scores (distinctive words)."""
    tfidf_scores = {}

    for word, tf in total_freq.items():
        df = doc_freq.get(word, 1)
        idf = np.log(num_docs / df)
        tfidf_scores[word] = tf * idf

    # Sort by TF-IDF score
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:top_n]


def categorize_word(word):
    """Categorize a word by likely topic."""
    categories = {
        'Legal': ['law', 'legal', 'attorney', 'lawyer', 'judge', 'court', 'trial',
                  'lawsuit', 'prosecution', 'defense', 'guilty', 'innocent', 'verdict'],
        'Financial': ['money', 'bank', 'account', 'payment', 'dollar', 'million',
                     'billion', 'fund', 'financial', 'investment', 'wire', 'transfer'],
        'Travel': ['flight', 'plane', 'airport', 'travel', 'flew', 'passenger',
                   'pilot', 'jet', 'helicopter', 'trip', 'island'],
        'People': ['girl', 'woman', 'man', 'boy', 'child', 'children', 'minor',
                   'victim', 'friend', 'employee', 'staff', 'guest'],
        'Places': ['house', 'home', 'apartment', 'mansion', 'estate', 'ranch',
                   'island', 'beach', 'hotel', 'residence', 'property'],
        'Time': ['year', 'month', 'week', 'day', 'time', 'date', 'morning',
                 'evening', 'night', 'afternoon', 'hour', 'minute'],
        'Communication': ['phone', 'call', 'email', 'message', 'letter', 'contact',
                         'conversation', 'spoke', 'told', 'asked', 'answered'],
    }

    for category, keywords in categories.items():
        if word in keywords or any(kw in word for kw in keywords):
            return category

    return 'Other'


def generate_word_cloud_data(freq_counter, top_n=200):
    """Generate data suitable for word cloud visualization."""
    return freq_counter.most_common(top_n)


def main():
    parser = argparse.ArgumentParser(description="Word frequency analysis")
    parser.add_argument("--top", type=int, default=200,
                        help="Number of top words to analyze (default: 200)")
    parser.add_argument("--min-length", type=int, default=3,
                        help="Minimum word length (default: 3)")
    parser.add_argument("--by-folder", action="store_true",
                        help="Analyze frequencies by subfolder")
    args = parser.parse_args()

    print("=== WORD FREQUENCY ANALYSIS ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load documents
    documents = load_documents(BASE_DIR)

    if not documents:
        print(f"No documents found in {BASE_DIR}")
        return

    # Calculate frequencies
    total_freq, folder_freq, doc_freq = calculate_frequencies(documents, args.min_length)

    total_words = sum(total_freq.values())
    unique_words = len(total_freq)

    print(f"Total words (after filtering): {total_words:,}")
    print(f"Unique words: {unique_words:,}")

    # Display results
    print("\n" + "="*60)
    print("WORD FREQUENCY ANALYSIS")
    print("="*60)

    # Top words
    print(f"\n--- Top {args.top} Most Frequent Words ---")
    top_words = total_freq.most_common(args.top)
    for i, (word, count) in enumerate(top_words[:50], 1):
        pct = count / total_words * 100
        category = categorize_word(word)
        print(f"{i:3d}. {word:<20} {count:>8,} ({pct:.2f}%) [{category}]")

    # Category breakdown
    print("\n--- Word Categories (Top 200) ---")
    category_counts = Counter()
    for word, count in top_words:
        category_counts[categorize_word(word)] += count

    for category, count in category_counts.most_common():
        pct = count / sum(category_counts.values()) * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")

    # Distinctive words (high TF-IDF)
    print("\n--- Most Distinctive Words (TF-IDF) ---")
    distinctive = calculate_tfidf_words(total_freq, doc_freq, len(documents), args.top)
    for i, (word, score) in enumerate(distinctive[:30], 1):
        freq = total_freq[word]
        docs = doc_freq[word]
        print(f"{i:3d}. {word:<20} freq: {freq:>6,}  docs: {docs:>5}  tfidf: {score:.1f}")

    # By folder analysis
    if args.by_folder and len(folder_freq) > 1:
        print("\n--- Top Words by Folder ---")
        for folder, freq in sorted(folder_freq.items()):
            print(f"\n{folder}:")
            for word, count in freq.most_common(10):
                print(f"  {word}: {count:,}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save top words
    words_data = []
    for word, count in total_freq.most_common(args.top * 2):
        words_data.append({
            'word': word,
            'frequency': count,
            'percentage': count / total_words * 100,
            'doc_frequency': doc_freq[word],
            'doc_percentage': doc_freq[word] / len(documents) * 100,
            'category': categorize_word(word)
        })

    words_df = pd.DataFrame(words_data)
    words_df.to_csv(OUTPUT_DIR / "word_frequencies.csv", index=False)
    print(f"  Word frequencies: {OUTPUT_DIR / 'word_frequencies.csv'}")

    # Save distinctive words
    distinctive_data = []
    for word, score in distinctive:
        distinctive_data.append({
            'word': word,
            'tfidf_score': score,
            'frequency': total_freq[word],
            'doc_frequency': doc_freq[word]
        })

    distinctive_df = pd.DataFrame(distinctive_data)
    distinctive_df.to_csv(OUTPUT_DIR / "distinctive_words.csv", index=False)
    print(f"  Distinctive words: {OUTPUT_DIR / 'distinctive_words.csv'}")

    # Save category summary
    category_df = pd.DataFrame([
        {'category': cat, 'word_count': count}
        for cat, count in category_counts.most_common()
    ])
    category_df.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)
    print(f"  Category summary: {OUTPUT_DIR / 'category_summary.csv'}")

    # Save word cloud data
    cloud_data = generate_word_cloud_data(total_freq, args.top)
    with open(OUTPUT_DIR / "wordcloud_data.json", 'w', encoding='utf-8') as f:
        json.dump(cloud_data, f, indent=2)
    print(f"  Word cloud data: {OUTPUT_DIR / 'wordcloud_data.json'}")

    # Save by-folder data if requested
    if args.by_folder:
        folder_data = {}
        for folder, freq in folder_freq.items():
            folder_data[folder] = freq.most_common(100)
        with open(OUTPUT_DIR / "folder_frequencies.json", 'w', encoding='utf-8') as f:
            json.dump(folder_data, f, indent=2)
        print(f"  Folder frequencies: {OUTPUT_DIR / 'folder_frequencies.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Top words bar chart
    fig, ax = plt.subplots(figsize=(12, 10))

    top_30 = total_freq.most_common(30)
    words = [w for w, _ in top_30]
    counts = [c for _, c in top_30]

    # Color by category
    colors = []
    color_map = {
        'Legal': '#e74c3c',
        'Financial': '#3498db',
        'Travel': '#2ecc71',
        'People': '#9b59b6',
        'Places': '#f39c12',
        'Time': '#1abc9c',
        'Communication': '#e91e63',
        'Other': '#95a5a6'
    }
    for word in words:
        cat = categorize_word(word)
        colors.append(color_map.get(cat, '#95a5a6'))

    bars = ax.barh(range(len(words)), counts, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top 30 Most Frequent Words', fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat) for cat, c in color_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_words_chart.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Top words chart: {OUTPUT_DIR / 'top_words_chart.pdf'}")

    # Category pie chart
    fig, ax = plt.subplots(figsize=(10, 10))

    categories = [cat for cat, _ in category_counts.most_common()]
    cat_values = [count for _, count in category_counts.most_common()]
    cat_colors = [color_map.get(cat, '#95a5a6') for cat in categories]

    ax.pie(cat_values, labels=categories, colors=cat_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('Word Categories Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_pie_chart.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Category pie chart: {OUTPUT_DIR / 'category_pie_chart.pdf'}")

    # Word frequency distribution (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))

    freqs = sorted(total_freq.values(), reverse=True)
    ax.loglog(range(1, len(freqs) + 1), freqs, 'b-', alpha=0.7)

    ax.set_xlabel('Word Rank (log scale)', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title("Zipf's Law: Word Frequency Distribution", fontsize=14)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zipf_distribution.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Zipf distribution: {OUTPUT_DIR / 'zipf_distribution.pdf'}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents analyzed: {len(documents)}")
    print(f"Total words: {total_words:,}")
    print(f"Unique words: {unique_words:,}")
    print(f"Vocabulary density: {unique_words/total_words*100:.2f}%")

    if top_words:
        print(f"\nMost frequent: '{top_words[0][0]}' ({top_words[0][1]:,} occurrences)")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nWord frequency analysis complete!")


if __name__ == "__main__":
    main()
