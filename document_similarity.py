#!/usr/bin/env python3
"""
Document Similarity Script - Find duplicate and near-duplicate documents.

Uses TF-IDF vectorization and cosine similarity to identify similar documents
in the corpus. Optionally uses sentence embeddings for semantic similarity.

Usage:
    python document_similarity.py                    # TF-IDF (fast)
    python document_similarity.py --method embedding # Semantic similarity
    python document_similarity.py --threshold 0.9    # Higher threshold = stricter
"""

from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./document_similarity_output")
DEFAULT_THRESHOLD = 0.8  # Minimum similarity to consider as "similar"
MAX_FEATURES = 10000  # Max features for TF-IDF


def load_documents(base_dir, max_length=50000):
    """Load all text files from the base directory."""
    print("Loading documents...")
    documents = []
    file_paths = []

    txt_files = sorted(base_dir.rglob("*.txt"))

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            # Skip empty documents
            if len(text) >= 50:
                # Truncate very long documents for efficiency
                documents.append(text[:max_length])
                file_paths.append(str(txt_file))
        except Exception:
            continue

    print(f"Loaded {len(documents)} documents\n")
    return documents, file_paths


def sanitize_text(text):
    """Remove control characters."""
    return ''.join(c for c in text if c.isprintable() or c in ' \t\n')


def compute_tfidf_similarity(documents, threshold, batch_size=1000):
    """Compute document similarity using TF-IDF."""
    print("Computing TF-IDF vectors...")

    # Clean documents
    clean_docs = [sanitize_text(doc) for doc in documents]

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(clean_docs)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Find similar pairs using batched computation
    print(f"\nFinding similar document pairs (threshold >= {threshold})...")
    n_docs = tfidf_matrix.shape[0]
    similar_pairs = []

    # Process in batches to manage memory
    for i in range(0, n_docs, batch_size):
        end_i = min(i + batch_size, n_docs)
        batch = tfidf_matrix[i:end_i]

        # Compare batch against all documents after it
        for j in range(i, n_docs, batch_size):
            end_j = min(j + batch_size, n_docs)

            if i == j:
                # Same batch - compute upper triangle only
                sim_matrix = cosine_similarity(batch)
                for bi in range(sim_matrix.shape[0]):
                    for bj in range(bi + 1, sim_matrix.shape[1]):
                        if sim_matrix[bi, bj] >= threshold:
                            similar_pairs.append((i + bi, i + bj, sim_matrix[bi, bj]))
            elif j > i:
                # Different batches
                other_batch = tfidf_matrix[j:end_j]
                sim_matrix = cosine_similarity(batch, other_batch)
                for bi in range(sim_matrix.shape[0]):
                    for bj in range(sim_matrix.shape[1]):
                        if sim_matrix[bi, bj] >= threshold:
                            similar_pairs.append((i + bi, j + bj, sim_matrix[bi, bj]))

        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {end_i}/{n_docs} documents processed...")

    return similar_pairs, tfidf_matrix


def compute_embedding_similarity(documents, threshold, batch_size=100):
    """Compute document similarity using sentence embeddings."""
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Computing document embeddings...")
    # Clean and truncate for embedding
    clean_docs = [sanitize_text(doc)[:5000] for doc in documents]

    # Compute embeddings in batches
    embeddings = model.encode(clean_docs, show_progress_bar=True, batch_size=32)

    print(f"  Embeddings shape: {embeddings.shape}")

    # Find similar pairs
    print(f"\nFinding similar document pairs (threshold >= {threshold})...")
    n_docs = len(embeddings)
    similar_pairs = []

    for i in range(0, n_docs, batch_size):
        end_i = min(i + batch_size, n_docs)
        batch = embeddings[i:end_i]

        for j in range(i, n_docs, batch_size):
            end_j = min(j + batch_size, n_docs)

            if i == j:
                sim_matrix = cosine_similarity(batch)
                for bi in range(sim_matrix.shape[0]):
                    for bj in range(bi + 1, sim_matrix.shape[1]):
                        if sim_matrix[bi, bj] >= threshold:
                            similar_pairs.append((i + bi, i + bj, sim_matrix[bi, bj]))
            elif j > i:
                other_batch = embeddings[j:end_j]
                sim_matrix = cosine_similarity(batch, other_batch)
                for bi in range(sim_matrix.shape[0]):
                    for bj in range(sim_matrix.shape[1]):
                        if sim_matrix[bi, bj] >= threshold:
                            similar_pairs.append((i + bi, j + bj, sim_matrix[bi, bj]))

        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {end_i}/{n_docs} documents processed...")

    return similar_pairs, embeddings


def cluster_similar_documents(similar_pairs, n_docs):
    """Group similar documents into clusters using Union-Find."""
    # Union-Find data structure
    parent = list(range(n_docs))
    rank = [0] * n_docs

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # Union similar documents
    for i, j, _ in similar_pairs:
        union(i, j)

    # Group by cluster
    clusters = defaultdict(list)
    for i in range(n_docs):
        clusters[find(i)].append(i)

    # Filter to clusters with more than one document
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}

    return clusters


def main():
    parser = argparse.ArgumentParser(description="Find similar documents")
    parser.add_argument("--method", choices=['tfidf', 'embedding'], default='tfidf',
                        help="Similarity method (default: tfidf)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    print("=== DOCUMENT SIMILARITY ANALYSIS ===\n")
    print(f"Method: {args.method.upper()}")
    print(f"Threshold: {args.threshold}\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load documents
    documents, file_paths = load_documents(BASE_DIR)

    if len(documents) < 2:
        print("Error: Need at least 2 documents for similarity analysis")
        return

    # Compute similarity
    if args.method == 'tfidf':
        similar_pairs, _ = compute_tfidf_similarity(documents, args.threshold)
    else:
        similar_pairs, _ = compute_embedding_similarity(documents, args.threshold)

    print(f"\nFound {len(similar_pairs)} similar document pairs")

    # Cluster similar documents
    clusters = cluster_similar_documents(similar_pairs, len(documents))
    print(f"Document clusters (duplicates/near-duplicates): {len(clusters)}")

    # Display results
    print("\n" + "="*60)
    print("SIMILARITY RESULTS")
    print("="*60)

    # Sort pairs by similarity
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n--- Top 20 Most Similar Document Pairs ---")
    for i, (idx1, idx2, sim) in enumerate(similar_pairs[:20], 1):
        file1 = Path(file_paths[idx1]).name
        file2 = Path(file_paths[idx2]).name
        print(f"{i:2d}. [{sim:.3f}] {file1}")
        print(f"           {file2}")

    # Cluster statistics
    print("\n--- Duplicate/Near-Duplicate Clusters ---")
    cluster_sizes = sorted([len(v) for v in clusters.values()], reverse=True)

    print(f"Total clusters: {len(clusters)}")
    print(f"Documents in clusters: {sum(cluster_sizes)}")
    print(f"Largest cluster: {cluster_sizes[0] if cluster_sizes else 0} documents")

    # Show top clusters
    print("\n--- Top 10 Largest Clusters ---")
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (_, members) in enumerate(sorted_clusters[:10], 1):
        print(f"\nCluster {i} ({len(members)} documents):")
        for idx in members[:5]:  # Show first 5 files
            print(f"  - {Path(file_paths[idx]).name}")
        if len(members) > 5:
            print(f"  ... and {len(members) - 5} more")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save similar pairs
    pairs_df = pd.DataFrame([
        {
            'file1': file_paths[idx1],
            'file2': file_paths[idx2],
            'similarity': sim
        }
        for idx1, idx2, sim in similar_pairs
    ]).sort_values('similarity', ascending=False)

    pairs_df.to_csv(OUTPUT_DIR / "similar_pairs.csv", index=False)
    print(f"  Similar pairs: {OUTPUT_DIR / 'similar_pairs.csv'}")

    # Save clusters
    cluster_data = {}
    for i, (_, members) in enumerate(sorted_clusters):
        cluster_data[f"cluster_{i}"] = {
            'size': len(members),
            'files': [file_paths[idx] for idx in members]
        }

    with open(OUTPUT_DIR / "document_clusters.json", 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, indent=2)
    print(f"  Document clusters: {OUTPUT_DIR / 'document_clusters.json'}")

    # Save cluster summary
    cluster_summary = pd.DataFrame([
        {
            'cluster_id': i,
            'size': len(members),
            'sample_file': Path(file_paths[members[0]]).name
        }
        for i, (_, members) in enumerate(sorted_clusters)
    ])
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    print(f"  Cluster summary: {OUTPUT_DIR / 'cluster_summary.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Similarity distribution histogram
    if similar_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))

        similarities = [sim for _, _, sim in similar_pairs]
        ax.hist(similarities, bins=50, color='steelblue', edgecolor='navy', alpha=0.8)

        ax.axvline(x=0.95, color='red', linestyle='--', label='Near-exact (0.95)')
        ax.axvline(x=0.90, color='orange', linestyle='--', label='Very similar (0.90)')

        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Number of Document Pairs', fontsize=12)
        ax.set_title(f'Document Similarity Distribution (threshold >= {args.threshold})', fontsize=14)
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "similarity_distribution.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Similarity distribution: {OUTPUT_DIR / 'similarity_distribution.pdf'}")

    # Cluster size distribution
    if cluster_sizes:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(range(min(30, len(cluster_sizes))), cluster_sizes[:30],
               color='steelblue', edgecolor='navy', alpha=0.8)

        ax.set_xlabel('Cluster Rank', fontsize=12)
        ax.set_ylabel('Cluster Size (# of documents)', fontsize=12)
        ax.set_title('Top 30 Duplicate Clusters by Size', fontsize=14)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cluster_sizes.pdf", format='pdf', dpi=150)
        plt.close()
        print(f"  Cluster sizes: {OUTPUT_DIR / 'cluster_sizes.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents analyzed: {len(documents)}")
    print(f"Similar pairs found: {len(similar_pairs)}")
    print(f"Duplicate clusters: {len(clusters)}")
    print(f"Documents in clusters: {sum(cluster_sizes)}")

    if similar_pairs:
        near_exact = sum(1 for _, _, s in similar_pairs if s >= 0.95)
        very_similar = sum(1 for _, _, s in similar_pairs if 0.90 <= s < 0.95)
        print(f"Near-exact duplicates (>=0.95): {near_exact} pairs")
        print(f"Very similar (0.90-0.95): {very_similar} pairs")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nDocument similarity analysis complete!")


if __name__ == "__main__":
    main()
