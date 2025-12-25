#!/usr/bin/env python3
"""
Topic Modeling Script - Discover themes in documents using BERTopic.

Uses transformer-based embeddings for semantic topic discovery across
the document corpus.

Setup:
    pip install -r requirements.txt

Usage:
    python topic_modeling.py              # Run with defaults
    python topic_modeling.py --nr-topics 20   # Specify number of topics
    python topic_modeling.py --min-topic-size 5  # Min documents per topic
"""

from pathlib import Path
import argparse
import json

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./topic_modeling_output")


def load_documents(base_dir):
    """Load all text files from the base directory."""
    print("Loading documents...")
    documents = []
    file_paths = []

    txt_files = sorted(base_dir.rglob("*.txt"))

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            # Skip empty or very short documents
            if len(text) >= 100:
                documents.append(text)
                file_paths.append(str(txt_file))
        except Exception as e:
            print(f"  Error reading {txt_file}: {e}")
            continue

    print(f"Loaded {len(documents)} documents\n")
    return documents, file_paths


def truncate_documents(documents, max_length=10000):
    """Truncate documents to max length for embedding efficiency."""
    return [doc[:max_length] for doc in documents]


def main():
    parser = argparse.ArgumentParser(description="Topic modeling with BERTopic")
    parser.add_argument("--nr-topics", type=int, default=None,
                        help="Number of topics (default: auto)")
    parser.add_argument("--min-topic-size", type=int, default=3,
                        help="Minimum documents per topic (default: 3)")
    parser.add_argument("--embedding-model", type=str,
                        default="all-MiniLM-L6-v2",
                        help="Sentence transformer model (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    print("=== TOPIC MODELING (BERTopic) ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load documents
    documents, file_paths = load_documents(BASE_DIR)

    if len(documents) < 10:
        print("Error: Need at least 10 documents for topic modeling")
        return

    # Truncate for efficiency
    print("Preparing documents...")
    docs_truncated = truncate_documents(documents)

    # Initialize embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)

    # Initialize and fit BERTopic
    print("\nFitting BERTopic model...")
    print(f"  Min topic size: {args.min_topic_size}")
    print(f"  Nr topics: {args.nr_topics or 'auto'}\n")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs_truncated)

    # Get topic info
    topic_info = topic_model.get_topic_info()

    print("\n" + "="*60)
    print("DISCOVERED TOPICS")
    print("="*60 + "\n")

    # Display topics (skip -1 which is outliers)
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            print(f"Topic {topic_id} (Outliers): {row['Count']} documents\n")
            continue

        count = row['Count']
        # Get top words for this topic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = ", ".join([word for word, _ in topic_words[:8]])
            print(f"Topic {topic_id} ({count} docs): {words}\n")

    # Save results
    print("\nSaving results...")

    # Save topic info
    topic_info.to_csv(OUTPUT_DIR / "topics.csv", index=False)
    print(f"  Topics saved to: {OUTPUT_DIR / 'topics.csv'}")

    # Save document-topic assignments
    doc_topics = []
    for i, (file_path, topic, prob) in enumerate(zip(file_paths, topics, probs)):
        doc_topics.append({
            "file": file_path,
            "topic": int(topic),
            "probability": float(prob) if prob is not None else None
        })

    with open(OUTPUT_DIR / "document_topics.json", 'w', encoding='utf-8') as f:
        json.dump(doc_topics, f, indent=2)
    print(f"  Document-topic mapping saved to: {OUTPUT_DIR / 'document_topics.json'}")

    # Save detailed topic words
    all_topics = {}
    for topic_id in topic_info['Topic'].unique():
        if topic_id != -1:
            words = topic_model.get_topic(topic_id)
            all_topics[str(topic_id)] = [{"word": w, "score": float(s)} for w, s in words]

    with open(OUTPUT_DIR / "topic_words.json", 'w', encoding='utf-8') as f:
        json.dump(all_topics, f, indent=2)
    print(f"  Topic words saved to: {OUTPUT_DIR / 'topic_words.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    try:
        # Topic hierarchy
        fig = topic_model.visualize_hierarchy()
        fig.write_html(OUTPUT_DIR / "topic_hierarchy.html")
        print(f"  Topic hierarchy: {OUTPUT_DIR / 'topic_hierarchy.html'}")
    except Exception as e:
        print(f"  Could not generate hierarchy: {e}")

    try:
        # Topic bar chart
        fig = topic_model.visualize_barchart(top_n_topics=min(15, len(topic_info)-1))
        fig.write_html(OUTPUT_DIR / "topic_barchart.html")
        print(f"  Topic barchart: {OUTPUT_DIR / 'topic_barchart.html'}")
    except Exception as e:
        print(f"  Could not generate barchart: {e}")

    try:
        # Intertopic distance map
        fig = topic_model.visualize_topics()
        fig.write_html(OUTPUT_DIR / "topic_distance_map.html")
        print(f"  Topic distance map: {OUTPUT_DIR / 'topic_distance_map.html'}")
    except Exception as e:
        print(f"  Could not generate distance map: {e}")

    # Save the model for later use
    topic_model.save(OUTPUT_DIR / "bertopic_model", serialization="safetensors")
    print(f"  Model saved to: {OUTPUT_DIR / 'bertopic_model'}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total documents analyzed: {len(documents)}")
    print(f"Topics discovered: {len(topic_info) - 1}")  # -1 for outlier topic
    print(f"Documents in outlier topic: {topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nTopic modeling complete!")


if __name__ == "__main__":
    main()
