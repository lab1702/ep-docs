#!/usr/bin/env python3
"""
Organization Extraction Script - Extract organization entities from text files using spaCy NER.

Extracts ORG entities (companies, institutions, agencies, etc.) from all documents
and generates organization frequency analysis with co-occurrence network.

Setup:
    python -m spacy download en_core_web_sm

Usage:
    python extract_organizations.py
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from itertools import combinations
import multiprocessing
import json

import spacy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./organization_extraction_output")
MIN_ORG_MENTIONS = 3  # Minimum mentions to include in network
MIN_COOCCURRENCE = 2  # Minimum co-occurrences for network edge

# Global variable for worker processes
_worker_nlp = None


def _init_worker():
    """Initialize spaCy model for each worker process."""
    global _worker_nlp
    _worker_nlp = spacy.load('en_core_web_sm')
    _worker_nlp.max_length = 2_000_000


def extract_organizations(text, nlp):
    """Extract organization entities from text using spaCy NER."""
    max_chunk = 100000
    organizations = set()

    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    # Clean up the organization name
                    name = ' '.join(ent.text.split()).strip()
                    if len(name) >= 2:
                        organizations.add(name)
        except Exception:
            continue

    return organizations


def normalize_org_name(name):
    """Normalize organization name for better matching."""
    # Remove common suffixes for grouping
    name = name.strip()

    # Remove possessive
    if name.endswith("'s"):
        name = name[:-2]

    return name


def sanitize_text(text):
    """Remove control characters and non-printable characters."""
    # Keep only printable ASCII and common extended chars
    return ''.join(c for c in text if c.isprintable() or c in ' \t')


def filter_organizations(organizations):
    """Filter out likely false positives."""
    filtered = set()

    # Common false positives to exclude
    exclude_terms = {
        'i', 'we', 'you', 'he', 'she', 'it', 'they',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'exhibit', 'page', 'document', 'section', 'chapter',
        'plaintiff', 'defendant', 'court', 'jury',
        'yes', 'no', 'ok', 'okay'
    }

    for org in organizations:
        # Sanitize first - remove control characters
        org = sanitize_text(org).strip()

        # Skip empty after sanitization
        if not org:
            continue

        org_lower = org.lower()

        # Skip if it's a common false positive
        if org_lower in exclude_terms:
            continue

        # Skip single characters
        if len(org) <= 1:
            continue

        # Skip if all digits
        if org.replace(' ', '').replace('-', '').isdigit():
            continue

        # Skip very long strings (likely parsing errors)
        if len(org) > 150:
            continue

        # Skip if contains non-ASCII characters (likely OCR garbage)
        try:
            org.encode('ascii')
        except UnicodeEncodeError:
            continue

        # Skip if more than 30% of characters are uppercase letters in a row (likely garbage)
        if len(org) > 10:
            upper_runs = sum(1 for i, c in enumerate(org) if c.isupper() and (i == 0 or org[i-1].isupper()))
            if upper_runs / len(org) > 0.5:
                continue

        # Skip if looks like a person name (simple heuristic)
        words = org.split()
        if len(words) == 2 and all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
            # Could be "John Smith" - but keep if has org keywords
            org_keywords = ['inc', 'corp', 'llc', 'ltd', 'company', 'group', 'bank',
                          'university', 'college', 'foundation', 'institute', 'association']
            if not any(kw in org_lower for kw in org_keywords):
                continue

        filtered.add(normalize_org_name(org))

    return filtered


def categorize_organization(org_name):
    """Categorize organization by type based on keywords."""
    org_lower = org_name.lower()

    categories = {
        'Financial': ['bank', 'financial', 'capital', 'investment', 'securities',
                     'fund', 'trust', 'credit', 'insurance', 'mortgage'],
        'Legal': ['law', 'legal', 'court', 'attorney', 'lawyer', 'firm'],
        'Government': ['department', 'agency', 'bureau', 'federal', 'state',
                      'county', 'city', 'office of', 'administration', 'fbi', 'cia', 'doj'],
        'Education': ['university', 'college', 'school', 'academy', 'institute'],
        'Media': ['news', 'media', 'times', 'post', 'journal', 'magazine',
                 'broadcasting', 'television', 'tv', 'radio'],
        'Aviation': ['airlines', 'airways', 'aviation', 'airport', 'air '],
        'Healthcare': ['hospital', 'medical', 'health', 'clinic', 'pharmaceutical'],
        'Foundation': ['foundation', 'charity', 'nonprofit', 'non-profit']
    }

    for category, keywords in categories.items():
        if any(kw in org_lower for kw in keywords):
            return category

    return 'Other'


def process_file(txt_file):
    """Process a single file - designed for parallel execution."""
    global _worker_nlp
    try:
        text = txt_file.read_text(encoding='utf-8', errors='ignore')
        organizations = extract_organizations(text, _worker_nlp)
        organizations = filter_organizations(organizations)

        if organizations:
            return (str(txt_file), list(organizations))
    except Exception:
        pass
    return None


def build_org_network(file_orgs, org_counts, min_mentions, min_cooccurrence):
    """Build co-occurrence network of organizations."""
    # Filter to orgs with minimum mentions
    frequent_orgs = {org for org, count in org_counts.items() if count >= min_mentions}

    # Count co-occurrences
    edge_weights = defaultdict(int)
    for file_path, orgs in file_orgs.items():
        orgs_in_file = set(orgs) & frequent_orgs
        if len(orgs_in_file) >= 2:
            for o1, o2 in combinations(sorted(orgs_in_file), 2):
                edge_weights[(o1, o2)] += 1

    # Build graph
    G = nx.Graph()

    # Add nodes
    for org in frequent_orgs:
        G.add_node(org, mentions=org_counts[org], category=categorize_organization(org))

    # Add edges
    for (o1, o2), weight in edge_weights.items():
        if weight >= min_cooccurrence:
            G.add_edge(o1, o2, weight=weight)

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G


def main():
    print("=== ORGANIZATION EXTRACTION (spaCy NER) ===\n")

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
    print("Extracting organizations from files...")
    file_orgs = {}
    total_files = len(txt_files)

    num_workers = min(multiprocessing.cpu_count(), total_files) if total_files > 0 else 1
    print(f"Using {num_workers} parallel workers\n")

    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(process_file, f): f for f in txt_files}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                file_path, orgs = result
                file_orgs[file_path] = orgs

            completed += 1
            if completed % 100 == 0 or completed == total_files:
                print(f"Progress: {completed}/{total_files} files ({100*completed/total_files:.1f}%)")

    print(f"\nFiles with organizations found: {len(file_orgs)}")

    # Aggregate organization counts
    org_counts = Counter()
    for orgs in file_orgs.values():
        org_counts.update(orgs)

    print(f"Unique organizations found: {len(org_counts)}")

    # Categorize organizations
    category_counts = Counter()
    for org in org_counts:
        category_counts[categorize_organization(org)] += org_counts[org]

    # Display top organizations
    print("\n" + "="*60)
    print("TOP ORGANIZATIONS")
    print("="*60)

    print("\n--- Top 40 Organizations by Mentions ---")
    for i, (org, count) in enumerate(org_counts.most_common(40), 1):
        category = categorize_organization(org)
        print(f"{i:2d}. {org}: {count} mentions [{category}]")

    # Display by category
    print("\n--- Mentions by Category ---")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count:,} mentions")

    # Build co-occurrence network
    print("\n" + "="*60)
    print("ORGANIZATION NETWORK")
    print("="*60 + "\n")

    print(f"Building network (min mentions: {MIN_ORG_MENTIONS}, min co-occurrence: {MIN_COOCCURRENCE})...")
    G = build_org_network(file_orgs, org_counts, MIN_ORG_MENTIONS, MIN_COOCCURRENCE)

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        # Find most connected organizations
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        print("\n--- Top 20 Most Connected Organizations ---")
        for i, (org, degree) in enumerate(degree_sorted[:20], 1):
            print(f"{i:2d}. {org} ({degree} connections)")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save raw extraction data
    with open(OUTPUT_DIR / "extracted_organizations.json", 'w', encoding='utf-8') as f:
        json.dump(file_orgs, f, indent=2)
    print(f"  Raw extractions: {OUTPUT_DIR / 'extracted_organizations.json'}")

    # Save frequency table with categories
    org_df = pd.DataFrame([
        {
            'organization': org,
            'mentions': count,
            'category': categorize_organization(org)
        }
        for org, count in org_counts.most_common()
    ])
    org_df.to_csv(OUTPUT_DIR / "organization_frequencies.csv", index=False)
    print(f"  Frequencies: {OUTPUT_DIR / 'organization_frequencies.csv'}")

    # Save category summary
    category_df = pd.DataFrame([
        {'category': cat, 'total_mentions': count}
        for cat, count in category_counts.most_common()
    ])
    category_df.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)
    print(f"  Category summary: {OUTPUT_DIR / 'category_summary.csv'}")

    # Save network files
    if G.number_of_nodes() > 0:
        # Nodes CSV
        nodes_data = []
        for node in G.nodes():
            nodes_data.append({
                'organization': node,
                'mentions': org_counts[node],
                'category': G.nodes[node]['category'],
                'degree': G.degree(node)
            })
        nodes_df = pd.DataFrame(nodes_data).sort_values('mentions', ascending=False)
        nodes_df.to_csv(OUTPUT_DIR / "network_nodes.csv", index=False)
        print(f"  Network nodes: {OUTPUT_DIR / 'network_nodes.csv'}")

        # Edges CSV
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'from': u,
                'to': v,
                'weight': data['weight']
            })
        edges_df = pd.DataFrame(edges_data).sort_values('weight', ascending=False)
        edges_df.to_csv(OUTPUT_DIR / "network_edges.csv", index=False)
        print(f"  Network edges: {OUTPUT_DIR / 'network_edges.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Top organizations bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    top_orgs = org_counts.most_common(30)
    orgs = [sanitize_text(org) for org, _ in top_orgs]  # Sanitize for display
    counts = [count for _, count in top_orgs]

    # Color by category
    colors = []
    color_map = {
        'Financial': '#3498db',
        'Legal': '#9b59b6',
        'Government': '#e74c3c',
        'Education': '#2ecc71',
        'Media': '#f39c12',
        'Aviation': '#1abc9c',
        'Healthcare': '#e91e63',
        'Foundation': '#00bcd4',
        'Other': '#95a5a6'
    }
    for org, _ in top_orgs:  # Use original name for categorization
        cat = categorize_organization(org)
        colors.append(color_map.get(cat, '#95a5a6'))

    bars = ax.barh(range(len(orgs)), counts, color=colors)
    ax.set_yticks(range(len(orgs)))
    ax.set_yticklabels(orgs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Mentions')
    ax.set_title('Top 30 Organizations by Mentions')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_organizations_chart.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Organizations chart: {OUTPUT_DIR / 'top_organizations_chart.pdf'}")

    # Category pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    categories = [cat for cat, _ in category_counts.most_common()]
    cat_values = [count for _, count in category_counts.most_common()]
    cat_colors = [color_map.get(cat, '#95a5a6') for cat in categories]

    ax.pie(cat_values, labels=categories, colors=cat_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 10})
    ax.set_title('Organization Mentions by Category')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_pie_chart.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Category pie chart: {OUTPUT_DIR / 'category_pie_chart.pdf'}")

    # Network visualization
    if G.number_of_nodes() >= 3:
        fig, ax = plt.subplots(figsize=(20, 20))

        # Layout
        pos = nx.spring_layout(G, k=2/G.number_of_nodes()**0.5, iterations=50, seed=42)

        # Node sizes and colors
        node_sizes = [100 + 20 * org_counts[n]**0.5 for n in G.nodes()]
        node_colors = [color_map.get(G.nodes[n]['category'], '#95a5a6') for n in G.nodes()]

        # Edge widths
        edge_widths = [0.5 + G[u][v]['weight'] * 0.3 for u, v in G.edges()]

        # Sanitize labels for display (remove any remaining problematic chars)
        safe_labels = {n: sanitize_text(n) for n in G.nodes()}

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=safe_labels, font_size=6, ax=ax)

        ax.set_title(f"Organization Co-occurrence Network\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "organization_network.pdf", format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Network visualization: {OUTPUT_DIR / 'organization_network.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed: {len(txt_files)}")
    print(f"Files with organizations: {len(file_orgs)}")
    print(f"Unique organizations: {len(org_counts)}")
    print(f"Total mentions: {sum(org_counts.values()):,}")
    if G.number_of_nodes() > 0:
        print(f"Network nodes: {G.number_of_nodes()}")
        print(f"Network edges: {G.number_of_edges()}")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nOrganization extraction complete!")


if __name__ == "__main__":
    main()
