#!/usr/bin/env python3
"""
Cross-Entity Analysis Script - Link people, organizations, locations, and dates.

Analyzes co-occurrences across different entity types to discover relationships
like which people are connected to which organizations, locations, and time periods.

Prerequisites:
    Run these first to generate entity data:
    - python extract_names.py
    - python extract_locations.py
    - python extract_organizations.py
    - python extract_timeline.py

Usage:
    python cross_entity_analysis.py
    python cross_entity_analysis.py --min-cooccurrence 3  # Stricter threshold
"""

from pathlib import Path
from collections import defaultdict, Counter
from itertools import product
import argparse
import json

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
OUTPUT_DIR = Path("./cross_entity_output")

# Input files from other extraction scripts
NAMES_FILE = Path("./extracted_names.json")
LOCATIONS_FILE = Path("./location_extraction_output/extracted_locations.json")
ORGS_FILE = Path("./organization_extraction_output/extracted_organizations.json")
TIMELINE_FILE = Path("./timeline_extraction_output/extracted_dates.json")

# Minimum occurrences to include entity
MIN_ENTITY_FREQ = 3
MIN_COOCCURRENCE = 2


def load_entities():
    """Load all extracted entities from JSON files."""
    entities = {
        'persons': {},      # file -> [persons]
        'locations': {},    # file -> [locations]
        'organizations': {},# file -> [organizations]
        'years': {}         # file -> [years]
    }

    # Load persons
    if NAMES_FILE.exists():
        with open(NAMES_FILE, 'r', encoding='utf-8') as f:
            entities['persons'] = json.load(f)
        print(f"  Loaded persons from {len(entities['persons'])} files")
    else:
        print(f"  Warning: {NAMES_FILE} not found")

    # Load locations (combine GPE, LOC, FAC)
    if LOCATIONS_FILE.exists():
        with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            raw_locations = json.load(f)
        for file_path, loc_data in raw_locations.items():
            all_locs = []
            if isinstance(loc_data, dict):
                all_locs.extend(loc_data.get('GPE', []))
                all_locs.extend(loc_data.get('LOC', []))
                all_locs.extend(loc_data.get('FAC', []))
            entities['locations'][file_path] = all_locs
        print(f"  Loaded locations from {len(entities['locations'])} files")
    else:
        print(f"  Warning: {LOCATIONS_FILE} not found")

    # Load organizations
    if ORGS_FILE.exists():
        with open(ORGS_FILE, 'r', encoding='utf-8') as f:
            entities['organizations'] = json.load(f)
        print(f"  Loaded organizations from {len(entities['organizations'])} files")
    else:
        print(f"  Warning: {ORGS_FILE} not found")

    # Load timeline (extract years)
    if TIMELINE_FILE.exists():
        with open(TIMELINE_FILE, 'r', encoding='utf-8') as f:
            raw_dates = json.load(f)
        for file_path, date_list in raw_dates.items():
            years = list(set(str(d['year']) for d in date_list if 'year' in d))
            if years:
                entities['years'][file_path] = years
        print(f"  Loaded years from {len(entities['years'])} files")
    else:
        print(f"  Warning: {TIMELINE_FILE} not found")

    return entities


def get_entity_frequencies(entities):
    """Count frequency of each entity across all files."""
    frequencies = {
        'persons': Counter(),
        'locations': Counter(),
        'organizations': Counter(),
        'years': Counter()
    }

    for entity_type, file_entities in entities.items():
        for file_path, entity_list in file_entities.items():
            frequencies[entity_type].update(entity_list)

    return frequencies


def compute_cooccurrences(entities, min_freq, min_cooccur):
    """Compute co-occurrences between different entity types."""
    # Get entity frequencies for filtering
    frequencies = get_entity_frequencies(entities)

    # Filter to frequent entities
    frequent = {}
    for entity_type, counts in frequencies.items():
        frequent[entity_type] = {e for e, c in counts.items() if c >= min_freq}

    print(f"\n  Frequent persons: {len(frequent['persons'])}")
    print(f"  Frequent locations: {len(frequent['locations'])}")
    print(f"  Frequent organizations: {len(frequent['organizations'])}")
    print(f"  Frequent years: {len(frequent['years'])}")

    # Get all files
    all_files = set()
    for file_entities in entities.values():
        all_files.update(file_entities.keys())

    # Compute co-occurrences for each entity type pair
    cooccurrences = {
        'person_org': defaultdict(int),
        'person_location': defaultdict(int),
        'person_year': defaultdict(int),
        'org_location': defaultdict(int),
        'org_year': defaultdict(int),
        'location_year': defaultdict(int)
    }

    for file_path in all_files:
        persons = set(entities['persons'].get(file_path, [])) & frequent['persons']
        locations = set(entities['locations'].get(file_path, [])) & frequent['locations']
        orgs = set(entities['organizations'].get(file_path, [])) & frequent['organizations']
        years = set(entities['years'].get(file_path, [])) & frequent['years']

        # Person-Org
        for p, o in product(persons, orgs):
            cooccurrences['person_org'][(p, o)] += 1

        # Person-Location
        for p, l in product(persons, locations):
            cooccurrences['person_location'][(p, l)] += 1

        # Person-Year
        for p, y in product(persons, years):
            cooccurrences['person_year'][(p, y)] += 1

        # Org-Location
        for o, l in product(orgs, locations):
            cooccurrences['org_location'][(o, l)] += 1

        # Org-Year
        for o, y in product(orgs, years):
            cooccurrences['org_year'][(o, y)] += 1

        # Location-Year
        for l, y in product(locations, years):
            cooccurrences['location_year'][(l, y)] += 1

    # Filter by minimum co-occurrence
    filtered = {}
    for rel_type, pairs in cooccurrences.items():
        filtered[rel_type] = {k: v for k, v in pairs.items() if v >= min_cooccur}

    return filtered, frequencies


def build_multi_entity_graph(cooccurrences, frequencies, top_n=200):
    """Build a multi-type entity graph."""
    G = nx.Graph()

    # Get top entities by frequency
    top_persons = [e for e, _ in frequencies['persons'].most_common(top_n)]
    top_orgs = [e for e, _ in frequencies['organizations'].most_common(top_n)]
    top_locations = [e for e, _ in frequencies['locations'].most_common(top_n)]

    # Add nodes with types
    for p in top_persons:
        if any(p in pair for pair in cooccurrences['person_org']) or \
           any(p in pair for pair in cooccurrences['person_location']):
            G.add_node(p, entity_type='person', freq=frequencies['persons'][p])

    for o in top_orgs:
        if any(o in pair for pair in cooccurrences['person_org']) or \
           any(o in pair for pair in cooccurrences['org_location']):
            G.add_node(o, entity_type='organization', freq=frequencies['organizations'][o])

    for l in top_locations:
        if any(l in pair for pair in cooccurrences['person_location']) or \
           any(l in pair for pair in cooccurrences['org_location']):
            G.add_node(l, entity_type='location', freq=frequencies['locations'][l])

    # Add edges
    for (p, o), weight in cooccurrences['person_org'].items():
        if p in G.nodes() and o in G.nodes():
            G.add_edge(p, o, weight=weight, rel_type='person-org')

    for (p, l), weight in cooccurrences['person_location'].items():
        if p in G.nodes() and l in G.nodes():
            G.add_edge(p, l, weight=weight, rel_type='person-location')

    for (o, l), weight in cooccurrences['org_location'].items():
        if o in G.nodes() and l in G.nodes():
            G.add_edge(o, l, weight=weight, rel_type='org-location')

    return G


def main():
    parser = argparse.ArgumentParser(description="Cross-entity relationship analysis")
    parser.add_argument("--min-freq", type=int, default=MIN_ENTITY_FREQ,
                        help=f"Minimum entity frequency (default: {MIN_ENTITY_FREQ})")
    parser.add_argument("--min-cooccurrence", type=int, default=MIN_COOCCURRENCE,
                        help=f"Minimum co-occurrence count (default: {MIN_COOCCURRENCE})")
    args = parser.parse_args()

    print("=== CROSS-ENTITY ANALYSIS ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load entities
    print("Loading extracted entities...")
    entities = load_entities()

    # Check if we have data
    total_files = sum(len(v) for v in entities.values())
    if total_files == 0:
        print("\nError: No entity data found. Run extraction scripts first:")
        print("  python extract_names.py")
        print("  python extract_locations.py")
        print("  python extract_organizations.py")
        print("  python extract_timeline.py")
        return

    # Compute co-occurrences
    print(f"\nComputing co-occurrences (min_freq={args.min_freq}, min_cooccur={args.min_cooccurrence})...")
    cooccurrences, frequencies = compute_cooccurrences(
        entities, args.min_freq, args.min_cooccurrence
    )

    # Display results
    print("\n" + "="*60)
    print("CROSS-ENTITY RELATIONSHIPS")
    print("="*60)

    # Person-Organization links
    print("\n--- Top 30 Person-Organization Links ---")
    person_org = sorted(cooccurrences['person_org'].items(), key=lambda x: x[1], reverse=True)
    for i, ((person, org), count) in enumerate(person_org[:30], 1):
        print(f"{i:2d}. {person} <-> {org} ({count} docs)")

    # Person-Location links
    print("\n--- Top 30 Person-Location Links ---")
    person_loc = sorted(cooccurrences['person_location'].items(), key=lambda x: x[1], reverse=True)
    for i, ((person, loc), count) in enumerate(person_loc[:30], 1):
        print(f"{i:2d}. {person} <-> {loc} ({count} docs)")

    # Org-Location links
    print("\n--- Top 30 Organization-Location Links ---")
    org_loc = sorted(cooccurrences['org_location'].items(), key=lambda x: x[1], reverse=True)
    for i, ((org, loc), count) in enumerate(org_loc[:30], 1):
        print(f"{i:2d}. {org} <-> {loc} ({count} docs)")

    # Person-Year links (activity timeline)
    print("\n--- Top 20 Most Active Persons by Year ---")
    person_year = sorted(cooccurrences['person_year'].items(), key=lambda x: x[1], reverse=True)
    # Group by person
    person_years = defaultdict(list)
    for (person, year), count in person_year:
        person_years[person].append((year, count))
    # Show top persons with their active years
    top_persons_activity = sorted(person_years.items(),
                                   key=lambda x: sum(c for _, c in x[1]), reverse=True)
    for i, (person, years) in enumerate(top_persons_activity[:20], 1):
        years_sorted = sorted(years, key=lambda x: x[0])
        year_range = f"{years_sorted[0][0]}-{years_sorted[-1][0]}"
        total = sum(c for _, c in years)
        print(f"{i:2d}. {person}: {year_range} ({total} mentions)")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save person-org links
    person_org_df = pd.DataFrame([
        {'person': p, 'organization': o, 'cooccurrences': c}
        for (p, o), c in sorted(cooccurrences['person_org'].items(), key=lambda x: x[1], reverse=True)
    ])
    person_org_df.to_csv(OUTPUT_DIR / "person_organization_links.csv", index=False)
    print(f"  Person-Org links: {OUTPUT_DIR / 'person_organization_links.csv'}")

    # Save person-location links
    person_loc_df = pd.DataFrame([
        {'person': p, 'location': l, 'cooccurrences': c}
        for (p, l), c in sorted(cooccurrences['person_location'].items(), key=lambda x: x[1], reverse=True)
    ])
    person_loc_df.to_csv(OUTPUT_DIR / "person_location_links.csv", index=False)
    print(f"  Person-Location links: {OUTPUT_DIR / 'person_location_links.csv'}")

    # Save org-location links
    org_loc_df = pd.DataFrame([
        {'organization': o, 'location': l, 'cooccurrences': c}
        for (o, l), c in sorted(cooccurrences['org_location'].items(), key=lambda x: x[1], reverse=True)
    ])
    org_loc_df.to_csv(OUTPUT_DIR / "org_location_links.csv", index=False)
    print(f"  Org-Location links: {OUTPUT_DIR / 'org_location_links.csv'}")

    # Save person-year links
    person_year_df = pd.DataFrame([
        {'person': p, 'year': y, 'cooccurrences': c}
        for (p, y), c in sorted(cooccurrences['person_year'].items(), key=lambda x: (x[0][0], x[0][1]))
    ])
    person_year_df.to_csv(OUTPUT_DIR / "person_year_links.csv", index=False)
    print(f"  Person-Year links: {OUTPUT_DIR / 'person_year_links.csv'}")

    # Save org-year links
    org_year_df = pd.DataFrame([
        {'organization': o, 'year': y, 'cooccurrences': c}
        for (o, y), c in sorted(cooccurrences['org_year'].items(), key=lambda x: (x[0][0], x[0][1]))
    ])
    org_year_df.to_csv(OUTPUT_DIR / "org_year_links.csv", index=False)
    print(f"  Org-Year links: {OUTPUT_DIR / 'org_year_links.csv'}")

    # Save combined relationship summary
    summary = {
        'person_org_count': len(cooccurrences['person_org']),
        'person_location_count': len(cooccurrences['person_location']),
        'person_year_count': len(cooccurrences['person_year']),
        'org_location_count': len(cooccurrences['org_location']),
        'org_year_count': len(cooccurrences['org_year']),
        'location_year_count': len(cooccurrences['location_year'])
    }
    with open(OUTPUT_DIR / "relationship_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {OUTPUT_DIR / 'relationship_summary.json'}")

    # Build and visualize multi-entity graph
    print("\nBuilding multi-entity graph...")
    G = build_multi_entity_graph(cooccurrences, frequencies, top_n=100)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Save graph data
    nodes_data = []
    for node in G.nodes():
        nodes_data.append({
            'entity': node,
            'type': G.nodes[node].get('entity_type', 'unknown'),
            'frequency': G.nodes[node].get('freq', 0),
            'degree': G.degree(node)
        })
    nodes_df = pd.DataFrame(nodes_data).sort_values('degree', ascending=False)
    nodes_df.to_csv(OUTPUT_DIR / "graph_nodes.csv", index=False)
    print(f"  Graph nodes: {OUTPUT_DIR / 'graph_nodes.csv'}")

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'entity1': u,
            'entity2': v,
            'weight': data.get('weight', 1),
            'relationship': data.get('rel_type', 'unknown')
        })
    edges_df = pd.DataFrame(edges_data).sort_values('weight', ascending=False)
    edges_df.to_csv(OUTPUT_DIR / "graph_edges.csv", index=False)
    print(f"  Graph edges: {OUTPUT_DIR / 'graph_edges.csv'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    if G.number_of_nodes() > 0:
        # Multi-entity network visualization
        fig, ax = plt.subplots(figsize=(24, 24))

        # Color map for entity types
        color_map = {
            'person': '#3498db',       # Blue
            'organization': '#e74c3c', # Red
            'location': '#2ecc71'      # Green
        }

        node_colors = [color_map.get(G.nodes[n].get('entity_type', 'unknown'), '#95a5a6')
                       for n in G.nodes()]
        node_sizes = [50 + 10 * G.nodes[n].get('freq', 1)**0.5 for n in G.nodes()]

        edge_widths = [0.5 + G[u][v].get('weight', 1) * 0.2 for u, v in G.edges()]

        # Layout
        pos = nx.spring_layout(G, k=3/G.number_of_nodes()**0.5, iterations=50, seed=42)

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#3498db', label='Person'),
            mpatches.Patch(color='#e74c3c', label='Organization'),
            mpatches.Patch(color='#2ecc71', label='Location')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

        ax.set_title(f"Cross-Entity Network\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "cross_entity_network.pdf", format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Network visualization: {OUTPUT_DIR / 'cross_entity_network.pdf'}")

    # Person-Year heatmap for top persons
    if cooccurrences['person_year']:
        # Get top 20 persons by total year mentions
        person_totals = defaultdict(int)
        for (person, year), count in cooccurrences['person_year'].items():
            person_totals[person] += count
        top_persons = [p for p, _ in sorted(person_totals.items(), key=lambda x: x[1], reverse=True)[:20]]

        # Get year range
        all_years = sorted(set(y for (_, y), _ in cooccurrences['person_year'].items()))
        if len(all_years) > 1:
            fig, ax = plt.subplots(figsize=(14, 10))

            matrix = []
            for person in top_persons:
                row = [cooccurrences['person_year'].get((person, year), 0) for year in all_years]
                matrix.append(row)

            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

            ax.set_xticks(range(len(all_years)))
            ax.set_xticklabels(all_years, rotation=45, ha='right')
            ax.set_yticks(range(len(top_persons)))
            ax.set_yticklabels(top_persons, fontsize=8)

            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Person', fontsize=12)
            ax.set_title('Person Activity by Year (Top 20)', fontsize=14)

            plt.colorbar(im, ax=ax, label='Mentions')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "person_year_heatmap.pdf", format='pdf', dpi=150)
            plt.close()
            print(f"  Person-Year heatmap: {OUTPUT_DIR / 'person_year_heatmap.pdf'}")

    # Relationship type distribution
    rel_counts = {
        'Person-Org': len(cooccurrences['person_org']),
        'Person-Location': len(cooccurrences['person_location']),
        'Person-Year': len(cooccurrences['person_year']),
        'Org-Location': len(cooccurrences['org_location']),
        'Org-Year': len(cooccurrences['org_year']),
        'Location-Year': len(cooccurrences['location_year'])
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(rel_counts.keys(), rel_counts.values(), color='steelblue', edgecolor='navy')
    ax.set_ylabel('Number of Relationships', fontsize=12)
    ax.set_title('Cross-Entity Relationship Counts', fontsize=14)
    plt.xticks(rotation=30, ha='right')

    for bar, count in zip(bars, rel_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "relationship_distribution.pdf", format='pdf', dpi=150)
    plt.close()
    print(f"  Relationship distribution: {OUTPUT_DIR / 'relationship_distribution.pdf'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Person-Organization links: {len(cooccurrences['person_org']):,}")
    print(f"Person-Location links: {len(cooccurrences['person_location']):,}")
    print(f"Person-Year links: {len(cooccurrences['person_year']):,}")
    print(f"Organization-Location links: {len(cooccurrences['org_location']):,}")
    print(f"Organization-Year links: {len(cooccurrences['org_year']):,}")
    print(f"Location-Year links: {len(cooccurrences['location_year']):,}")
    print(f"\nMulti-entity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nCross-entity analysis complete!")


if __name__ == "__main__":
    main()
