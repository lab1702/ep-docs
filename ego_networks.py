#!/usr/bin/env python3
"""
Ego Networks Script - Generate subgraphs centered on specific individuals.

Creates ego networks showing a person's direct connections (1-hop) and
optionally their connections' connections (2-hop).

Prerequisites:
    python generate_network.py  # Must run first to create network CSVs

Usage:
    python ego_networks.py --person "JEFFREY EPSTEIN"  # Specific person
    python ego_networks.py --search "CLINTON"          # Search for matches
    python ego_networks.py --top 10                    # Top 10 most connected
    python ego_networks.py --person "NAME" --depth 2   # Include 2-hop neighbors
"""

from pathlib import Path
import argparse
import json
import re

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
EDGES_CSV = Path("./network_edges_spacy.csv")
NODES_CSV = Path("./network_nodes_spacy.csv")
OUTPUT_DIR = Path("./ego_networks_output")


def load_network(min_weight=1):
    """Load network from CSV files."""
    if not EDGES_CSV.exists() or not NODES_CSV.exists():
        print(f"Error: Network files not found.")
        print(f"Run generate_network.py first to create {EDGES_CSV} and {NODES_CSV}")
        return None, None

    edges_df = pd.read_csv(EDGES_CSV)
    nodes_df = pd.read_csv(NODES_CSV)

    # Filter by minimum weight
    edges_df = edges_df[edges_df['weight'] >= min_weight]

    # Build graph
    G = nx.Graph()

    # Add nodes
    node_appearances = dict(zip(nodes_df['name'], nodes_df['appearances']))
    for name, appearances in node_appearances.items():
        G.add_node(name, appearances=appearances)

    # Add edges
    for _, row in edges_df.iterrows():
        if row['from'] in G.nodes() and row['to'] in G.nodes():
            G.add_edge(row['from'], row['to'], weight=row['weight'])

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G, node_appearances


def search_persons(G, query):
    """Search for persons matching a query."""
    query_upper = query.upper()
    matches = []
    for node in G.nodes():
        if query_upper in node.upper():
            matches.append((node, G.degree(node)))
    return sorted(matches, key=lambda x: x[1], reverse=True)


def extract_ego_network(G, person, depth=1):
    """Extract ego network for a person."""
    if person not in G.nodes():
        return None

    # Get neighbors at specified depth
    if depth == 1:
        neighbors = set(G.neighbors(person))
        ego_nodes = neighbors | {person}
    else:
        # BFS to get nodes within depth
        ego_nodes = {person}
        frontier = {person}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(G.neighbors(node))
            ego_nodes.update(next_frontier)
            frontier = next_frontier

    # Create subgraph
    ego_graph = G.subgraph(ego_nodes).copy()

    return ego_graph


def calculate_ego_stats(G, ego_graph, person):
    """Calculate statistics for an ego network."""
    stats = {
        'ego': person,
        'ego_appearances': G.nodes[person].get('appearances', 0),
        'degree': G.degree(person),
        'ego_network_nodes': ego_graph.number_of_nodes(),
        'ego_network_edges': ego_graph.number_of_edges(),
        'ego_density': nx.density(ego_graph),
    }

    # Centrality in full network
    stats['betweenness'] = nx.betweenness_centrality(G).get(person, 0)

    # Get neighbor details
    neighbors = list(G.neighbors(person))
    neighbor_weights = [(n, G[person][n]['weight']) for n in neighbors]
    neighbor_weights.sort(key=lambda x: x[1], reverse=True)

    stats['top_connections'] = neighbor_weights[:20]
    stats['total_edge_weight'] = sum(w for _, w in neighbor_weights)
    stats['avg_edge_weight'] = stats['total_edge_weight'] / len(neighbors) if neighbors else 0

    # Clustering coefficient
    stats['clustering'] = nx.clustering(G, person)

    return stats


def visualize_ego_network(ego_graph, person, node_appearances, output_path, depth=1):
    """Visualize an ego network."""
    fig, ax = plt.subplots(figsize=(16, 16))

    # Node colors: ego is red, direct neighbors are blue, 2-hop are gray
    node_colors = []
    for node in ego_graph.nodes():
        if node == person:
            node_colors.append('#e74c3c')  # Red for ego
        elif ego_graph.has_edge(person, node):
            node_colors.append('#3498db')  # Blue for direct connections
        else:
            node_colors.append('#95a5a6')  # Gray for 2-hop

    # Node sizes based on connection to ego or appearances
    node_sizes = []
    for node in ego_graph.nodes():
        if node == person:
            node_sizes.append(1000)
        elif ego_graph.has_edge(person, node):
            weight = ego_graph[person][node]['weight']
            node_sizes.append(200 + weight * 50)
        else:
            node_sizes.append(100)

    # Edge widths
    edge_widths = []
    edge_colors = []
    for u, v in ego_graph.edges():
        weight = ego_graph[u][v]['weight']
        if u == person or v == person:
            edge_widths.append(1 + weight * 0.5)
            edge_colors.append('#e74c3c')
        else:
            edge_widths.append(0.5 + weight * 0.2)
            edge_colors.append('#cccccc')

    # Layout - ego in center
    pos = nx.spring_layout(ego_graph, k=2/ego_graph.number_of_nodes()**0.5,
                           iterations=50, seed=42)

    # Draw
    nx.draw_networkx_edges(ego_graph, pos, width=edge_widths, alpha=0.6,
                           edge_color=edge_colors, ax=ax)
    nx.draw_networkx_nodes(ego_graph, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(ego_graph, pos, font_size=7, ax=ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label=f'Ego: {person}'),
        mpatches.Patch(color='#3498db', label='Direct connections'),
    ]
    if depth > 1:
        legend_elements.append(mpatches.Patch(color='#95a5a6', label='2-hop connections'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_title(f"Ego Network: {person}\n"
                 f"Nodes: {ego_graph.number_of_nodes()} | "
                 f"Edges: {ego_graph.number_of_edges()} | "
                 f"Depth: {depth}", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()


def format_ego_report(stats):
    """Format ego network statistics as text."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"EGO NETWORK: {stats['ego']}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("--- Network Position ---")
    lines.append(f"Document appearances: {stats['ego_appearances']}")
    lines.append(f"Direct connections (degree): {stats['degree']}")
    lines.append(f"Betweenness centrality: {stats['betweenness']:.4f}")
    lines.append(f"Clustering coefficient: {stats['clustering']:.4f}")
    lines.append("")

    lines.append("--- Ego Network Stats ---")
    lines.append(f"Nodes in ego network: {stats['ego_network_nodes']}")
    lines.append(f"Edges in ego network: {stats['ego_network_edges']}")
    lines.append(f"Ego network density: {stats['ego_density']:.4f}")
    lines.append(f"Total edge weight: {stats['total_edge_weight']}")
    lines.append(f"Average edge weight: {stats['avg_edge_weight']:.2f}")
    lines.append("")

    lines.append("--- Top Connections (by shared documents) ---")
    for i, (name, weight) in enumerate(stats['top_connections'], 1):
        lines.append(f"  {i:2d}. {name}: {weight} shared docs")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate ego networks")
    parser.add_argument("--person", type=str, default=None,
                        help="Person name to generate ego network for")
    parser.add_argument("--search", type=str, default=None,
                        help="Search for persons by name")
    parser.add_argument("--top", type=int, default=None,
                        help="Generate ego networks for top N most connected")
    parser.add_argument("--depth", type=int, default=1, choices=[1, 2],
                        help="Network depth: 1=direct connections, 2=include 2-hop (default: 1)")
    parser.add_argument("--min-weight", type=int, default=2,
                        help="Minimum edge weight to include (default: 2)")
    args = parser.parse_args()

    print("=== EGO NETWORKS ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load network
    print(f"Loading network (min edge weight: {args.min_weight})...")
    G, node_appearances = load_network(min_weight=args.min_weight)

    if G is None:
        return

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Handle search mode
    if args.search:
        print(f"\n--- Searching for: '{args.search}' ---")
        matches = search_persons(G, args.search)

        if not matches:
            print("No matches found.")
            return

        print(f"Found {len(matches)} matches:\n")
        for person, degree in matches[:20]:
            print(f"  {person}: {degree} connections")

        print("\nUse --person 'NAME' to generate ego network for a specific person.")
        return

    # Determine which persons to process
    persons_to_process = []

    if args.person:
        # Search for exact or partial match
        if args.person in G.nodes():
            persons_to_process = [(args.person, G.degree(args.person))]
        else:
            matches = search_persons(G, args.person)
            if matches:
                print(f"Exact match not found. Did you mean one of these?")
                for person, degree in matches[:10]:
                    print(f"  {person}: {degree} connections")
                return
            else:
                print(f"Person '{args.person}' not found in network.")
                return

    elif args.top:
        # Get top N most connected
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        persons_to_process = degree_sorted[:args.top]

    else:
        # Default: show top 10 and prompt
        print("\n--- Top 20 Most Connected Persons ---")
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        for i, (person, degree) in enumerate(degree_sorted[:20], 1):
            print(f"  {i:2d}. {person}: {degree} connections")

        print("\nUse --person 'NAME' or --top N to generate ego networks.")
        return

    # Process each person
    all_stats = []

    for person, degree in persons_to_process:
        print(f"\nProcessing: {person} ({degree} connections)")

        # Extract ego network
        ego_graph = extract_ego_network(G, person, depth=args.depth)

        if ego_graph is None:
            print(f"  Error: Could not extract ego network")
            continue

        # Calculate stats
        stats = calculate_ego_stats(G, ego_graph, person)
        all_stats.append(stats)

        # Print report
        print(format_ego_report(stats))

        # Generate visualization
        safe_name = re.sub(r'[^\w\s-]', '', person).replace(' ', '_')[:40]
        output_path = OUTPUT_DIR / f"ego_{safe_name}.pdf"
        visualize_ego_network(ego_graph, person, node_appearances, output_path, args.depth)
        print(f"  Visualization saved: {output_path}")

        # Save ego network data
        ego_nodes_df = pd.DataFrame([
            {
                'name': n,
                'appearances': ego_graph.nodes[n].get('appearances', 0),
                'degree_in_ego': ego_graph.degree(n),
                'is_ego': n == person,
                'direct_connection': ego_graph.has_edge(person, n) if n != person else False
            }
            for n in ego_graph.nodes()
        ]).sort_values('degree_in_ego', ascending=False)

        ego_edges_df = pd.DataFrame([
            {
                'from': u,
                'to': v,
                'weight': ego_graph[u][v]['weight'],
                'involves_ego': u == person or v == person
            }
            for u, v in ego_graph.edges()
        ]).sort_values('weight', ascending=False)

        ego_nodes_df.to_csv(OUTPUT_DIR / f"ego_{safe_name}_nodes.csv", index=False)
        ego_edges_df.to_csv(OUTPUT_DIR / f"ego_{safe_name}_edges.csv", index=False)

    # Save summary if multiple persons processed
    if len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("SAVING SUMMARY")
        print("=" * 60 + "\n")

        summary_df = pd.DataFrame([
            {
                'name': s['ego'],
                'appearances': s['ego_appearances'],
                'degree': s['degree'],
                'betweenness': s['betweenness'],
                'clustering': s['clustering'],
                'ego_network_nodes': s['ego_network_nodes'],
                'ego_network_edges': s['ego_network_edges'],
                'ego_density': s['ego_density'],
                'avg_edge_weight': s['avg_edge_weight']
            }
            for s in all_stats
        ]).sort_values('degree', ascending=False)

        summary_df.to_csv(OUTPUT_DIR / "ego_summary.csv", index=False)
        print(f"  Summary: {OUTPUT_DIR / 'ego_summary.csv'}")

        # Save all stats as JSON
        for s in all_stats:
            s['top_connections'] = [(n, int(w)) for n, w in s['top_connections']]

        with open(OUTPUT_DIR / "ego_stats.json", 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2)
        print(f"  Stats: {OUTPUT_DIR / 'ego_stats.json'}")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nEgo networks complete!")


if __name__ == "__main__":
    main()
