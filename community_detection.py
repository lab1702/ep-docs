#!/usr/bin/env python3
"""
Community Detection Script - Identify clusters in the co-occurrence network.

Uses the Louvain algorithm to detect communities and calculates centrality
metrics to identify key individuals.

Prerequisites:
    python generate_network.py  # Must run first to create network CSVs

Usage:
    python community_detection.py                # Run with defaults
    python community_detection.py --min-weight 5 # Higher edge weight threshold
    python community_detection.py --top 50       # Show top 50 by centrality
"""

from pathlib import Path
import argparse
import json

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import community as community_louvain

# Configuration
EDGES_CSV = Path("./network_edges_spacy.csv")
NODES_CSV = Path("./network_nodes_spacy.csv")
OUTPUT_DIR = Path("./community_detection_output")


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
    for name in node_appearances:
        G.add_node(name, appearances=node_appearances.get(name, 0))

    # Add edges
    for _, row in edges_df.iterrows():
        if row['from'] in G.nodes() and row['to'] in G.nodes():
            G.add_edge(row['from'], row['to'], weight=row['weight'])

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    return G, node_appearances


def detect_communities(G):
    """Detect communities using Louvain algorithm."""
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    return partition


def calculate_centrality_metrics(G):
    """Calculate various centrality metrics."""
    metrics = {}

    print("  Calculating degree centrality...")
    metrics['degree'] = dict(G.degree())

    print("  Calculating weighted degree...")
    metrics['weighted_degree'] = dict(G.degree(weight='weight'))

    print("  Calculating betweenness centrality...")
    metrics['betweenness'] = nx.betweenness_centrality(G, weight='weight')

    print("  Calculating PageRank...")
    metrics['pagerank'] = nx.pagerank(G, weight='weight')

    print("  Calculating eigenvector centrality...")
    try:
        metrics['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
    except nx.PowerIterationFailedConvergence:
        print("    (eigenvector centrality did not converge, using degree as fallback)")
        metrics['eigenvector'] = {n: d / max(metrics['degree'].values())
                                   for n, d in metrics['degree'].items()}

    return metrics


def get_community_summary(G, partition, node_appearances):
    """Generate summary statistics for each community."""
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    summaries = []
    for comm_id, members in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
        # Get subgraph for this community
        subgraph = G.subgraph(members)

        # Find most connected member within community
        internal_degrees = dict(subgraph.degree())
        top_member = max(internal_degrees.items(), key=lambda x: x[1])[0] if internal_degrees else None

        # Calculate total appearances
        total_appearances = sum(node_appearances.get(m, 0) for m in members)

        summaries.append({
            'community_id': comm_id,
            'size': len(members),
            'internal_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph) if len(members) > 1 else 0,
            'total_appearances': total_appearances,
            'top_member': top_member,
            'members': members
        })

    return summaries


def visualize_communities(G, partition, output_path, node_appearances):
    """Create network visualization colored by community."""
    print("\nGenerating community visualization...")

    # Get unique communities and assign colors
    communities = set(partition.values())
    n_communities = len(communities)
    colors = cm.get_cmap('tab20', max(n_communities, 20))

    # Node colors based on community
    node_colors = [colors(partition[node] % 20) for node in G.nodes()]

    # Node sizes based on appearances
    node_sizes = [100 + 30 * (node_appearances.get(n, 1) ** 0.5) for n in G.nodes()]

    # Edge widths
    edge_widths = [0.3 + G[u][v]['weight'] * 0.2 for u, v in G.edges()]

    # Create figure
    n_nodes = G.number_of_nodes()
    fig_size = max(20, min(50, n_nodes ** 0.5 * 2))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Layout - use community-aware layout if available
    pos = nx.spring_layout(G, k=2/n_nodes**0.5, iterations=100, seed=42, weight='weight')

    # Draw
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax)

    ax.set_title(f"Network Communities (Louvain Algorithm)\n"
                 f"Nodes: {G.number_of_nodes()} | Communities: {n_communities}",
                 fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Community detection in co-occurrence network")
    parser.add_argument("--min-weight", type=int, default=3,
                        help="Minimum edge weight to include (default: 3)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top individuals to show per metric (default: 20)")
    args = parser.parse_args()

    print("=== COMMUNITY DETECTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load network
    print(f"Loading network (min edge weight: {args.min_weight})...")
    G, node_appearances = load_network(min_weight=args.min_weight)

    if G is None:
        return

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}\n")

    if G.number_of_nodes() < 3:
        print("Error: Network too small for community detection")
        return

    # Detect communities
    print("Detecting communities (Louvain algorithm)...")
    partition = detect_communities(G)
    n_communities = len(set(partition.values()))
    print(f"  Found {n_communities} communities\n")

    # Calculate modularity
    modularity = community_louvain.modularity(partition, G, weight='weight')
    print(f"  Modularity score: {modularity:.4f}")
    print("  (Higher modularity = stronger community structure)\n")

    # Community summaries
    print("="*60)
    print("COMMUNITY SUMMARIES")
    print("="*60 + "\n")

    summaries = get_community_summary(G, partition, node_appearances)

    for i, summary in enumerate(summaries[:15]):  # Show top 15 communities
        print(f"Community {summary['community_id']} ({summary['size']} members)")
        print(f"  Internal edges: {summary['internal_edges']}")
        print(f"  Density: {summary['density']:.3f}")
        print(f"  Key figure: {summary['top_member']}")
        # Show top 5 members by appearance
        top_members = sorted(summary['members'],
                           key=lambda x: node_appearances.get(x, 0),
                           reverse=True)[:5]
        print(f"  Top members: {', '.join(top_members)}")
        print()

    # Calculate centrality metrics
    print("="*60)
    print("CENTRALITY METRICS")
    print("="*60 + "\n")

    print("Calculating centrality metrics...")
    metrics = calculate_centrality_metrics(G)

    # Display top individuals by each metric
    metric_names = {
        'degree': 'Degree (# connections)',
        'weighted_degree': 'Weighted Degree (connection strength)',
        'betweenness': 'Betweenness (bridge between groups)',
        'pagerank': 'PageRank (influence)',
        'eigenvector': 'Eigenvector (connected to important nodes)'
    }

    for metric_key, metric_name in metric_names.items():
        print(f"\n--- Top {args.top} by {metric_name} ---")
        sorted_nodes = sorted(metrics[metric_key].items(), key=lambda x: x[1], reverse=True)
        for i, (name, value) in enumerate(sorted_nodes[:args.top], 1):
            comm = partition[name]
            if metric_key in ['betweenness', 'pagerank', 'eigenvector']:
                print(f"{i:3d}. {name:<40} {value:.4f}  (Community {comm})")
            else:
                print(f"{i:3d}. {name:<40} {value:>6}  (Community {comm})")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save community assignments
    community_df = pd.DataFrame([
        {
            'name': name,
            'community': partition[name],
            'appearances': node_appearances.get(name, 0),
            'degree': metrics['degree'][name],
            'weighted_degree': metrics['weighted_degree'][name],
            'betweenness': metrics['betweenness'][name],
            'pagerank': metrics['pagerank'][name],
            'eigenvector': metrics['eigenvector'][name]
        }
        for name in G.nodes()
    ]).sort_values(['community', 'pagerank'], ascending=[True, False])

    community_df.to_csv(OUTPUT_DIR / "node_communities.csv", index=False)
    print(f"  Node communities: {OUTPUT_DIR / 'node_communities.csv'}")

    # Save community summaries
    summary_df = pd.DataFrame([
        {
            'community_id': s['community_id'],
            'size': s['size'],
            'internal_edges': s['internal_edges'],
            'density': s['density'],
            'total_appearances': s['total_appearances'],
            'key_figure': s['top_member']
        }
        for s in summaries
    ])
    summary_df.to_csv(OUTPUT_DIR / "community_summaries.csv", index=False)
    print(f"  Community summaries: {OUTPUT_DIR / 'community_summaries.csv'}")

    # Save detailed community members as JSON
    community_members = {}
    for s in summaries:
        community_members[str(s['community_id'])] = {
            'size': s['size'],
            'key_figure': s['top_member'],
            'members': sorted(s['members'],
                            key=lambda x: node_appearances.get(x, 0),
                            reverse=True)
        }

    with open(OUTPUT_DIR / "community_members.json", 'w', encoding='utf-8') as f:
        json.dump(community_members, f, indent=2)
    print(f"  Community members: {OUTPUT_DIR / 'community_members.json'}")

    # Generate visualization
    visualize_communities(G, partition, OUTPUT_DIR / "community_network.pdf", node_appearances)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")
    print(f"Communities found: {n_communities}")
    print(f"Modularity: {modularity:.4f}")
    print(f"Largest community: {summaries[0]['size']} members")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nCommunity detection complete!")


if __name__ == "__main__":
    main()
