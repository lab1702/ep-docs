#!/usr/bin/env python3
"""
Entity Profiles Script - Build comprehensive dossiers for individuals.

Aggregates all information about each person across documents including
associated organizations, locations, dates, co-occurring people, and context.

Prerequisites:
    Run extraction scripts first:
    - python extract_names.py
    - python extract_locations.py
    - python extract_organizations.py
    - python extract_timeline.py

Usage:
    python entity_profiles.py                    # Build all profiles
    python entity_profiles.py --top 100          # Top 100 most mentioned
    python entity_profiles.py --search "EPSTEIN" # Search for specific person
"""

from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import re

import pandas as pd

# Configuration
BASE_DIR = Path("./epstein_pdfs")
OUTPUT_DIR = Path("./entity_profiles_output")

# Input files from other extraction scripts
NAMES_FILE = Path("./extracted_names.json")
LOCATIONS_FILE = Path("./location_extraction_output/extracted_locations.json")
ORGS_FILE = Path("./organization_extraction_output/extracted_organizations.json")
TIMELINE_FILE = Path("./timeline_extraction_output/extracted_dates.json")

# Minimum mentions to generate a profile
MIN_MENTIONS = 5
CONTEXT_LENGTH = 200  # Characters of context to extract


def load_entities():
    """Load all extracted entities from JSON files."""
    entities = {
        'persons': {},
        'locations': {},
        'organizations': {},
        'dates': {}
    }

    if NAMES_FILE.exists():
        with open(NAMES_FILE, 'r', encoding='utf-8') as f:
            entities['persons'] = json.load(f)
        print(f"  Loaded persons from {len(entities['persons'])} files")

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

    if ORGS_FILE.exists():
        with open(ORGS_FILE, 'r', encoding='utf-8') as f:
            entities['organizations'] = json.load(f)
        print(f"  Loaded organizations from {len(entities['organizations'])} files")

    if TIMELINE_FILE.exists():
        with open(TIMELINE_FILE, 'r', encoding='utf-8') as f:
            entities['dates'] = json.load(f)
        print(f"  Loaded dates from {len(entities['dates'])} files")

    return entities


def get_person_mentions(entities):
    """Count how many files each person appears in."""
    person_files = defaultdict(set)
    for file_path, persons in entities['persons'].items():
        for person in persons:
            person_files[person].add(file_path)
    return {p: len(files) for p, files in person_files.items()}, person_files


def extract_context(file_path, person_name, max_contexts=5):
    """Extract text context around person mentions."""
    contexts = []
    try:
        text = Path(file_path).read_text(encoding='utf-8', errors='ignore')
        # Clean text
        text = ' '.join(text.split())

        # Find mentions (case-insensitive)
        pattern = re.compile(re.escape(person_name), re.IGNORECASE)
        for match in pattern.finditer(text):
            start = max(0, match.start() - CONTEXT_LENGTH)
            end = min(len(text), match.end() + CONTEXT_LENGTH)
            context = text[start:end].strip()
            if context and len(context) > 50:
                # Add ellipsis if truncated
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                contexts.append(context)
                if len(contexts) >= max_contexts:
                    break
    except Exception:
        pass
    return contexts


def build_profile(person, person_files, entities, include_context=True):
    """Build a comprehensive profile for a person."""
    files = person_files[person]

    profile = {
        'name': person,
        'total_mentions': len(files),
        'documents': list(files),
        'organizations': Counter(),
        'locations': Counter(),
        'years': Counter(),
        'co_occurring_persons': Counter(),
        'contexts': []
    }

    for file_path in files:
        # Get co-occurring organizations
        orgs = entities['organizations'].get(file_path, [])
        profile['organizations'].update(orgs)

        # Get co-occurring locations
        locs = entities['locations'].get(file_path, [])
        profile['locations'].update(locs)

        # Get co-occurring dates/years
        dates = entities['dates'].get(file_path, [])
        for d in dates:
            if isinstance(d, dict) and 'year' in d:
                profile['years'][str(d['year'])] += 1

        # Get co-occurring persons (excluding self)
        persons = entities['persons'].get(file_path, [])
        for p in persons:
            if p != person:
                profile['co_occurring_persons'][p] += 1

    # Extract context snippets
    if include_context:
        for file_path in list(files)[:10]:  # Limit to 10 files for context
            contexts = extract_context(file_path, person, max_contexts=2)
            profile['contexts'].extend(contexts)
        profile['contexts'] = profile['contexts'][:20]  # Limit total contexts

    # Convert Counters to sorted lists
    profile['top_organizations'] = profile['organizations'].most_common(20)
    profile['top_locations'] = profile['locations'].most_common(20)
    profile['top_years'] = sorted(profile['years'].items(), key=lambda x: x[0])
    profile['top_associates'] = profile['co_occurring_persons'].most_common(20)

    # Summary stats
    profile['stats'] = {
        'unique_organizations': len(profile['organizations']),
        'unique_locations': len(profile['locations']),
        'unique_years': len(profile['years']),
        'unique_associates': len(profile['co_occurring_persons']),
        'year_range': f"{min(profile['years'].keys())}-{max(profile['years'].keys())}" if profile['years'] else "N/A"
    }

    return profile


def format_profile_text(profile):
    """Format profile as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"PROFILE: {profile['name']}")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Total Document Mentions: {profile['total_mentions']}")
    lines.append(f"Unique Associates: {profile['stats']['unique_associates']}")
    lines.append(f"Unique Organizations: {profile['stats']['unique_organizations']}")
    lines.append(f"Unique Locations: {profile['stats']['unique_locations']}")
    lines.append(f"Year Range: {profile['stats']['year_range']}")
    lines.append("")

    if profile['top_associates']:
        lines.append("--- Top Associates (co-occurring persons) ---")
        for name, count in profile['top_associates'][:10]:
            lines.append(f"  {name}: {count} shared documents")
        lines.append("")

    if profile['top_organizations']:
        lines.append("--- Associated Organizations ---")
        for org, count in profile['top_organizations'][:10]:
            lines.append(f"  {org}: {count} mentions")
        lines.append("")

    if profile['top_locations']:
        lines.append("--- Associated Locations ---")
        for loc, count in profile['top_locations'][:10]:
            lines.append(f"  {loc}: {count} mentions")
        lines.append("")

    if profile['top_years']:
        lines.append("--- Activity by Year ---")
        for year, count in profile['top_years']:
            bar = "#" * min(30, count)
            lines.append(f"  {year}: {count:3d} {bar}")
        lines.append("")

    if profile['contexts']:
        lines.append("--- Sample Context Snippets ---")
        for i, ctx in enumerate(profile['contexts'][:5], 1):
            lines.append(f"\n  [{i}] {ctx}")
        lines.append("")

    return "\n".join(lines)


def search_persons(query, person_counts):
    """Search for persons matching a query."""
    query_upper = query.upper()
    matches = []
    for person, count in person_counts.items():
        if query_upper in person.upper():
            matches.append((person, count))
    return sorted(matches, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Build entity profiles for individuals")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top persons to profile (default: 50)")
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS,
                        help=f"Minimum mentions to include (default: {MIN_MENTIONS})")
    parser.add_argument("--search", type=str, default=None,
                        help="Search for specific person by name")
    parser.add_argument("--no-context", action="store_true",
                        help="Skip extracting context snippets (faster)")
    args = parser.parse_args()

    print("=== ENTITY PROFILES ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load entities
    print("Loading extracted entities...")
    entities = load_entities()

    if not entities['persons']:
        print("\nError: No person data found. Run extraction scripts first:")
        print("  python extract_names.py")
        return

    # Get person mention counts
    print("\nAnalyzing person mentions...")
    person_counts, person_files = get_person_mentions(entities)
    print(f"Total unique persons: {len(person_counts)}")

    # Filter by minimum mentions
    qualified = {p: c for p, c in person_counts.items() if c >= args.min_mentions}
    print(f"Persons with >= {args.min_mentions} mentions: {len(qualified)}")

    # Handle search mode
    if args.search:
        print(f"\n--- Searching for: '{args.search}' ---")
        matches = search_persons(args.search, qualified)

        if not matches:
            print("No matches found.")
            return

        print(f"Found {len(matches)} matches:\n")
        for person, count in matches[:20]:
            print(f"  {person}: {count} mentions")

        # Build profiles for matches
        print("\nBuilding profiles for matches...")
        profiles = []
        for person, _ in matches[:10]:  # Top 10 matches
            print(f"  Processing: {person}")
            profile = build_profile(person, person_files, entities,
                                   include_context=not args.no_context)
            profiles.append(profile)
            print(format_profile_text(profile))

        # Save search results
        search_file = OUTPUT_DIR / f"search_{args.search.replace(' ', '_')}.json"
        with open(search_file, 'w', encoding='utf-8') as f:
            # Convert Counters for JSON serialization
            for p in profiles:
                p['organizations'] = dict(p['organizations'])
                p['locations'] = dict(p['locations'])
                p['years'] = dict(p['years'])
                p['co_occurring_persons'] = dict(p['co_occurring_persons'])
            json.dump(profiles, f, indent=2)
        print(f"\nSearch results saved to: {search_file}")
        return

    # Build profiles for top persons
    print(f"\nBuilding profiles for top {args.top} persons...")
    top_persons = sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:args.top]

    profiles = []
    for i, (person, count) in enumerate(top_persons, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(top_persons)}")
        profile = build_profile(person, person_files, entities,
                               include_context=not args.no_context)
        profiles.append(profile)

    # Display top profiles
    print("\n" + "=" * 70)
    print("TOP ENTITY PROFILES")
    print("=" * 70)

    for profile in profiles[:10]:
        print(format_profile_text(profile))

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60 + "\n")

    # Save all profiles as JSON
    profiles_for_json = []
    for p in profiles:
        profile_copy = p.copy()
        profile_copy['organizations'] = dict(p['organizations'])
        profile_copy['locations'] = dict(p['locations'])
        profile_copy['years'] = dict(p['years'])
        profile_copy['co_occurring_persons'] = dict(p['co_occurring_persons'])
        profiles_for_json.append(profile_copy)

    with open(OUTPUT_DIR / "all_profiles.json", 'w', encoding='utf-8') as f:
        json.dump(profiles_for_json, f, indent=2)
    print(f"  All profiles: {OUTPUT_DIR / 'all_profiles.json'}")

    # Save summary table
    summary_data = []
    for p in profiles:
        summary_data.append({
            'name': p['name'],
            'mentions': p['total_mentions'],
            'associates': p['stats']['unique_associates'],
            'organizations': p['stats']['unique_organizations'],
            'locations': p['stats']['unique_locations'],
            'year_range': p['stats']['year_range'],
            'top_associate': p['top_associates'][0][0] if p['top_associates'] else '',
            'top_organization': p['top_organizations'][0][0] if p['top_organizations'] else '',
            'top_location': p['top_locations'][0][0] if p['top_locations'] else ''
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "profile_summary.csv", index=False)
    print(f"  Summary table: {OUTPUT_DIR / 'profile_summary.csv'}")

    # Save individual profile text files for top 20
    profiles_dir = OUTPUT_DIR / "individual_profiles"
    profiles_dir.mkdir(exist_ok=True)
    for i, profile in enumerate(profiles[:20], 1):
        safe_name = re.sub(r'[^\w\s-]', '', profile['name']).replace(' ', '_')[:50]
        filename = f"{i:02d}_{safe_name}.txt"
        with open(profiles_dir / filename, 'w', encoding='utf-8') as f:
            f.write(format_profile_text(profile))
    print(f"  Individual profiles: {profiles_dir}/")

    # Save associate network (who appears with whom)
    associate_edges = []
    for p in profiles:
        for associate, count in p['top_associates'][:10]:
            if count >= 3:  # Minimum shared documents
                associate_edges.append({
                    'person': p['name'],
                    'associate': associate,
                    'shared_documents': count
                })

    if associate_edges:
        associate_df = pd.DataFrame(associate_edges)
        associate_df.to_csv(OUTPUT_DIR / "associate_network.csv", index=False)
        print(f"  Associate network: {OUTPUT_DIR / 'associate_network.csv'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Profiles generated: {len(profiles)}")
    print(f"Total unique persons: {len(person_counts)}")
    print(f"Persons with >= {args.min_mentions} mentions: {len(qualified)}")

    if profiles:
        avg_associates = sum(p['stats']['unique_associates'] for p in profiles) / len(profiles)
        avg_orgs = sum(p['stats']['unique_organizations'] for p in profiles) / len(profiles)
        print(f"Average associates per person: {avg_associates:.1f}")
        print(f"Average organizations per person: {avg_orgs:.1f}")

    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nEntity profiles complete!")
    print("\nTip: Use --search 'NAME' to search for specific individuals")


if __name__ == "__main__":
    main()
