#!/usr/bin/env python3
"""
Alias Resolution Script - Merge name variants and find duplicates.

Uses fuzzy string matching to identify name variants (nicknames, misspellings,
different formats) and group them into canonical forms.

Usage:
    python alias_resolution.py                    # Find aliases
    python alias_resolution.py --threshold 85    # Adjust similarity threshold
    python alias_resolution.py --apply           # Apply aliases to extracted_names.json
"""

from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import re

import pandas as pd
from difflib import SequenceMatcher

# Configuration
NAMES_FILE = Path("./extracted_names.json")
OUTPUT_DIR = Path("./alias_resolution_output")

# Common nickname mappings
NICKNAME_MAP = {
    'WILLIAM': ['BILL', 'BILLY', 'WILL', 'WILLIE', 'WM'],
    'ROBERT': ['BOB', 'BOBBY', 'ROB', 'ROBBY', 'ROBBIE'],
    'RICHARD': ['RICK', 'RICKY', 'DICK', 'RICH', 'RICHIE'],
    'JAMES': ['JIM', 'JIMMY', 'JAMIE', 'JAS'],
    'MICHAEL': ['MIKE', 'MIKEY', 'MICK'],
    'JOSEPH': ['JOE', 'JOEY', 'JOS'],
    'THOMAS': ['TOM', 'TOMMY', 'THOM'],
    'CHARLES': ['CHARLIE', 'CHUCK', 'CHAS'],
    'CHRISTOPHER': ['CHRIS', 'CHRISTY', 'KIT'],
    'DANIEL': ['DAN', 'DANNY', 'DANNIE'],
    'MATTHEW': ['MATT', 'MATTY'],
    'ANTHONY': ['TONY', 'ANT'],
    'DONALD': ['DON', 'DONNIE', 'DONNY'],
    'STEVEN': ['STEVE', 'STEVIE', 'STEPHEN'],
    'EDWARD': ['ED', 'EDDIE', 'EDDY', 'TED', 'TEDDY'],
    'KENNETH': ['KEN', 'KENNY'],
    'RONALD': ['RON', 'RONNIE', 'RONNY'],
    'JEFFREY': ['JEFF', 'GEOFF', 'GEOFFREY'],
    'JONATHAN': ['JON', 'JOHN', 'JOHNNY'],
    'BENJAMIN': ['BEN', 'BENNY', 'BENJI'],
    'ALEXANDER': ['ALEX', 'SANDY', 'XANDER'],
    'ELIZABETH': ['LIZ', 'LIZZY', 'BETH', 'BETTY', 'ELIZA'],
    'JENNIFER': ['JEN', 'JENNY', 'JENN'],
    'MARGARET': ['MAGGIE', 'MEG', 'PEGGY', 'MARGE'],
    'PATRICIA': ['PAT', 'PATTY', 'TRISH', 'TRICIA'],
    'KATHERINE': ['KATE', 'KATIE', 'KATHY', 'KAT', 'CATHERINE'],
    'VIRGINIA': ['GINNY', 'GINGER', 'VIRGIE'],
    'DEBORAH': ['DEB', 'DEBBIE', 'DEBBY'],
    'REBECCA': ['BECKY', 'BECCA'],
    'SAMANTHA': ['SAM', 'SAMMY'],
    'ALEXANDRA': ['ALEX', 'SANDY', 'ALEXA'],
}

# Build reverse lookup
NICKNAME_REVERSE = {}
for canonical, nicknames in NICKNAME_MAP.items():
    for nick in nicknames:
        NICKNAME_REVERSE[nick] = canonical


def load_names():
    """Load extracted names from JSON file."""
    if not NAMES_FILE.exists():
        print(f"Error: {NAMES_FILE} not found")
        print("Run extract_names.py first")
        return None, None

    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        file_names = json.load(f)

    # Count name frequencies
    name_counts = Counter()
    for names in file_names.values():
        name_counts.update(names)

    return file_names, name_counts


def normalize_name(name):
    """Normalize a name for comparison."""
    # Uppercase
    name = name.upper()
    # Remove punctuation except hyphens
    name = re.sub(r'[^\w\s\-]', '', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name


def get_name_parts(name):
    """Split a name into parts."""
    parts = name.split()
    return {
        'full': name,
        'first': parts[0] if parts else '',
        'last': parts[-1] if parts else '',
        'middle': ' '.join(parts[1:-1]) if len(parts) > 2 else '',
        'parts': parts
    }


def similarity_score(name1, name2):
    """Calculate similarity score between two names."""
    return SequenceMatcher(None, name1, name2).ratio() * 100


def names_match(name1, name2, threshold=80):
    """Check if two names likely refer to the same person."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    # Exact match
    if n1 == n2:
        return True, 100, 'exact'

    parts1 = get_name_parts(n1)
    parts2 = get_name_parts(n2)

    # Same last name required for most matches
    if parts1['last'] != parts2['last']:
        # Check if one is substring of other (for hyphenated names)
        if not (parts1['last'] in parts2['last'] or parts2['last'] in parts1['last']):
            return False, 0, None

    # Check first name with nicknames
    first1 = parts1['first']
    first2 = parts2['first']

    # Direct first name match
    if first1 == first2:
        score = similarity_score(n1, n2)
        if score >= threshold:
            return True, score, 'same_first_last'

    # Check nickname mappings
    canonical1 = NICKNAME_REVERSE.get(first1, first1)
    canonical2 = NICKNAME_REVERSE.get(first2, first2)

    if canonical1 == canonical2:
        return True, 90, 'nickname'

    # Check if first1 is nickname of first2 or vice versa
    if first1 in NICKNAME_MAP.get(first2, []) or first2 in NICKNAME_MAP.get(first1, []):
        return True, 90, 'nickname'

    # Initial match (J. SMITH vs JOHN SMITH)
    if len(first1) == 1 and first2.startswith(first1):
        return True, 85, 'initial'
    if len(first2) == 1 and first1.startswith(first2):
        return True, 85, 'initial'

    # Fuzzy first name match
    first_sim = similarity_score(first1, first2)
    if first_sim >= 80 and parts1['last'] == parts2['last']:
        return True, first_sim, 'fuzzy_first'

    # Full name fuzzy match
    full_sim = similarity_score(n1, n2)
    if full_sim >= threshold:
        return True, full_sim, 'fuzzy_full'

    return False, full_sim, None


def find_alias_groups(name_counts, threshold=80, min_freq=2):
    """Find groups of names that are likely aliases."""
    # Filter to names with minimum frequency
    names = [n for n, c in name_counts.items() if c >= min_freq]
    print(f"  Analyzing {len(names)} names (min frequency: {min_freq})")

    # Sort by frequency (descending) - more frequent names become canonical
    names.sort(key=lambda n: name_counts[n], reverse=True)

    # Track which names have been assigned to groups
    assigned = set()
    alias_groups = []

    for i, name1 in enumerate(names):
        if name1 in assigned:
            continue

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(names)}")

        # Start a new group with this name as canonical
        group = {
            'canonical': name1,
            'canonical_freq': name_counts[name1],
            'aliases': [],
            'total_freq': name_counts[name1]
        }

        # Find aliases
        for name2 in names:
            if name2 == name1 or name2 in assigned:
                continue

            match, score, match_type = names_match(name1, name2, threshold)

            if match:
                group['aliases'].append({
                    'name': name2,
                    'frequency': name_counts[name2],
                    'score': score,
                    'match_type': match_type
                })
                group['total_freq'] += name_counts[name2]
                assigned.add(name2)

        if group['aliases']:
            alias_groups.append(group)
            assigned.add(name1)

    # Sort groups by total frequency
    alias_groups.sort(key=lambda g: g['total_freq'], reverse=True)

    return alias_groups


def apply_aliases(file_names, alias_groups):
    """Apply alias resolution to the name data."""
    # Build alias mapping
    alias_map = {}
    for group in alias_groups:
        canonical = group['canonical']
        for alias in group['aliases']:
            alias_map[alias['name']] = canonical

    # Apply mapping
    resolved = {}
    changes = 0

    for file_path, names in file_names.items():
        resolved_names = []
        for name in names:
            if name in alias_map:
                resolved_names.append(alias_map[name])
                changes += 1
            else:
                resolved_names.append(name)
        resolved[file_path] = list(set(resolved_names))

    return resolved, alias_map, changes


def main():
    parser = argparse.ArgumentParser(description="Resolve name aliases")
    parser.add_argument("--threshold", type=int, default=80,
                        help="Similarity threshold (0-100, default: 80)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum name frequency to analyze (default: 2)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply aliases to create resolved_names.json")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top alias groups to display (default: 50)")
    args = parser.parse_args()

    print("=== ALIAS RESOLUTION ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load names
    print("Loading extracted names...")
    file_names, name_counts = load_names()

    if file_names is None:
        return

    print(f"  Total unique names: {len(name_counts)}")
    print(f"  Total name occurrences: {sum(name_counts.values())}")

    # Find alias groups
    print(f"\nFinding aliases (threshold: {args.threshold}%)...")
    alias_groups = find_alias_groups(name_counts, args.threshold, args.min_freq)

    print(f"\nAlias groups found: {len(alias_groups)}")

    # Calculate statistics
    total_aliases = sum(len(g['aliases']) for g in alias_groups)
    names_affected = sum(1 + len(g['aliases']) for g in alias_groups)

    print(f"Total aliases identified: {total_aliases}")
    print(f"Names affected: {names_affected}")

    # Display results
    print("\n" + "="*60)
    print("TOP ALIAS GROUPS")
    print("="*60)

    for i, group in enumerate(alias_groups[:args.top], 1):
        print(f"\n{i}. {group['canonical']} (freq: {group['canonical_freq']})")
        print(f"   Total combined frequency: {group['total_freq']}")
        print(f"   Aliases:")
        for alias in sorted(group['aliases'], key=lambda a: a['frequency'], reverse=True):
            print(f"     - {alias['name']} (freq: {alias['frequency']}, "
                  f"score: {alias['score']:.0f}%, type: {alias['match_type']})")

    # Match type breakdown
    match_types = Counter()
    for group in alias_groups:
        for alias in group['aliases']:
            match_types[alias['match_type']] += 1

    print("\n--- Alias Match Types ---")
    for match_type, count in match_types.most_common():
        print(f"  {match_type}: {count}")

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    # Save alias groups
    with open(OUTPUT_DIR / "alias_groups.json", 'w', encoding='utf-8') as f:
        json.dump(alias_groups, f, indent=2)
    print(f"  Alias groups: {OUTPUT_DIR / 'alias_groups.json'}")

    # Save as CSV for easy review
    alias_rows = []
    for group in alias_groups:
        for alias in group['aliases']:
            alias_rows.append({
                'canonical': group['canonical'],
                'canonical_freq': group['canonical_freq'],
                'alias': alias['name'],
                'alias_freq': alias['frequency'],
                'score': alias['score'],
                'match_type': alias['match_type']
            })

    alias_df = pd.DataFrame(alias_rows)
    alias_df.to_csv(OUTPUT_DIR / "alias_mapping.csv", index=False)
    print(f"  Alias mapping: {OUTPUT_DIR / 'alias_mapping.csv'}")

    # Save canonical names list
    canonical_df = pd.DataFrame([
        {
            'canonical_name': g['canonical'],
            'total_frequency': g['total_freq'],
            'alias_count': len(g['aliases']),
            'aliases': '; '.join(a['name'] for a in g['aliases'])
        }
        for g in alias_groups
    ])
    canonical_df.to_csv(OUTPUT_DIR / "canonical_names.csv", index=False)
    print(f"  Canonical names: {OUTPUT_DIR / 'canonical_names.csv'}")

    # Apply aliases if requested
    if args.apply:
        print("\nApplying aliases...")
        resolved, alias_map, changes = apply_aliases(file_names, alias_groups)

        # Save resolved names
        with open(OUTPUT_DIR / "resolved_names.json", 'w', encoding='utf-8') as f:
            json.dump(resolved, f, indent=2)
        print(f"  Resolved names: {OUTPUT_DIR / 'resolved_names.json'}")

        # Save simple alias map
        with open(OUTPUT_DIR / "alias_map.json", 'w', encoding='utf-8') as f:
            json.dump(alias_map, f, indent=2)
        print(f"  Alias map: {OUTPUT_DIR / 'alias_map.json'}")

        print(f"\n  Applied {changes} alias substitutions")

        # Show impact
        old_unique = len(name_counts)
        new_counts = Counter()
        for names in resolved.values():
            new_counts.update(names)
        new_unique = len(new_counts)

        print(f"  Unique names before: {old_unique}")
        print(f"  Unique names after: {new_unique}")
        print(f"  Reduction: {old_unique - new_unique} ({(old_unique - new_unique)/old_unique*100:.1f}%)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Names analyzed: {len(name_counts)}")
    print(f"Alias groups found: {len(alias_groups)}")
    print(f"Total aliases: {total_aliases}")
    print(f"Similarity threshold: {args.threshold}%")

    if alias_groups:
        largest = alias_groups[0]
        print(f"\nLargest group: {largest['canonical']}")
        print(f"  - {len(largest['aliases'])} aliases")
        print(f"  - Combined frequency: {largest['total_freq']}")

    print(f"\nResults saved to: {OUTPUT_DIR}/")

    if not args.apply:
        print("\nTip: Use --apply to create resolved_names.json with aliases merged")

    print("\nAlias resolution complete!")


if __name__ == "__main__":
    main()
