# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project downloads and analyzes documents from the DOJ Epstein library (Court Records, DOJ Disclosures, FOIA). It builds a co-occurrence network of people mentioned in the documents and provides tools to visualize and explore that network.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Requires `pdftotext` for PDF conversion:
```bash
sudo apt install poppler-utils  # Debian/Ubuntu
```

## Common Commands

```bash
# Download PDFs from DOJ
python download_epstein_pdfs.py          # Interactive mode
python download_epstein_pdfs.py -y       # Skip confirmation
python download_epstein_pdfs.py --download-saved  # Resume from saved URLs

# Convert PDFs to text
cd epstein_pdfs && ../convert_pdfs.sh

# Analyze text file quality (identify garbage/empty files)
python check_txt_files.py ./epstein_pdfs

# Extract person names using spaCy NER (parallel processing)
python extract_names.py

# Extract locations (countries, cities, facilities)
python extract_locations.py

# Extract organizations (companies, institutions, agencies)
python extract_organizations.py

# Extract timeline (dates and chronology)
python extract_timeline.py

# Generate network from extracted names
python generate_network.py

# Run sentiment analysis with VADER
python sentiment_analysis.py

# Run topic modeling with BERTopic
python topic_modeling.py                  # Auto-detect topics
python topic_modeling.py --nr-topics 20   # Specify number of topics

# Run community detection on the network
python community_detection.py             # Detect communities + centrality
python community_detection.py --top 30    # Show top 30 per metric

# Find duplicate/similar documents
python document_similarity.py                    # TF-IDF (fast)
python document_similarity.py --method embedding # Semantic similarity
python document_similarity.py --threshold 0.9    # Stricter matching

# Cross-entity analysis (requires extraction scripts to run first)
python cross_entity_analysis.py                  # Link people-orgs-locations-dates
python cross_entity_analysis.py --min-cooccurrence 5  # Stricter threshold

# Build entity profiles (dossiers for individuals)
python entity_profiles.py                        # Top 50 persons
python entity_profiles.py --top 100              # Top 100 persons
python entity_profiles.py --search "EPSTEIN"     # Search specific person

# Generate ego networks (subgraph around a person)
python ego_networks.py --search "CLINTON"        # Search for matches
python ego_networks.py --person "BILL CLINTON"   # Specific person
python ego_networks.py --top 10                  # Top 10 most connected
python ego_networks.py --person "NAME" --depth 2 # Include 2-hop neighbors

# Extract key phrases (significant n-grams)
python key_phrases.py                            # Extract key phrases
python key_phrases.py --top 200                  # Top 200 phrases
python key_phrases.py --ngram-max 4              # Up to 4-grams
python key_phrases.py --per-doc                  # Also per-document phrases

# Detect redactions in documents
python redaction_detection.py                    # Scan for redactions
python redaction_detection.py --show-samples     # Show redaction samples
python redaction_detection.py --top 50           # Top 50 redacted docs

# Resolve name aliases (merge variants like Bill/William)
python alias_resolution.py                       # Find aliases
python alias_resolution.py --threshold 85        # Adjust similarity
python alias_resolution.py --apply               # Create resolved_names.json

# Extract quotes from depositions/testimony
python quote_attribution.py                      # Extract quotes
python quote_attribution.py --search "EPSTEIN"   # Search specific speaker
python quote_attribution.py --min-length 50      # Longer quotes only

# Word frequency analysis
python word_frequency.py                         # Basic frequency analysis
python word_frequency.py --top 500               # Top 500 words
python word_frequency.py --min-length 4          # Words with 4+ characters
python word_frequency.py --by-folder             # Analyze by subfolder

# Start the interactive network viewer
python serve.py  # Opens browser to http://localhost:8080
```

## Architecture

### Data Pipeline
1. `download_epstein_pdfs.py` - Scrapes and downloads PDFs, saves URLs to `epstein_pdfs/pdf_urls.json`
2. `convert_pdfs.sh` - Batch converts PDFs to text files using pdftotext
3. `check_txt_files.py` - Classifies text files as real_text/empty/garbage
4. `extract_names.py` - Uses spaCy NER with multiprocessing to extract person names → `extracted_names.json`
5. `extract_locations.py` - Uses spaCy NER to extract GPE/LOC/FAC entities → `location_extraction_output/`
6. `extract_organizations.py` - Uses spaCy NER to extract ORG entities → `organization_extraction_output/`
7. `extract_timeline.py` - Extracts dates using NER + regex → `timeline_extraction_output/`
8. `generate_network.py` - Builds co-occurrence network from `extracted_names.json` → CSV and PDF outputs
9. `topic_modeling.py` - BERTopic-based topic discovery → `topic_modeling_output/`
10. `community_detection.py` - Louvain community detection + centrality metrics → `community_detection_output/`
11. `document_similarity.py` - TF-IDF/embedding similarity for duplicate detection → `document_similarity_output/`
12. `cross_entity_analysis.py` - Links people ↔ orgs ↔ locations ↔ dates → `cross_entity_output/`
13. `entity_profiles.py` - Builds comprehensive dossiers for individuals → `entity_profiles_output/`
14. `ego_networks.py` - Generates subgraphs centered on specific individuals → `ego_networks_output/`
15. `key_phrases.py` - Extracts significant n-grams using TF-IDF → `key_phrases_output/`
16. `redaction_detection.py` - Identifies redacted sections in documents → `redaction_detection_output/`
17. `alias_resolution.py` - Merges name variants using fuzzy matching → `alias_resolution_output/`
18. `quote_attribution.py` - Extracts "who said what" from depositions → `quote_attribution_output/`
19. `word_frequency.py` - Analyzes vocabulary patterns and word frequencies → `word_frequency_output/`

### Network Viewer (Web Application)
- `network_viewer.html` - Sigma.js/Graphology-based WebGL visualization
- `layout-worker.js` - Web Worker for background force-directed layout calculation
- `serve.py` - Simple HTTP server (port 8080)

### Key Configuration Variables
- `generate_network.py`: `MIN_APPEARANCES` (default 5), `MIN_EDGE_WEIGHT` (default 3)
- `extract_names.py`: `BASE_DIR` (default `./epstein_pdfs`)
- `extract_locations.py`: `BASE_DIR` (default `./epstein_pdfs`)
- `extract_organizations.py`: `BASE_DIR`, `MIN_ORG_MENTIONS` (default 3), `MIN_COOCCURRENCE` (default 2)
- `extract_timeline.py`: `BASE_DIR`, `MIN_YEAR` (default 1970), `MAX_YEAR` (default 2025)
- `sentiment_analysis.py`: `BASE_DIR` (default `./epstein_pdfs`)
- `topic_modeling.py`: `--nr-topics` (auto), `--min-topic-size` (default 3)
- `community_detection.py`: `--min-weight` (default 3), `--top` (default 20)
- `document_similarity.py`: `--method` (tfidf/embedding), `--threshold` (default 0.8)
- `cross_entity_analysis.py`: `--min-freq` (default 3), `--min-cooccurrence` (default 2)
- `entity_profiles.py`: `--top` (default 50), `--min-mentions` (default 5), `--search`
- `ego_networks.py`: `--person`, `--search`, `--top`, `--depth` (1 or 2), `--min-weight` (default 2)
- `key_phrases.py`: `--top` (default 100), `--ngram-min` (default 2), `--ngram-max` (default 3), `--per-doc`
- `redaction_detection.py`: `--top` (default 30), `--show-samples`, `--min-redactions` (default 1)
- `alias_resolution.py`: `--threshold` (default 80), `--min-freq` (default 2), `--apply`
- `quote_attribution.py`: `--min-length` (default 30), `--search`, `--top` (default 50)
- `word_frequency.py`: `--top` (default 200), `--min-length` (default 3), `--by-folder`

### Output Files
- `extracted_names.json` - NER results: `{file_path: [names]}`
- `network_nodes_spacy.csv` - Node list with appearance counts
- `network_edges_spacy.csv` - Edge list with co-occurrence weights
- `network_map_spacy.pdf` / `network_map_top100.pdf` - Static network visualizations
- `topic_modeling_output/` - Topic modeling results:
  - `topics.csv` - Topic info with document counts
  - `document_topics.json` - Document-to-topic assignments
  - `topic_words.json` - Keywords for each topic
  - `topic_hierarchy.html` / `topic_barchart.html` / `topic_distance_map.html` - Visualizations
- `community_detection_output/` - Community detection results:
  - `node_communities.csv` - Node assignments with centrality metrics
  - `community_summaries.csv` - Summary stats per community
  - `community_members.json` - Full member lists per community
  - `community_network.pdf` - Color-coded community visualization
- `location_extraction_output/` - Location extraction results:
  - `extracted_locations.json` - Raw location extractions per file
  - `gpe_frequencies.csv` / `loc_frequencies.csv` / `fac_frequencies.csv` - Frequency tables by type
  - `all_locations.csv` - Combined location frequencies
  - `top_gpe_chart.pdf` / `location_types_pie.pdf` - Visualizations
- `organization_extraction_output/` - Organization extraction results:
  - `extracted_organizations.json` - Raw organization extractions per file
  - `organization_frequencies.csv` - Frequencies with category labels
  - `category_summary.csv` - Mentions aggregated by category
  - `network_nodes.csv` / `network_edges.csv` - Organization co-occurrence network
  - `top_organizations_chart.pdf` / `organization_network.pdf` - Visualizations
- `timeline_extraction_output/` - Timeline extraction results:
  - `extracted_dates.json` - Raw date extractions with context per file
  - `all_dates.csv` - All parsed dates with file references
  - `year_counts.csv` / `month_counts.csv` - Aggregated counts
  - `timeline_by_year.pdf` / `monthly_heatmap.pdf` / `cumulative_timeline.pdf` - Visualizations
- `document_similarity_output/` - Document similarity results:
  - `similar_pairs.csv` - Document pairs with similarity scores
  - `document_clusters.json` - Groups of duplicate/similar documents
  - `cluster_summary.csv` - Cluster sizes and sample files
  - `similarity_distribution.pdf` / `cluster_sizes.pdf` - Visualizations
- `cross_entity_output/` - Cross-entity analysis results:
  - `person_organization_links.csv` - Person-org co-occurrences
  - `person_location_links.csv` - Person-location co-occurrences
  - `person_year_links.csv` - Person activity by year
  - `org_location_links.csv` / `org_year_links.csv` - Org relationships
  - `graph_nodes.csv` / `graph_edges.csv` - Multi-entity network
  - `cross_entity_network.pdf` / `person_year_heatmap.pdf` - Visualizations
- `entity_profiles_output/` - Entity profile results:
  - `all_profiles.json` - Complete profile data for all persons
  - `profile_summary.csv` - Summary table with key stats per person
  - `associate_network.csv` - Who appears with whom
  - `individual_profiles/` - Text files for top 20 persons
- `ego_networks_output/` - Ego network results:
  - `ego_[NAME].pdf` - Visualization of person's network
  - `ego_[NAME]_nodes.csv` / `ego_[NAME]_edges.csv` - Network data
  - `ego_summary.csv` - Comparison stats for multiple persons
  - `ego_stats.json` - Detailed statistics
- `key_phrases_output/` - Key phrase extraction results:
  - `key_phrases.csv` - All extracted phrases with scores
  - `phrases_[category].csv` - Phrases grouped by category
  - `document_phrases.json` - Per-document key phrases (with --per-doc)
  - `top_phrases_chart.pdf` / `category_distribution.pdf` - Visualizations
- `redaction_detection_output/` - Redaction detection results:
  - `redacted_documents.csv` - Documents with redaction counts by pattern
  - `pattern_counts.csv` - Total counts per redaction pattern
  - `severity_breakdown.csv` - Documents grouped by redaction severity
  - `detailed_redactions.json` - Full redaction details with samples
  - `pattern_distribution.pdf` / `severity_pie_chart.pdf` / `top_redacted_docs.pdf` - Visualizations
- `alias_resolution_output/` - Alias resolution results:
  - `alias_groups.json` - Groups of names with their aliases
  - `alias_mapping.csv` - Alias → canonical name mappings
  - `canonical_names.csv` - Canonical names with alias counts
  - `resolved_names.json` - Names with aliases merged (with --apply)
  - `alias_map.json` - Simple alias lookup dictionary
- `quote_attribution_output/` - Quote attribution results:
  - `all_quotes.csv` - All extracted quotes with speaker and source
  - `speaker_summary.csv` - Speakers ranked by quote count
  - `quotes_by_speaker.json` - Quotes grouped by speaker
  - `quote_types.csv` - Breakdown by extraction method
  - `matched_speakers.csv` - Speakers matched to known names
- `word_frequency_output/` - Word frequency analysis results:
  - `word_frequencies.csv` - Word frequencies with categories
  - `distinctive_words.csv` - Words with high TF-IDF scores
  - `category_summary.csv` - Word counts by topic category
  - `wordcloud_data.json` - Data for word cloud visualization
  - `folder_frequencies.json` - Frequencies by subfolder (with --by-folder)
  - `top_words_chart.pdf` / `category_pie_chart.pdf` / `zipf_distribution.pdf` - Visualizations
