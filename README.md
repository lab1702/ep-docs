# Epstein Documents Analysis

Tools for downloading and analyzing documents from the DOJ Epstein library.

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

## Scripts

### Data Collection & Preparation

#### download_epstein_pdfs.py

Downloads all PDFs from the DOJ Epstein library (Court Records, DOJ Disclosures, FOIA).

```bash
python download_epstein_pdfs.py          # Interactive mode
python download_epstein_pdfs.py -y       # Skip confirmation
python download_epstein_pdfs.py --download-saved  # Resume from saved URLs
```

#### convert_pdfs.sh

Converts all PDF files to text using `pdftotext`. Text files are created alongside the original PDFs.

```bash
cd epstein_pdfs
../convert_pdfs.sh
```

#### check_txt_files.py

Analyzes .txt files and classifies them as real text, empty, or garbage.

```bash
python check_txt_files.py ./epstein_pdfs
```

---

### Entity Extraction

#### extract_names.py

Extracts person names from text files using spaCy NER with parallel processing.

```bash
python extract_names.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `./epstein_pdfs` | Directory containing .txt files |

#### extract_locations.py

Extracts geographic entities (countries, cities, facilities) using spaCy NER.

```bash
python extract_locations.py
```

**Output:** `location_extraction_output/`

#### extract_organizations.py

Extracts organization names (companies, institutions, agencies) with categorization.

```bash
python extract_organizations.py
```

**Output:** `organization_extraction_output/`

#### extract_timeline.py

Extracts dates and builds chronological timelines from documents.

```bash
python extract_timeline.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_YEAR` | `1970` | Minimum year to include |
| `MAX_YEAR` | `2025` | Maximum year to include |

**Output:** `timeline_extraction_output/`

---

### Network Analysis

#### generate_network.py

Builds a co-occurrence network from extracted names and generates visualizations.

```bash
python extract_names.py      # Run first to extract names
python generate_network.py   # Then generate network
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_APPEARANCES` | `5` | Minimum file appearances to include a person |
| `MIN_EDGE_WEIGHT` | `3` | Minimum co-occurrences to show an edge |

#### community_detection.py

Detects communities in the network using Louvain algorithm and calculates centrality metrics.

```bash
python community_detection.py             # Detect communities
python community_detection.py --top 30    # Show top 30 per metric
```

**Output:** `community_detection_output/`

#### ego_networks.py

Generates subgraphs centered on specific individuals.

```bash
python ego_networks.py --search "CLINTON"        # Search for matches
python ego_networks.py --person "BILL CLINTON"   # Specific person
python ego_networks.py --top 10                  # Top 10 most connected
python ego_networks.py --person "NAME" --depth 2 # Include 2-hop neighbors
```

**Output:** `ego_networks_output/`

---

### Text Analysis

#### sentiment_analysis.py

Performs sentiment analysis on text files using NLTK VADER.

```bash
python sentiment_analysis.py
```

#### topic_modeling.py

Discovers topics using BERTopic with sentence embeddings.

```bash
python topic_modeling.py                  # Auto-detect topics
python topic_modeling.py --nr-topics 20   # Specify number of topics
```

**Output:** `topic_modeling_output/`

#### key_phrases.py

Extracts significant n-grams and phrases using TF-IDF.

```bash
python key_phrases.py                     # Extract key phrases
python key_phrases.py --top 200           # Top 200 phrases
python key_phrases.py --ngram-max 4       # Up to 4-grams
python key_phrases.py --per-doc           # Also per-document phrases
```

**Output:** `key_phrases_output/`

#### word_frequency.py

Analyzes vocabulary patterns and word frequencies across the corpus.

```bash
python word_frequency.py                  # Basic frequency analysis
python word_frequency.py --top 500        # Top 500 words
python word_frequency.py --min-length 4   # Words with 4+ characters
python word_frequency.py --by-folder      # Analyze by subfolder
```

**Output:** `word_frequency_output/`

#### quote_attribution.py

Extracts quotes and testimony from depositions with speaker attribution.

```bash
python quote_attribution.py                      # Extract quotes
python quote_attribution.py --search "EPSTEIN"   # Search specific speaker
python quote_attribution.py --min-length 50      # Longer quotes only
```

**Output:** `quote_attribution_output/`

---

### Document Analysis

#### document_similarity.py

Finds duplicate and similar documents using TF-IDF or semantic embeddings.

```bash
python document_similarity.py                    # TF-IDF (fast)
python document_similarity.py --method embedding # Semantic similarity
python document_similarity.py --threshold 0.9    # Stricter matching
```

**Output:** `document_similarity_output/`

#### redaction_detection.py

Identifies redacted sections in documents (black boxes, [REDACTED] markers, etc.).

```bash
python redaction_detection.py                    # Scan for redactions
python redaction_detection.py --show-samples     # Show redaction samples
python redaction_detection.py --top 50           # Top 50 redacted docs
```

**Output:** `redaction_detection_output/`

---

### Cross-Reference & Profiling

#### cross_entity_analysis.py

Links people, organizations, locations, and dates across documents.

```bash
python cross_entity_analysis.py                  # Link entities
python cross_entity_analysis.py --min-cooccurrence 5  # Stricter threshold
```

**Output:** `cross_entity_output/`

#### entity_profiles.py

Builds comprehensive dossiers for individuals aggregating all available information.

```bash
python entity_profiles.py                        # Top 50 persons
python entity_profiles.py --top 100              # Top 100 persons
python entity_profiles.py --search "EPSTEIN"     # Search specific person
```

**Output:** `entity_profiles_output/`

#### alias_resolution.py

Merges name variants (nicknames, misspellings) using fuzzy string matching.

```bash
python alias_resolution.py                       # Find aliases
python alias_resolution.py --threshold 85        # Adjust similarity
python alias_resolution.py --apply               # Create resolved_names.json
```

**Output:** `alias_resolution_output/`

---

### Network Viewer (Web Application)

Interactive web-based visualization for exploring the network graph. Handles large networks (400K+ edges) with WebGL rendering.

```bash
python serve.py
```

This starts a local server on port 8080 and opens the viewer in your browser.

**Features:**
- Zoom in/out (scroll) and pan (drag) to explore the network
- Filter by minimum appearances and edge weight using sliders
- Search to highlight matching nodes (yellow) without hiding others
- Click nodes to see their connections and edge weights
- Force-directed layout with progress indicator (runs in background)
- Adaptive label display based on zoom level

**Controls:**
| Control | Description |
|---------|-------------|
| Scroll | Zoom in/out |
| Drag | Pan the view |
| Click node | Show node details and connections |
| Search box | Highlight matching nodes |
| Min Appearances | Filter nodes by appearance count |
| Min Edge Weight | Filter edges by co-occurrence weight |
| Run Layout | Reorganize graph using force-directed algorithm |

**Files:**
- `network_viewer.html` - Main viewer application
- `layout-worker.js` - Web worker for background layout calculation
- `serve.py` - Simple HTTP server to run the viewer

---

## Output Files

### Core Data
| File | Description |
|------|-------------|
| `epstein_pdfs/` | Downloaded PDF files |
| `epstein_pdfs/pdf_urls.json` | Scraped PDF URLs (JSON format) |
| `epstein_pdfs/pdf_urls.csv` | Scraped PDF URLs with local paths (CSV format) |
| `txt_analysis_results.csv` | File quality report |
| `extracted_names.json` | Extracted person names from NER |

### Network Files
| File | Description |
|------|-------------|
| `network_nodes_spacy.csv` | Network node list with appearance counts |
| `network_edges_spacy.csv` | Network edge list with co-occurrence weights |
| `network_map_spacy.pdf` | Full network visualization |
| `network_map_top100.pdf` | Top 100 individuals network |

### Analysis Output Directories
| Directory | Script | Description |
|-----------|--------|-------------|
| `topic_modeling_output/` | topic_modeling.py | Topics, keywords, visualizations |
| `community_detection_output/` | community_detection.py | Communities, centrality metrics |
| `location_extraction_output/` | extract_locations.py | Geographic entities by type |
| `organization_extraction_output/` | extract_organizations.py | Organizations with categories |
| `timeline_extraction_output/` | extract_timeline.py | Dates and chronology |
| `document_similarity_output/` | document_similarity.py | Similar/duplicate documents |
| `cross_entity_output/` | cross_entity_analysis.py | Entity relationships |
| `entity_profiles_output/` | entity_profiles.py | Individual dossiers |
| `ego_networks_output/` | ego_networks.py | Per-person network subgraphs |
| `key_phrases_output/` | key_phrases.py | Significant phrases |
| `word_frequency_output/` | word_frequency.py | Vocabulary analysis |
| `redaction_detection_output/` | redaction_detection.py | Redacted sections |
| `alias_resolution_output/` | alias_resolution.py | Name variant mappings |
| `quote_attribution_output/` | quote_attribution.py | Attributed quotes |

---

## Recommended Workflow

1. **Download documents:**
   ```bash
   python download_epstein_pdfs.py -y
   ```

2. **Convert to text:**
   ```bash
   cd epstein_pdfs && ../convert_pdfs.sh
   ```

3. **Extract entities (run in parallel):**
   ```bash
   python extract_names.py
   python extract_locations.py
   python extract_organizations.py
   python extract_timeline.py
   ```

4. **Build network:**
   ```bash
   python generate_network.py
   python community_detection.py
   ```

5. **Run analyses:**
   ```bash
   python topic_modeling.py
   python document_similarity.py
   python cross_entity_analysis.py
   python entity_profiles.py
   ```

6. **Explore interactively:**
   ```bash
   python serve.py
   ```
