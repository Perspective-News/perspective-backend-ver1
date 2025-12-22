# Perspective News Data Fetching and Storing

This repository contains an end-to-end news ingestion pipeline for Perspective News. It fetches articles from a list of RSS feeds and URLs, cleans and normalizes the data, clusters articles into events based on semantic similarity, summarizes each event, and produces a JSON file describing the events.

## Files

- `main.py` – Python script that implements the pipeline.
- `sources.csv` – CSV file listing news sources (South Asian balanced dataset).
- `requirements.txt` – Python dependencies required to run the pipeline.

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:

   ```bash
   python main.py
   ```

   By default, the script reads `sources.csv` in the current directory and writes output to `output_events.json`. You can specify a different sources file or output path with command-line arguments:

   ```bash
   python main.py --sources_path path/to/sources.csv --output_path path/to/output.json
   ```

The pipeline will fetch up to 50 articles per source (configurable via `max_items_per_source` argument), clean and cluster them, and generate a summary for each event. The output JSON contains metadata about each event and the list of articles grouped in it.

## Extending sources

The `sources.csv` file should have the following columns:

- `source_name` – Name of the news source.
- `source_type` – Either `rss` or `url_list`.
- `source_url` – RSS feed URL or a URL pointing to a list of article links.
- `country` – (optional) Country or region of the source.

Add new sources by appending rows to this CSV file. Ensure that each source has a valid feed or list URL.

## License

This project is provided as-is for research and educational purposes.
