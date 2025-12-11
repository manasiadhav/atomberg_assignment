# YouTube Share of Voice (SoV) Analyzer

A Python-based tool to analyze brand mentions and engagement across YouTube search results. This tool helps track Share of Voice (SoV) for brands in the smart fan market, providing insights into brand presence, engagement metrics, and sentiment analysis.

## Features

- **Brand Mention Tracking**: Track mentions of multiple brands across YouTube videos
- **Engagement Analysis**: Measure views, likes, and comments for each brand
- **Sentiment Analysis**: Analyze sentiment of video titles and descriptions
- **Share of Voice Metrics**: Calculate comprehensive SoV scores
- **Multi-Query Support**: Analyze multiple search queries simultaneously
- **CSV Export**: Export results for further analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-sov-analyzer.git
   cd youtube-sov-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the analyzer:
   ```bash
   python sov_analyzer.py
   ```

2. The script will:
   - Search YouTube for the specified queries
   - Analyze brand mentions and engagement
   - Generate a report in the console
   - Save results to `atomberg_sov_analysis.csv`

## Configuration

You can modify the following in the `main()` function:
- Search queries
- Number of results per query
- Output file name

## Output

The script provides:
- Console output with detailed metrics
- CSV export with raw data
- Marketing recommendations based on the analysis

## Dependencies

- Python 3.7+
- youtube-search-python
- textblob
- pandas
- numpy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
