import pandas as pd
from youtubesearchpython import VideosSearch
from textblob import TextBlob
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Any
import numpy as np

class BrandMentions:
    """Track brand mentions, engagement, and sentiment across videos."""
    
    def __init__(self):
        # List of brands to track (can be expanded)
        self.brands = ['atomberg', 'havells', 'crompton', 'bajaj', 'orient', 'usha']
        self.mentions = {brand: 0 for brand in self.brands}
        self.engagement = {brand: {'likes': 0, 'comments': 0, 'views': 0} for brand in self.brands}
        self.sentiments = {brand: [] for brand in self.brands}
        self.positive_mentions = {brand: 0 for brand in self.brands}
        self.total_mentions = {brand: 0 for brand in self.brands}

class SOVAnalyzer:
    """Analyze Share of Voice for brands across YouTube search results."""
    
    def __init__(self, queries: List[str], max_results: int = 20):
        """
        Initialize the SOV analyzer.
        
        Args:
            queries: List of search queries to analyze
            max_results: Maximum number of results per query
        """
        self.queries = queries
        self.max_results = max_results
        self.brand_data = BrandMentions()
        self.results = []
    
    def search_youtube(self, query: str) -> List[Dict[str, Any]]:
        """Search YouTube for the given query and return results."""
        videos_search = VideosSearch(query, limit=self.max_results)
        results = videos_search.result()
        return results['result']
    
    def analyze_mentions(self, text: str) -> List[str]:
        """Analyze text for brand mentions."""
        text_lower = text.lower()
        found_brands = []
        for brand in self.brand_data.brands:
            if re.search(rf'\b{brand}\b', text_lower):
                found_brands.append(brand)
        return found_brands
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of the given text."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Returns polarity between -1 and 1
    
    def process_video(self, video: Dict[str, Any]) -> None:
        """Process a single video's data."""
        try:
            title = video.get('title', '')
            description = video.get('description', '')
            video_text = f"{title}. {description}"
            
            # Skip if no relevant text
            if not video_text.strip():
                return
        
            # Get engagement metrics with proper error handling
            try:
                view_text = video.get('viewCount', {}).get('text', '0')
                views = int(''.join(filter(str.isdigit, view_text.split()[0])))
            except (AttributeError, ValueError, IndexError):
                views = 0
                
            try:
                likes = int(video.get('likes', 0)) if video.get('likes') else 0
            except (TypeError, ValueError):
                likes = 0
                
            try:
                comments = int(video.get('commentCount', 0)) if video.get('commentCount') else 0
            except (TypeError, ValueError):
                comments = 0
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(video_text)
            
            # Check for brand mentions
            mentioned_brands = set()
            mentioned_brands.update(self.analyze_mentions(title))
            mentioned_brands.update(self.analyze_mentions(description))
            
            # Update metrics for mentioned brands
            for brand in mentioned_brands:
                self.brand_data.mentions[brand] += 1
                self.brand_data.engagement[brand]['views'] += views
                self.brand_data.engagement[brand]['likes'] += likes
                self.brand_data.engagement[brand]['comments'] += comments
                self.brand_data.sentiments[brand].append(sentiment)
                self.brand_data.total_mentions[brand] += 1
                if sentiment > 0.1:  # Considered positive
                    self.brand_data.positive_mentions[brand] += 1
                    
        except Exception as e:
            print(f"Error processing video: {str(e)}")
    
    def process_queries(self) -> None:
        """Process all search queries."""
        for query in self.queries:
            try:
                print(f"\nProcessing query: {query}")
                results = self.search_youtube(query)
                print(f"Found {len(results)} videos")
                for i, video in enumerate(results, 1):
                    print(f"  - Processing video {i}/{len(results)}", end='\r')
                    self.process_video(video)
                print()  # New line after processing all videos for this query
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                continue
    
    def calculate_sov_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate Share of Voice metrics."""
        total_mentions = sum(self.brand_data.mentions.values()) or 1
        total_engagement = {
            'views': sum(brand_data['views'] for brand_data in self.brand_data.engagement.values()),
            'likes': sum(brand_data['likes'] for brand_data in self.brand_data.engagement.values()),
            'comments': sum(brand_data['comments'] for brand_data in self.brand_data.engagement.values())
        }
        
        # Normalize engagement metrics
        max_views = max((data['views'] for data in self.brand_data.engagement.values()), default=1)
        max_likes = max((data['likes'] for data in self.brand_data.engagement.values()), default=1)
        max_comments = max((data['comments'] for data in self.brand_data.engagement.values()), default=1)
        
        sov_metrics = {}
        
        for brand in self.brand_data.brands:
            # Calculate mention share
            mention_share = (self.brand_data.mentions[brand] / total_mentions) * 100 if total_mentions > 0 else 0
            
            # Calculate engagement score (normalized to 0-100)
            try:
                views_ratio = (self.brand_data.engagement[brand]['views'] / max_views) if max_views > 0 else 0
                likes_ratio = (self.brand_data.engagement[brand]['likes'] / max_likes) if max_likes > 0 else 0
                comments_ratio = (self.brand_data.engagement[brand]['comments'] / max_comments) if max_comments > 0 else 0
                
                engagement_score = (
                    0.4 * views_ratio +
                    0.3 * likes_ratio +
                    0.3 * comments_ratio
                ) * 100
            except (ZeroDivisionError, KeyError):
                engagement_score = 0
            
            # Calculate sentiment metrics
            sentiments = self.brand_data.sentiments[brand]
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            positive_share = (self.brand_data.positive_mentions[brand] / 
                            self.brand_data.total_mentions[brand] * 100) if self.brand_data.total_mentions[brand] > 0 else 0
            
            # Calculate final SoV score (weighted average)
            sov_score = (
                0.4 * mention_share +
                0.4 * engagement_score +
                0.2 * positive_share
            )
            
            sov_metrics[brand] = {
                'mention_share': mention_share,
                'engagement_score': engagement_score,
                'avg_sentiment': avg_sentiment,
                'positive_share': positive_share,
                'sov_score': sov_score,
                'total_mentions': self.brand_data.mentions[brand],
                'total_engagement': sum(self.brand_data.engagement[brand].values())
            }
        
        return sov_metrics
    
    def generate_recommendations(self, sov_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate marketing recommendations based on SoV analysis."""
        recommendations = []
        atomberg_metrics = sov_metrics.get('atomberg', {})
        
        if not atomberg_metrics:
            return ["No data available for Atomberg."]
        
        # Find top competitor
        competitors = {k: v for k, v in sov_metrics.items() if k != 'atomberg' and v['sov_score'] > 0}
        top_competitor = max(competitors.items(), key=lambda x: x[1]['sov_score'], default=(None, None))
        
        # Generate recommendations based on metrics
        if atomberg_metrics['mention_share'] < 30:
            recommendations.append(
                "Increase brand mentions in video content and descriptions. Consider "
                "sponsoring tech reviewers or creating more branded content."
            )
        
        if atomberg_metrics['engagement_score'] < 50:
            recommendations.append(
                "Focus on creating more engaging content. Consider tutorials, "
                "comparisons, or behind-the-scenes content to boost engagement."
            )
        
        if top_competitor[0] and top_competitor[1]['sov_score'] > atomberg_metrics['sov_score']:
            recommendations.append(
                f"{top_competitor[0].title()} is leading in SoV. Analyze their "
                "content strategy and consider similar approaches while maintaining "
                "Atomberg's unique value propositions."
            )
        
        # Sentiment-based recommendations
        if atomberg_metrics['positive_share'] < 60:
            recommendations.append(
                "Monitor and address any negative sentiment. Consider engaging with "
                "customers in comments to improve brand perception."
            )
        
        return recommendations or ["Continue current strategy as Atomberg is performing well in the market."]
    
    def export_to_csv(self, sov_metrics: Dict[str, Dict[str, float]], filename: str = "sov_analysis.csv") -> None:
        """Export analysis results to a CSV file."""
        rows = []
        for brand, metrics in sov_metrics.items():
            row = {'brand': brand}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('sov_score', ascending=False)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

def main():
    # Define search queries
    queries = [
        "smart fan review",
        "best smart fan India",
        "BLDC ceiling fan",
        "energy saving fan",
        "smart fan comparison"
    ]
    
    # Initialize and run analysis
    print("Starting Share of Voice Analysis...")
    analyzer = SOVAnalyzer(queries, max_results=20)
    analyzer.process_queries()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    sov_metrics = analyzer.calculate_sov_metrics()
    
    # Export results
    analyzer.export_to_csv(sov_metrics, "atomberg_sov_analysis.csv")
    
    # Print summary
    print("\nShare of Voice Analysis Results:")
    print("=" * 80)
    print(f"{'Brand':<10} {'SoV Score':<12} {'Mentions':<10} {'Engagement':<12} {'Sentiment':<10} {'+ve Share':<10}")
    print("-" * 80)
    
    for brand, metrics in sorted(sov_metrics.items(), key=lambda x: x[1]['sov_score'], reverse=True):
        if metrics['sov_score'] > 0:  # Only show brands with some presence
            print(f"{brand:<10} {metrics['sov_score']:>7.1f}     "
                  f"{metrics['total_mentions']:>4}       "
                  f"{metrics['engagement_score']:>6.1f}     "
                  f"{metrics['avg_sentiment']:>+5.2f}     "
                  f"{metrics['positive_share']:>5.1f}%")
    
    # Generate and print recommendations
    print("\nMarketing Recommendations:")
    print("=" * 80)
    for i, rec in enumerate(analyzer.generate_recommendations(sov_metrics), 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
