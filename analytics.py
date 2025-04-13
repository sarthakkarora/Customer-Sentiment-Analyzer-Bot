import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import prophet
from elasticsearch import AsyncElasticsearch
import json
import os
from loguru import logger

class CustomerAnalytics:
    def __init__(self):
        self.es = AsyncElasticsearch(
            [os.getenv("ELASTICSEARCH_HOST", "localhost:9200")],
            http_auth=(os.getenv("ELASTICSEARCH_USER"), os.getenv("ELASTICSEARCH_PASSWORD"))
        )
        self.scaler = StandardScaler()
        
    async def get_customer_segments(self, time_range: str = "30d") -> Dict[str, Any]:
        """Analyze customer segments using clustering."""
        # Get customer data from Elasticsearch
        response = await self.es.search(
            index="customer-metrics",
            body={
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{time_range}/d"
                        }
                    }
                },
                "size": 10000
            }
        )
        
        # Prepare data for clustering
        data = []
        for hit in response['hits']['hits']:
            data.append({
                'customer_id': hit['_source']['customer_id'],
                'sentiment_score': hit['_source']['sentiment_score'],
                'churn_risk': hit['_source']['churn_risk'],
                'interaction_count': hit['_source'].get('interaction_count', 0)
            })
            
        df = pd.DataFrame(data)
        
        # Normalize features
        features = ['sentiment_score', 'churn_risk', 'interaction_count']
        X = self.scaler.fit_transform(df[features])
        
        # Apply clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['segment'] = kmeans.fit_predict(X)
        
        # Analyze segments
        segments = {}
        for segment in df['segment'].unique():
            segment_data = df[df['segment'] == segment]
            segments[f"segment_{segment}"] = {
                'size': len(segment_data),
                'avg_sentiment': segment_data['sentiment_score'].mean(),
                'avg_churn_risk': segment_data['churn_risk'].mean(),
                'avg_interactions': segment_data['interaction_count'].mean()
            }
            
        return segments
        
    async def analyze_trends(self, metric: str, time_range: str = "90d") -> Dict[str, Any]:
        """Analyze trends in customer metrics."""
        # Get time series data
        response = await self.es.search(
            index="customer-metrics",
            body={
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{time_range}/d"
                        }
                    }
                },
                "aggs": {
                    "daily_metrics": {
                        "date_histogram": {
                            "field": "timestamp",
                            "calendar_interval": "day"
                        },
                        "aggs": {
                            "avg_metric": {"avg": {"field": metric}}
                        }
                    }
                }
            }
        )
        
        # Prepare data for analysis
        dates = []
        values = []
        for bucket in response['aggregations']['daily_metrics']['buckets']:
            dates.append(datetime.fromisoformat(bucket['key_as_string']))
            values.append(bucket['avg_metric']['value'])
            
        # Create time series
        ts = pd.Series(values, index=dates)
        
        # Decompose time series
        decomposition = seasonal_decompose(ts, period=7)
        
        # Check for stationarity
        adf_result = adfuller(ts.dropna())
        
        # Forecast using Prophet
        prophet_df = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        model = prophet.Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        return {
            'trend': decomposition.trend.tolist(),
            'seasonal': decomposition.seasonal.tolist(),
            'residual': decomposition.resid.tolist(),
            'stationary': adf_result[1] < 0.05,
            'forecast': {
                'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'yhat': forecast['yhat'].tolist(),
                'yhat_lower': forecast['yhat_lower'].tolist(),
                'yhat_upper': forecast['yhat_upper'].tolist()
            }
        }
        
    async def analyze_correlations(self, time_range: str = "30d") -> Dict[str, float]:
        """Analyze correlations between different metrics."""
        # Get metrics data
        response = await self.es.search(
            index="customer-metrics",
            body={
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{time_range}/d"
                        }
                    }
                },
                "size": 10000
            }
        )
        
        # Prepare data
        data = []
        for hit in response['hits']['hits']:
            data.append(hit['_source'])
            
        df = pd.DataFrame(data)
        
        # Calculate correlations
        metrics = ['sentiment_score', 'churn_risk', 'response_time', 'quality_score']
        correlations = {}
        
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                metric1 = metrics[i]
                metric2 = metrics[j]
                pearson_corr, _ = pearsonr(df[metric1], df[metric2])
                spearman_corr, _ = spearmanr(df[metric1], df[metric2])
                correlations[f"{metric1}_{metric2}"] = {
                    'pearson': float(pearson_corr),
                    'spearman': float(spearman_corr)
                }
                
        return correlations
        
    async def generate_insights(self, time_range: str = "30d") -> Dict[str, Any]:
        """Generate comprehensive insights from customer data."""
        # Get all relevant data
        segments = await self.get_customer_segments(time_range)
        trends = await self.analyze_trends('sentiment_score', time_range)
        correlations = await self.analyze_correlations(time_range)
        
        # Generate insights
        insights = {
            'customer_segments': segments,
            'sentiment_trends': trends,
            'metric_correlations': correlations,
            'key_findings': []
        }
        
        # Analyze segments
        for segment, data in segments.items():
            if data['avg_churn_risk'] > 0.7:
                insights['key_findings'].append({
                    'type': 'high_risk_segment',
                    'segment': segment,
                    'size': data['size'],
                    'risk_level': data['avg_churn_risk']
                })
                
        # Analyze trends
        if not trends['stationary']:
            insights['key_findings'].append({
                'type': 'non_stationary_trend',
                'metric': 'sentiment_score',
                'implication': 'Trend shows significant changes over time'
            })
            
        # Analyze correlations
        for metric_pair, corr in correlations.items():
            if abs(corr['pearson']) > 0.7:
                insights['key_findings'].append({
                    'type': 'strong_correlation',
                    'metrics': metric_pair,
                    'correlation': corr['pearson']
                })
                
        return insights
        
    def create_visualizations(self, insights: Dict[str, Any]) -> Dict[str, str]:
        """Create visualizations for insights."""
        visualizations = {}
        
        # Create segment visualization
        fig_segments = make_subplots(rows=1, cols=3)
        
        for segment, data in insights['customer_segments'].items():
            fig_segments.add_trace(
                go.Scatter(
                    x=[data['avg_sentiment']],
                    y=[data['avg_churn_risk']],
                    size=[data['size']],
                    name=segment,
                    mode='markers'
                ),
                row=1, col=1
            )
            
        fig_segments.update_layout(title="Customer Segments")
        visualizations['segments'] = fig_segments.to_json()
        
        # Create trend visualization
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=insights['sentiment_trends']['forecast']['dates'],
            y=insights['sentiment_trends']['forecast']['yhat'],
            name='Forecast'
        ))
        fig_trends.add_trace(go.Scatter(
            x=insights['sentiment_trends']['forecast']['dates'],
            y=insights['sentiment_trends']['forecast']['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        fig_trends.add_trace(go.Scatter(
            x=insights['sentiment_trends']['forecast']['dates'],
            y=insights['sentiment_trends']['forecast']['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
        fig_trends.update_layout(title="Sentiment Trend Forecast")
        visualizations['trends'] = fig_trends.to_json()
        
        # Create correlation heatmap
        metrics = ['sentiment_score', 'churn_risk', 'response_time', 'quality_score']
        corr_matrix = np.zeros((len(metrics), len(metrics)))
        
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                if i < j:
                    key = f"{metrics[i]}_{metrics[j]}"
                    corr_matrix[i,j] = insights['metric_correlations'][key]['pearson']
                    corr_matrix[j,i] = corr_matrix[i,j]
                elif i == j:
                    corr_matrix[i,j] = 1.0
                    
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=metrics,
            y=metrics,
            colorscale='RdBu'
        ))
        fig_corr.update_layout(title="Metric Correlations")
        visualizations['correlations'] = fig_corr.to_json()
        
        return visualizations 