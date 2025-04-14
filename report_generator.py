import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdfkit
from jinja2 import Environment, FileSystemLoader
import os
from loguru import logger
from analytics import CustomerAnalytics

class ReportGenerator:
    def __init__(self):
        self.analytics = CustomerAnalytics()
        self.template_env = Environment(
            loader=FileSystemLoader('templates')
        )
        
    async def generate_daily_report(self) -> str:
        """Generate daily customer experience report."""
        # Get insights for the last 24 hours
        insights = await self.analytics.generate_insights("1d")
        visualizations = self.analytics.create_visualizations(insights)
        
        # Prepare report data
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_customers': sum(segment['size'] for segment in insights['customer_segments'].values()),
            'high_risk_customers': sum(1 for finding in insights['key_findings'] 
                                     if finding['type'] == 'high_risk_segment'),
            'segments': insights['customer_segments'],
            'trends': insights['sentiment_trends'],
            'key_findings': insights['key_findings'],
            'visualizations': visualizations
        }
        
        # Render HTML template
        template = self.template_env.get_template('daily_report.html')
        html_content = template.render(**report_data)
        
        # Generate PDF
        pdf_path = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        os.makedirs('reports', exist_ok=True)
        pdfkit.from_string(html_content, pdf_path)
        
        return pdf_path
        
    async def generate_weekly_report(self) -> str:
        """Generate weekly customer experience report."""
        # Get insights for the last week
        insights = await self.analytics.generate_insights("7d")
        visualizations = self.analytics.create_visualizations(insights)
        
        # Calculate weekly metrics
        weekly_metrics = {
            'avg_sentiment': np.mean([segment['avg_sentiment'] 
                                    for segment in insights['customer_segments'].values()]),
            'avg_churn_risk': np.mean([segment['avg_churn_risk'] 
                                     for segment in insights['customer_segments'].values()]),
            'total_interactions': sum(segment['avg_interactions'] * segment['size'] 
                                    for segment in insights['customer_segments'].values())
        }
        
        # Prepare report data
        report_data = {
            'start_date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'weekly_metrics': weekly_metrics,
            'segments': insights['customer_segments'],
            'trends': insights['sentiment_trends'],
            'key_findings': insights['key_findings'],
            'visualizations': visualizations
        }
        
        # Render HTML template
        template = self.template_env.get_template('weekly_report.html')
        html_content = template.render(**report_data)
        
        # Generate PDF
        pdf_path = f"reports/weekly_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        os.makedirs('reports', exist_ok=True)
        pdfkit.from_string(html_content, pdf_path)
        
        return pdf_path
        
    async def generate_custom_report(self, time_range: str, metrics: List[str]) -> str:
        """Generate custom report based on specified time range and metrics."""
        insights = await self.analytics.generate_insights(time_range)
        visualizations = self.analytics.create_visualizations(insights)
        
        # Filter metrics based on request
        filtered_metrics = {
            metric: value for metric, value in insights.items()
            if metric in metrics
        }
        
        # Prepare report data
        report_data = {
            'time_range': time_range,
            'metrics': filtered_metrics,
            'visualizations': visualizations,
            'key_findings': insights['key_findings']
        }
        
        # Render HTML template
        template = self.template_env.get_template('custom_report.html')
        html_content = template.render(**report_data)
        
        # Generate PDF
        pdf_path = f"reports/custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        os.makedirs('reports', exist_ok=True)
        pdfkit.from_string(html_content, pdf_path)
        
        return pdf_path 