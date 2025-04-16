from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Dict, List, Any
import plotly.graph_objects as go
from datetime import datetime, timedelta
from loguru import logger
from analytics import CustomerAnalytics
from report_generator import ReportGenerator
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

class Dashboard:
    def __init__(self):
        self.analytics = CustomerAnalytics()
        self.report_generator = ReportGenerator()
        
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard."""
        insights = await self.analytics.generate_insights("1h")
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_customers': sum(segment['size'] for segment in insights['customer_segments'].values()),
                'avg_sentiment': np.mean([segment['avg_sentiment'] for segment in insights['customer_segments'].values()]),
                'avg_churn_risk': np.mean([segment['avg_churn_risk'] for segment in insights['customer_segments'].values()]),
                'high_risk_customers': sum(1 for finding in insights['key_findings'] 
                                         if finding['type'] == 'high_risk_segment')
            },
            'alerts': insights['key_findings']
        }
        
    def create_realtime_visualizations(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create real-time visualizations for dashboard."""
        visualizations = {}
        
        # Create metrics gauge
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['metrics']['avg_sentiment'],
            title={'text': "Average Sentiment"},
            gauge={'axis': {'range': [-1, 1]}}
        ))
        visualizations['sentiment_gauge'] = fig_metrics.to_json()
        
        # Create alerts timeline
        fig_alerts = go.Figure()
        alerts = metrics['alerts']
        if alerts:
            fig_alerts.add_trace(go.Scatter(
                x=[alert['timestamp'] for alert in alerts],
                y=[1] * len(alerts),
                mode='markers',
                marker=dict(
                    size=10,
                    color='red'
                ),
                text=[alert['type'] for alert in alerts]
            ))
        visualizations['alerts_timeline'] = fig_alerts.to_json()
        
        return visualizations

dashboard = Dashboard()

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            metrics = await dashboard.get_realtime_metrics()
            visualizations = dashboard.create_realtime_visualizations(metrics)
            await websocket.send_json({
                'metrics': metrics,
                'visualizations': visualizations
            })
            await asyncio.sleep(5)  # Update every 5 seconds
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)

@app.get("/dashboard")
async def get_dashboard():
    """Serve the dashboard HTML page."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Experience Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .card { padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .metric { font-size: 24px; font-weight: bold; }
            .alert { color: red; }
        </style>
    </head>
    <body>
        <h1>Real-time Customer Experience Dashboard</h1>
        <div class="grid">
            <div class="card">
                <h2>Sentiment Gauge</h2>
                <div id="sentiment-gauge"></div>
            </div>
            <div class="card">
                <h2>Alerts Timeline</h2>
                <div id="alerts-timeline"></div>
            </div>
        </div>
        <script>
            const ws = new WebSocket('ws://' + window.location.host + '/ws/metrics');
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const metrics = data.metrics;
                const visualizations = data.visualizations;
                
                // Update visualizations
                Plotly.newPlot('sentiment-gauge', JSON.parse(visualizations.sentiment_gauge));
                Plotly.newPlot('alerts-timeline', JSON.parse(visualizations.alerts_timeline));
            };
        </script>
    </body>
    </html>
    """)

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics."""
    return await dashboard.get_realtime_metrics()

@app.get("/api/visualizations")
async def get_visualizations():
    """Get current visualizations."""
    metrics = await dashboard.get_realtime_metrics()
    return dashboard.create_realtime_visualizations(metrics) 