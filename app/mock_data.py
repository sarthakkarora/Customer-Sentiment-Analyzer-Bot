import random
import time
from datetime import datetime

class MockDataGenerator:
    def __init__(self):
        self.sentiment_base = 0.5
        self.churn_risk_base = 0.3
        self.active_customers = 1000

    def generate_metrics(self):
        # Add some random variation
        self.sentiment_base += random.uniform(-0.1, 0.1)
        self.sentiment_base = max(-1, min(1, self.sentiment_base))

        self.churn_risk_base += random.uniform(-0.05, 0.05)
        self.churn_risk_base = max(0, min(1, self.churn_risk_base))

        self.active_customers += random.randint(-10, 10)
        self.active_customers = max(0, self.active_customers)

        # Generate alerts if metrics cross thresholds
        alerts = []
        if self.sentiment_base < 0:
            alerts.append({
                "type": "low_sentiment",
                "message": f"Low sentiment detected: {self.sentiment_base:.2f}",
                "timestamp": datetime.now().isoformat()
            })
        if self.churn_risk_base > 0.7:
            alerts.append({
                "type": "high_churn_risk",
                "message": f"High churn risk detected: {self.churn_risk_base:.2f}",
                "timestamp": datetime.now().isoformat()
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "sentiment": self.sentiment_base,
                "churn_risk": self.churn_risk_base,
                "active_customers": self.active_customers
            },
            "alerts": alerts
        } 