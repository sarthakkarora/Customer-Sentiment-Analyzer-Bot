from prometheus_client import start_http_server, Counter, Gauge, Histogram
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from loguru import logger
import sentry_sdk
from elasticsearch import Elasticsearch
from datetime import datetime
import json
import os

class CustomerExperienceMonitor:
    def __init__(self):
        self._initialize_metrics()
        self._initialize_tracing()
        self._initialize_logging()
        self._initialize_error_tracking()
        self._initialize_elasticsearch()
        
    def _initialize_metrics(self):
        # Response metrics
        self.response_time = Histogram(
            'customer_response_time_seconds',
            'Time taken to generate responses',
            ['company', 'agent']
        )
        self.response_quality = Gauge(
            'customer_response_quality',
            'Quality score of generated responses',
            ['company', 'agent']
        )
        
        # Customer metrics
        self.customer_sentiment = Gauge(
            'customer_sentiment_score',
            'Customer sentiment score',
            ['company', 'customer_id']
        )
        self.customer_churn_risk = Gauge(
            'customer_churn_risk',
            'Customer churn risk score',
            ['company', 'customer_id']
        )
        
        # System metrics
        self.active_tickets = Gauge(
            'active_customer_tickets',
            'Number of active customer tickets',
            ['company']
        )
        self.ticket_resolution_time = Histogram(
            'ticket_resolution_time_seconds',
            'Time taken to resolve tickets',
            ['company', 'priority']
        )
        
    def _initialize_tracing(self):
        trace.set_tracer_provider(TracerProvider())
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317"),
                insecure=True
            )
        )
        trace.get_tracer_provider().add_span_processor(span_processor)
        
    def _initialize_logging(self):
        logger.add(
            "logs/customer_service_{time}.log",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            level="INFO"
        )
        
    def _initialize_error_tracking(self):
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            traces_sample_rate=1.0,
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
    def _initialize_elasticsearch(self):
        self.es = Elasticsearch(
            [os.getenv("ELASTICSEARCH_HOST", "localhost:9200")],
            http_auth=(os.getenv("ELASTICSEARCH_USER"), os.getenv("ELASTICSEARCH_PASSWORD"))
        )
        
    def log_response_metrics(self, company: str, agent: str, response_time: float, quality_score: float):
        """Log metrics for a generated response."""
        self.response_time.labels(company=company, agent=agent).observe(response_time)
        self.response_quality.labels(company=company, agent=agent).set(quality_score)
        
        # Log to Elasticsearch
        self.es.index(
            index="customer-responses",
            body={
                "timestamp": datetime.utcnow(),
                "company": company,
                "agent": agent,
                "response_time": response_time,
                "quality_score": quality_score
            }
        )
        
    def log_customer_metrics(self, company: str, customer_id: str, sentiment_score: float, churn_risk: float):
        """Log metrics for customer interactions."""
        self.customer_sentiment.labels(company=company, customer_id=customer_id).set(sentiment_score)
        self.customer_churn_risk.labels(company=company, customer_id=customer_id).set(churn_risk)
        
        # Log to Elasticsearch
        self.es.index(
            index="customer-metrics",
            body={
                "timestamp": datetime.utcnow(),
                "company": company,
                "customer_id": customer_id,
                "sentiment_score": sentiment_score,
                "churn_risk": churn_risk
            }
        )
        
    def log_ticket_metrics(self, company: str, ticket_id: str, priority: str, resolution_time: float):
        """Log metrics for ticket resolution."""
        self.ticket_resolution_time.labels(company=company, priority=priority).observe(resolution_time)
        self.active_tickets.labels(company=company).dec()
        
        # Log to Elasticsearch
        self.es.index(
            index="ticket-metrics",
            body={
                "timestamp": datetime.utcnow(),
                "company": company,
                "ticket_id": ticket_id,
                "priority": priority,
                "resolution_time": resolution_time
            }
        )
        
    def log_error(self, error: Exception, context: dict):
        """Log errors with context."""
        logger.error(f"Error occurred: {str(error)}", extra=context)
        sentry_sdk.capture_exception(error, extra=context)
        
    def log_performance_metrics(self, metrics: dict):
        """Log general performance metrics."""
        # Log to Elasticsearch
        self.es.index(
            index="performance-metrics",
            body={
                "timestamp": datetime.utcnow(),
                **metrics
            }
        )
        
    def start_monitoring(self, port: int = 8000):
        """Start the monitoring server."""
        start_http_server(port)
        logger.info(f"Monitoring server started on port {port}")
        
    def instrument_fastapi(self, app):
        """Instrument FastAPI application for tracing."""
        FastAPIInstrumentor.instrument_app(app)
        
    def get_customer_insights(self, company: str, customer_id: str) -> dict:
        """Get insights about a customer's interactions."""
        # Query Elasticsearch
        response = self.es.search(
            index="customer-metrics",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"company": company}},
                            {"term": {"customer_id": customer_id}}
                        ]
                    }
                },
                "aggs": {
                    "avg_sentiment": {"avg": {"field": "sentiment_score"}},
                    "avg_churn_risk": {"avg": {"field": "churn_risk"}},
                    "interaction_count": {"value_count": {"field": "customer_id"}}
                }
            }
        )
        
        return {
            "average_sentiment": response["aggregations"]["avg_sentiment"]["value"],
            "average_churn_risk": response["aggregations"]["avg_churn_risk"]["value"],
            "interaction_count": response["aggregations"]["interaction_count"]["value"]
        }
        
    def get_system_health(self) -> dict:
        """Get system health metrics."""
        return {
            "active_tickets": self.active_tickets._value.get(),
            "average_response_time": self.response_time._sum.get() / self.response_time._count.get() if self.response_time._count.get() > 0 else 0,
            "average_quality_score": self.response_quality._value.get()
        } 