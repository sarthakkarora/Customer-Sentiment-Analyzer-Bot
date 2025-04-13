import asyncio
from typing import Dict, List, Any
import json
from datetime import datetime
import aiokafka
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from fastapi import WebSocket
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import redis
from elasticsearch import AsyncElasticsearch
import logging
from loguru import logger
import os

class RealTimeProcessor:
    def __init__(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092").split(","),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0))
        )
        self.es = AsyncElasticsearch(
            [os.getenv("ELASTICSEARCH_HOST", "localhost:9200")],
            http_auth=(os.getenv("ELASTICSEARCH_USER"), os.getenv("ELASTICSEARCH_PASSWORD"))
        )
        self._initialize_models()
        self._initialize_kafka_topics()
        self.websocket_connections: Dict[str, WebSocket] = {}
        
    def _initialize_models(self):
        """Initialize ML models for real-time processing."""
        # Load pre-trained models
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.emotion_model = load_model('models/emotion_model.h5')
        self.churn_model = load_model('models/churn_model.h5')
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            self.sentiment_model.model = self.sentiment_model.model.cuda()
            self.embeddings_model = self.embeddings_model.cuda()
            self.emotion_model = self.emotion_model.cuda()
            self.churn_model = self.churn_model.cuda()
            
    def _initialize_kafka_topics(self):
        """Initialize required Kafka topics."""
        admin_client = KafkaAdminClient(
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
        )
        
        topics = [
            NewTopic(name="customer_complaints", num_partitions=3, replication_factor=1),
            NewTopic(name="customer_responses", num_partitions=3, replication_factor=1),
            NewTopic(name="customer_metrics", num_partitions=3, replication_factor=1),
            NewTopic(name="system_alerts", num_partitions=1, replication_factor=1)
        ]
        
        try:
            admin_client.create_topics(new_topics=topics, validate_only=False)
        except Exception as e:
            logger.warning(f"Topics may already exist: {str(e)}")
            
    async def process_complaint(self, complaint: Dict[str, Any]):
        """Process a customer complaint in real-time."""
        # Generate embeddings
        complaint_text = complaint['text']
        embedding = self.embeddings_model.encode([complaint_text])[0]
        
        # Analyze sentiment
        sentiment = self.sentiment_model(complaint_text)[0]
        
        # Predict emotions
        emotion_scores = self.emotion_model.predict(np.array([embedding]))[0]
        
        # Predict churn risk
        churn_risk = self.churn_model.predict(np.array([embedding]))[0][0]
        
        # Prepare metrics
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'customer_id': complaint['customer_id'],
            'complaint_id': complaint['complaint_id'],
            'sentiment': sentiment['label'],
            'sentiment_score': sentiment['score'],
            'emotion_scores': emotion_scores.tolist(),
            'churn_risk': float(churn_risk),
            'embedding': embedding.tolist()
        }
        
        # Store in Redis for quick access
        self.redis_client.hset(
            f"complaint:{complaint['complaint_id']}",
            mapping=metrics
        )
        
        # Send to Kafka
        self.kafka_producer.send('customer_metrics', value=metrics)
        
        # Store in Elasticsearch
        await self.es.index(
            index="customer-complaints",
            body=metrics
        )
        
        # Notify connected websockets
        await self._notify_websockets(metrics)
        
        return metrics
        
    async def process_response(self, response: Dict[str, Any]):
        """Process a generated response in real-time."""
        # Generate embeddings
        response_text = response['text']
        embedding = self.embeddings_model.encode([response_text])[0]
        
        # Analyze quality
        quality_score = self._analyze_response_quality(response_text, embedding)
        
        # Prepare metrics
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'customer_id': response['customer_id'],
            'complaint_id': response['complaint_id'],
            'response_id': response['response_id'],
            'quality_score': quality_score,
            'embedding': embedding.tolist()
        }
        
        # Store in Redis
        self.redis_client.hset(
            f"response:{response['response_id']}",
            mapping=metrics
        )
        
        # Send to Kafka
        self.kafka_producer.send('customer_responses', value=metrics)
        
        # Store in Elasticsearch
        await self.es.index(
            index="customer-responses",
            body=metrics
        )
        
        # Notify connected websockets
        await self._notify_websockets(metrics)
        
        return metrics
        
    def _analyze_response_quality(self, text: str, embedding: np.ndarray) -> float:
        """Analyze the quality of a response."""
        # Calculate similarity with ideal responses
        ideal_responses = self.redis_client.smembers("ideal_responses")
        similarities = []
        
        for ideal_response in ideal_responses:
            ideal_embedding = np.array(json.loads(ideal_response))
            similarity = np.dot(embedding, ideal_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ideal_embedding)
            )
            similarities.append(similarity)
            
        # Calculate quality score
        quality_score = np.mean(similarities) if similarities else 0.5
        
        return float(quality_score)
        
    async def _notify_websockets(self, data: Dict[str, Any]):
        """Notify all connected websockets about new data."""
        for websocket in self.websocket_connections.values():
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error sending to websocket: {str(e)}")
                
    async def consume_kafka_messages(self):
        """Consume messages from Kafka topics."""
        consumer = aiokafka.AIOKafkaConsumer(
            'customer_complaints',
            'customer_responses',
            'customer_metrics',
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092").split(","),
            group_id="real_time_processor"
        )
        
        await consumer.start()
        try:
            async for msg in consumer:
                data = json.loads(msg.value)
                if msg.topic == 'customer_complaints':
                    await self.process_complaint(data)
                elif msg.topic == 'customer_responses':
                    await self.process_response(data)
                elif msg.topic == 'customer_metrics':
                    await self._notify_websockets(data)
        finally:
            await consumer.stop()
            
    async def start_processing(self):
        """Start the real-time processing loop."""
        while True:
            try:
                await self.consume_kafka_messages()
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying
                
    def register_websocket(self, client_id: str, websocket: WebSocket):
        """Register a new websocket connection."""
        self.websocket_connections[client_id] = websocket
        
    def unregister_websocket(self, client_id: str):
        """Unregister a websocket connection."""
        if client_id in self.websocket_connections:
            del self.websocket_connections[client_id] 