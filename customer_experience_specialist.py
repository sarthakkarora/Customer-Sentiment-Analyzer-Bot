from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime
import random
import os
import json
from dotenv import load_dotenv
import openai
import nltk
import spacy
from spacy.lang.en import English
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from celery import Celery
import redis


load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzers
sentiment_analyzer = SentimentIntensityAnalyzer()
text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize Redis and Celery
redis_client = redis.Redis(host='localhost', port=6379, db=0)
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

# Initialize database
engine = db.create_engine('postgresql://user:password@localhost/customer_service')
Session = sessionmaker(bind=engine)

@dataclass
class CustomerInfo:
    name: str
    complaint: str
    sentiment: str
    is_vip: bool
    lifetime_value: float
    order_history: dict
    previous_complaints: Optional[list]
    last_interaction_notes: Optional[str]
    customer_id: Optional[str] = None
    preferred_language: str = "en"
    communication_channel: str = "email"
    timezone: str = "UTC"

class CustomerExperienceSpecialist:
    def __init__(self, company_name: str, agent_name: str, agent_number: str):
        self.company_name = company_name
        self.agent_name = agent_name
        self.agent_number = agent_number
        self.ticket_id = self._generate_ticket_id()
        self.conversation_history = []
        self.session = Session()
        self.app = FastAPI()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/analyze")
        async def analyze_complaint(customer: CustomerInfo):
            return self._analyze_complaint(customer.complaint)

        @self.app.post("/generate_response")
        async def generate_response(customer: CustomerInfo):
            return self.generate_response(customer)

    def _generate_ticket_id(self) -> str:
        """Generate a unique ticket ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"TICKET-{timestamp}-{random.randint(1000, 9999)}"

    def _analyze_complaint(self, complaint: str) -> Dict:
        """Enhanced complaint analysis using multiple NLP techniques."""
        doc = nlp(complaint)
        
        # Extract entities and key phrases
        entities = [ent.text for ent in doc.ents]
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Multiple sentiment analysis
        vader_sentiment = sentiment_analyzer.polarity_scores(complaint)
        textblob_sentiment = TextBlob(complaint).sentiment
        bert_sentiment = text_classifier(complaint)[0]
        
        # Extract topics and categories
        topics = self._extract_topics(complaint)
        
        # Analyze emotional content
        emotions = self._analyze_emotions(complaint)
        
        # Generate context-aware analysis using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the following customer complaint and provide detailed insights about issues, emotions, and context."},
                {"role": "user", "content": complaint}
            ],
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": {
                "vader": vader_sentiment,
                "textblob": {"polarity": textblob_sentiment.polarity, "subjectivity": textblob_sentiment.subjectivity},
                "bert": bert_sentiment
            },
            "topics": topics,
            "emotions": emotions,
            "analysis": analysis
        }

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from the complaint."""
        doc = nlp(text)
        topics = []
        
        # Extract noun phrases as potential topics
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only consider multi-word phrases
                topics.append(chunk.text)
        
        return topics

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content using multiple approaches."""
        emotions = {
            "anger": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "fear": 0.0,
            "surprise": 0.0
        }
        
        # Use OpenAI for emotion analysis
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the emotional content of this text and provide scores for anger, joy, sadness, fear, and surprise."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        # Parse emotion scores from response
        try:
            emotion_scores = json.loads(response.choices[0].message.content)
            emotions.update(emotion_scores)
        except:
            pass
        
        return emotions

    def _generate_ai_response(self, customer: CustomerInfo, context: Dict) -> str:
        """Generate a sophisticated response using multiple AI models."""
        # Create a detailed prompt for GPT-4
        prompt = f"""
        You are a customer experience specialist for {self.company_name}. 
        
        Customer Profile:
        - Name: {customer.name}
        - VIP Status: {'Yes' if customer.is_vip else 'No'}
        - LTV: ${customer.lifetime_value}
        - Previous Complaints: {customer.previous_complaints}
        - Preferred Language: {customer.preferred_language}
        - Timezone: {customer.timezone}
        
        Current Complaint:
        {customer.complaint}
        
        Analysis:
        {context['analysis']}
        
        Sentiment Analysis:
        {json.dumps(context['sentiment'], indent=2)}
        
        Emotional Analysis:
        {json.dumps(context['emotions'], indent=2)}
        
        Generate a personalized, empathetic response that:
        1. Acknowledges the specific issues and emotions
        2. Shows genuine understanding of the customer's situation
        3. Provides concrete, actionable solutions
        4. Includes appropriate compensation based on customer value
        5. Demonstrates commitment to improvement
        6. Uses appropriate tone based on sentiment analysis
        7. Includes relevant follow-up actions
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert customer service representative who excels at turning negative experiences into positive ones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    def _get_compensation_strategy(self, customer: CustomerInfo, context: Dict) -> List[str]:
        """Use AI to determine the most appropriate compensation strategy."""
        # Create a detailed prompt for compensation analysis
        prompt = f"""
        Based on the following customer information and analysis, suggest appropriate compensation:
        
        Customer Profile:
        - Name: {customer.name}
        - VIP Status: {'Yes' if customer.is_vip else 'No'}
        - LTV: ${customer.lifetime_value}
        - Previous Complaints: {customer.previous_complaints}
        
        Complaint Analysis:
        {context['analysis']}
        
        Sentiment Analysis:
        {json.dumps(context['sentiment'], indent=2)}
        
        Emotional Analysis:
        {json.dumps(context['emotions'], indent=2)}
        
        Suggest specific compensation items that would be most effective in this situation.
        Consider:
        1. Customer value and loyalty
        2. Severity of the issue
        3. Emotional impact
        4. Previous interactions
        5. Industry standards
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in customer retention and satisfaction strategies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.split('\n')

    def generate_response(self, customer: CustomerInfo) -> str:
        """Generate a complete AI-enhanced response email."""
        # Analyze the complaint
        context = self._analyze_complaint(customer.complaint)
        
        # Generate AI response
        ai_response = self._generate_ai_response(customer, context)
        
        # Get compensation strategy
        compensation_items = self._get_compensation_strategy(customer, context)
        
        # Format the email
        email = f"""Subject: Re: Your Complaint #{self.ticket_id}

Hi {customer.name.split()[0]},

{ai_response}

**Compensation Details:**
{chr(10).join(f"- {item}" for item in compensation_items)}

I'll personally follow up on this within 24 hours. Need anything faster? Text me at {self.agent_number} — I'm here for you.

**PS:** Your next order is on us — use code `OOPSIE` at checkout for a treat on the house.

Warm regards,
{self.agent_name}
Customer Experience Specialist
{self.company_name}"""

        # Store conversation history
        self.conversation_history.append({
            "ticket_id": self.ticket_id,
            "customer": customer.name,
            "complaint": customer.complaint,
            "response": email,
            "timestamp": datetime.now().isoformat(),
            "analysis": context
        })

        # Store in database
        self._store_interaction(customer, context, email)

        return email

    def _store_interaction(self, customer: CustomerInfo, context: Dict, response: str):
        """Store the interaction in the database."""
        interaction = {
            "customer_id": customer.customer_id,
            "ticket_id": self.ticket_id,
            "complaint": customer.complaint,
            "response": response,
            "analysis": json.dumps(context),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Redis for quick access
        redis_client.hset(f"interaction:{self.ticket_id}", mapping=interaction)
        
        # Store in PostgreSQL for long-term storage
        self.session.execute(
            "INSERT INTO customer_interactions (customer_id, ticket_id, complaint, response, analysis, timestamp) "
            "VALUES (:customer_id, :ticket_id, :complaint, :response, :analysis, :timestamp)",
            interaction
        )
        self.session.commit()

# Example usage
if __name__ == "__main__":
    # Create a sample customer
    customer = CustomerInfo(
        name="John Doe",
        complaint="My package was delivered late and the product was defective. This is the second time this has happened this month.",
        sentiment="angry",
        is_vip=True,
        lifetime_value=1500.00,
        order_history={"total_orders": 10, "return_rate": 0.05},
        previous_complaints=["Late delivery last month"],
        last_interaction_notes="Customer prefers email communication",
        customer_id="CUST123",
        preferred_language="en",
        communication_channel="email",
        timezone="America/New_York"
    )

    # Initialize the specialist
    specialist = CustomerExperienceSpecialist(
        company_name="Acme Corp",
        agent_name="Sarah Johnson",
        agent_number="+1-555-123-4567"
    )

    # Generate and print the response
    response = specialist.generate_response(customer)
    print(response) 
