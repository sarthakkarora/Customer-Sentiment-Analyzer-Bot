from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import uvicorn
import asyncio
import json
from datetime import datetime
import random
from typing import Dict, List
import numpy as np
from loguru import logger
from ml_models import SentimentAnalyzer, EmotionDetector, ChurnPredictor
from monitoring import SystemMonitor
from personalized_tracker import PersonalizedTracker
from behavior_analyzer import BehaviorAnalyzer
from reward_system import RewardSystem

# Create FastAPI app
app = FastAPI(
    title="Customer Experience Analytics with Personalized Rewards",
    description="Real-time customer experience monitoring and analytics"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Initialize components
sentiment_analyzer = SentimentAnalyzer()
emotion_detector = EmotionDetector()
churn_predictor = ChurnPredictor()
system_monitor = SystemMonitor()
personalized_tracker = PersonalizedTracker()
behavior_analyzer = BehaviorAnalyzer()
reward_system = RewardSystem()

# Store active WebSocket connections
active_connections = set()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Generate mock customer interaction data
            customer_id = f"cust_{random.randint(1000, 9999)}"
            data = generate_mock_data(customer_id)
            
            # Process the data through all components
            processed_data = await process_data(data, customer_id)
            
            # Send the enhanced data to the client
            await websocket.send_json(processed_data)
            
            # Wait before sending next update
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        active_connections.remove(websocket)

def generate_mock_data(customer_id: str) -> Dict:
    """Generate mock customer interaction data"""
    messages = [
        "I love your products!",
        "When will my order arrive?",
        "The quality is amazing",
        "I'm having issues with my account",
        "Can you help me with a return?",
        "Your customer service is great!",
        "I'm disappointed with my purchase",
        "The app is not working properly",
        "Thank you for the quick response",
        "I need help with my subscription"
    ]
    
    return {
        "customer_id": customer_id,
        "timestamp": datetime.now().isoformat(),
        "message": random.choice(messages),
        "interaction_type": random.choice(["chat", "email", "phone", "social"]),
        "app_usage": {
            "sessions": random.randint(1, 10),
            "duration": random.randint(5, 60),
            "features_used": random.sample(["search", "cart", "wishlist", "profile"], random.randint(1, 4))
        }
    }

async def process_data(data: Dict, customer_id: str) -> Dict:
    """Process customer interaction data through all components"""
    # Analyze sentiment and emotion
    sentiment = sentiment_analyzer.analyze(data["message"])
    emotion = emotion_detector.detect(data["message"])
    
    # Predict churn risk
    churn_risk = churn_predictor.predict({
        "sentiment": sentiment["score"],
        "emotion": emotion["primary"],
        "interaction_type": data["interaction_type"],
        "app_usage": data["app_usage"]
    })
    
    # Update personalized tracker
    personalized_tracker.update_profile(customer_id, {
        "message": data["message"],
        "sentiment": sentiment,
        "emotion": emotion,
        "timestamp": data["timestamp"],
        "interaction_type": data["interaction_type"],
        "app_usage": data["app_usage"]
    })
    
    # Get customer profile
    profile = personalized_tracker.get_profile(customer_id)
    
    # Analyze behavior
    behavior = behavior_analyzer.analyze_message_patterns(data["message"])
    behavior_type = behavior_analyzer.identify_behavior_type(profile["interaction_history"])
    engagement_quality = behavior_analyzer.calculate_engagement_quality(profile)
    next_actions = behavior_analyzer.predict_next_actions(profile)
    response_strategy = behavior_analyzer.generate_personalized_response(profile)
    
    # Generate rewards
    rewards = reward_system.generate_personalized_rewards(
        customer_id=customer_id,
        loyalty_level=profile["loyalty_level"],
        frustration_level=profile["frustration_level"],
        engagement_score=profile["engagement_score"]
    )
    
    # Update reward points
    reward_system.update_customer_rewards(
        customer_id=customer_id,
        loyalty_level=profile["loyalty_level"],
        engagement_score=profile["engagement_score"]
    )
    
    # Get reward history
    reward_history = reward_system.get_reward_history(customer_id)
    
    # Prepare response
    return {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "metrics": {
            "sentiment": sentiment,
            "emotion": emotion,
            "churn_risk": churn_risk,
            "frustration_level": profile["frustration_level"],
            "engagement_score": profile["engagement_score"],
            "loyalty_level": profile["loyalty_level"]
        },
        "behavior": {
            "type": behavior_type,
            "patterns": behavior,
            "engagement_quality": engagement_quality,
            "next_actions": next_actions,
            "response_strategy": response_strategy
        },
        "rewards": {
            "available": rewards,
            "history": reward_history,
            "points": reward_history["reward_points"]
        },
        "profile": {
            "join_date": profile["join_date"],
            "total_interactions": len(profile["interaction_history"]),
            "app_usage_patterns": profile["app_usage_patterns"],
            "sentiment_trend": profile["sentiment_trend"][-5:] if len(profile["sentiment_trend"]) >= 5 else profile["sentiment_trend"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 