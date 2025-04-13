import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sentence_transformers import SentenceTransformer
import faiss
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import mlflow
import wandb
from loguru import logger
import joblib
import os

class CustomerExperiencePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.openai_embeddings = OpenAIEmbeddings()
        self._initialize_mlflow()
        self._initialize_wandb()
        self._initialize_pinecone()
        
    def _initialize_mlflow(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("customer_experience")
        
    def _initialize_wandb(self):
        wandb.init(project="customer-experience", entity=os.getenv("WANDB_ENTITY"))
        
    def _initialize_pinecone(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        self.index = pinecone.Index("customer-experience")
        
    def train_churn_prediction_model(self, data: pd.DataFrame):
        """Train a model to predict customer churn."""
        X = data.drop(['churn', 'customer_id'], axis=1)
        y = data['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(),
            'xgboost': xgb.XGBClassifier(),
            'lightgbm': lgb.LGBMClassifier(),
            'catboost': cb.CatBoostClassifier(verbose=False)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"churn_prediction_{name}"):
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                # Log metrics
                mlflow.log_metric("accuracy", score)
                wandb.log({f"{name}_accuracy": score})
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
                # Save model
                joblib.dump(model, f"models/churn_{name}.joblib")
                
        self.models['churn'] = best_model
        logger.info(f"Best churn prediction model: {best_model.__class__.__name__} with accuracy {best_score}")
        
    def train_sentiment_analysis_model(self, data: pd.DataFrame):
        """Train a model for sentiment analysis."""
        # Convert text to embeddings
        X = self.embeddings_model.encode(data['text'].values)
        y = data['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train neural network
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')  # 3 sentiment classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        with mlflow.start_run(run_name="sentiment_analysis"):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                batch_size=32
            )
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": history.history['accuracy'][-1],
                "val_accuracy": history.history['val_accuracy'][-1]
            })
            wandb.log({
                "accuracy": history.history['accuracy'][-1],
                "val_accuracy": history.history['val_accuracy'][-1]
            })
            
        self.models['sentiment'] = model
        model.save("models/sentiment_model.h5")
        
    def train_response_quality_model(self, data: pd.DataFrame):
        """Train a model to predict response quality."""
        X = self.embeddings_model.encode(data['response'].values)
        y = data['quality_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train regression model
        model = xgb.XGBRegressor()
        
        with mlflow.start_run(run_name="response_quality"):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("r2_score", score)
            wandb.log({"r2_score": score})
            
        self.models['response_quality'] = model
        joblib.dump(model, "models/response_quality.joblib")
        
    def predict_churn_probability(self, customer_data: dict) -> float:
        """Predict probability of customer churn."""
        if 'churn' not in self.models:
            raise ValueError("Churn prediction model not trained")
            
        # Prepare features
        features = pd.DataFrame([customer_data])
        return self.models['churn'].predict_proba(features)[0][1]
        
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment using trained model."""
        if 'sentiment' not in self.models:
            raise ValueError("Sentiment analysis model not trained")
            
        # Get embeddings
        embedding = self.embeddings_model.encode([text])
        
        # Predict sentiment
        prediction = self.models['sentiment'].predict(embedding)
        sentiment_scores = prediction[0]
        
        return {
            'positive': float(sentiment_scores[0]),
            'neutral': float(sentiment_scores[1]),
            'negative': float(sentiment_scores[2])
        }
        
    def predict_response_quality(self, response: str) -> float:
        """Predict quality score of a response."""
        if 'response_quality' not in self.models:
            raise ValueError("Response quality model not trained")
            
        # Get embeddings
        embedding = self.embeddings_model.encode([response])
        
        # Predict quality
        return float(self.models['response_quality'].predict(embedding)[0])
        
    def find_similar_cases(self, query: str, k: int = 5) -> list:
        """Find similar past cases using vector similarity search."""
        # Get query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding[0].tolist(),
            top_k=k,
            include_metadata=True
        )
        
        return results['matches']
        
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.embeddings_model.encode(texts)
        
    def store_case_embedding(self, case_id: str, text: str, metadata: dict):
        """Store a case embedding in the vector database."""
        embedding = self.embeddings_model.encode([text])[0]
        
        self.index.upsert(
            vectors=[(case_id, embedding.tolist(), metadata)]
        ) 