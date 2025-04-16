from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
import re
from collections import Counter

class BehaviorAnalyzer:
    def __init__(self):
        self.patterns = {
            'frustration': [
                r'(frustrat|annoy|angry|upset|disappoint)',
                r'(not working|broken|error|issue|problem)',
                r'(why|how come|cant|cannot|wont|will not)',
                r'(terrible|awful|horrible|bad)'
            ],
            'satisfaction': [
                r'(great|excellent|amazing|wonderful|perfect)',
                r'(thank|thanks|appreciate|helpful)',
                r'(love|enjoy|happy|pleased)',
                r'(smooth|easy|simple|quick)'
            ],
            'urgency': [
                r'(urgent|immediately|right now|asap)',
                r'(emergency|critical|important)',
                r'(need|must|have to)'
            ],
            'loyalty': [
                r'(long time|regular|frequent|always)',
                r'(recommend|refer|share)',
                r'(loyal|faithful|dedicated)'
            ]
        }
        
        self.behavior_models = {
            'shopper': {
                'patterns': ['browse', 'compare', 'price', 'deal', 'offer'],
                'metrics': ['time_spent', 'page_views', 'cart_adds']
            },
            'researcher': {
                'patterns': ['how to', 'what is', 'explain', 'information'],
                'metrics': ['search_queries', 'article_views', 'time_spent']
            },
            'problem_solver': {
                'patterns': ['help', 'issue', 'problem', 'fix', 'support'],
                'metrics': ['support_tickets', 'chat_sessions', 'resolution_time']
            },
            'social_engager': {
                'patterns': ['share', 'comment', 'review', 'community'],
                'metrics': ['social_posts', 'comments', 'likes']
            }
        }

    def analyze_message_patterns(self, message: str) -> Dict[str, float]:
        """Analyze message for behavioral patterns"""
        message = message.lower()
        scores = {}
        
        for category, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message))
                score += matches
            scores[category] = min(1.0, score / 5)  # Normalize to 0-1
        
        return scores

    def identify_behavior_type(self, interaction_history: List[Dict]) -> Dict[str, float]:
        """Identify customer behavior type based on interaction history"""
        if not interaction_history:
            return {'unknown': 1.0}
            
        # Analyze recent messages
        recent_messages = [i['message'] for i in interaction_history[-10:]]
        message_text = ' '.join(recent_messages).lower()
        
        behavior_scores = {}
        for behavior_type, model in self.behavior_models.items():
            score = 0
            for pattern in model['patterns']:
                matches = len(re.findall(pattern, message_text))
                score += matches
            behavior_scores[behavior_type] = min(1.0, score / 5)
        
        # Normalize scores
        total = sum(behavior_scores.values())
        if total > 0:
            behavior_scores = {k: v/total for k, v in behavior_scores.items()}
        
        return behavior_scores

    def calculate_engagement_quality(self, interaction_history: List[Dict]) -> Dict[str, float]:
        """Calculate quality of customer engagement"""
        if not interaction_history:
            return {'quality_score': 0.0, 'metrics': {}}
            
        recent_interactions = interaction_history[-30:]  # Last 30 interactions
        
        metrics = {
            'response_time': [],
            'message_length': [],
            'interaction_frequency': [],
            'topic_diversity': set(),
            'sentiment_scores': []
        }
        
        for interaction in recent_interactions:
            metrics['response_time'].append(interaction.get('response_time', 0))
            metrics['message_length'].append(len(interaction.get('message', '')))
            metrics['topic_diversity'].add(interaction.get('type', ''))
            metrics['sentiment_scores'].append(interaction.get('sentiment', 0))
        
        # Calculate quality metrics
        quality_metrics = {
            'avg_response_time': np.mean(metrics['response_time']) if metrics['response_time'] else 0,
            'avg_message_length': np.mean(metrics['message_length']) if metrics['message_length'] else 0,
            'topic_diversity': len(metrics['topic_diversity']) / len(recent_interactions),
            'avg_sentiment': np.mean(metrics['sentiment_scores']) if metrics['sentiment_scores'] else 0,
            'interaction_frequency': len(recent_interactions) / 30  # Normalize to daily average
        }
        
        # Calculate overall quality score
        quality_score = (
            0.3 * (1 - min(quality_metrics['avg_response_time'] / 300, 1)) +  # Response time
            0.2 * min(quality_metrics['avg_message_length'] / 100, 1) +  # Message length
            0.2 * quality_metrics['topic_diversity'] +  # Topic diversity
            0.2 * (quality_metrics['avg_sentiment'] + 1) / 2 +  # Sentiment
            0.1 * quality_metrics['interaction_frequency']  # Frequency
        )
        
        return {
            'quality_score': min(1.0, max(0.0, quality_score)),
            'metrics': quality_metrics
        }

    def predict_next_actions(self, interaction_history: List[Dict]) -> List[Dict]:
        """Predict likely next actions based on behavior patterns"""
        if not interaction_history:
            return []
            
        recent_interactions = interaction_history[-10:]
        behavior_type = self.identify_behavior_type(recent_interactions)
        dominant_behavior = max(behavior_type.items(), key=lambda x: x[1])[0]
        
        predictions = []
        
        # Predict based on behavior type
        if dominant_behavior == 'shopper':
            predictions.extend([
                {'action': 'browse_products', 'probability': 0.7},
                {'action': 'check_prices', 'probability': 0.6},
                {'action': 'add_to_cart', 'probability': 0.5}
            ])
        elif dominant_behavior == 'researcher':
            predictions.extend([
                {'action': 'read_articles', 'probability': 0.7},
                {'action': 'search_topics', 'probability': 0.6},
                {'action': 'watch_tutorials', 'probability': 0.5}
            ])
        elif dominant_behavior == 'problem_solver':
            predictions.extend([
                {'action': 'contact_support', 'probability': 0.7},
                {'action': 'search_solutions', 'probability': 0.6},
                {'action': 'read_faq', 'probability': 0.5}
            ])
        elif dominant_behavior == 'social_engager':
            predictions.extend([
                {'action': 'share_content', 'probability': 0.7},
                {'action': 'write_review', 'probability': 0.6},
                {'action': 'comment_post', 'probability': 0.5}
            ])
        
        # Add general predictions based on recent activity
        last_interaction = recent_interactions[-1]
        if last_interaction.get('sentiment', 0) < -0.5:
            predictions.append({'action': 'complain', 'probability': 0.8})
        elif last_interaction.get('sentiment', 0) > 0.5:
            predictions.append({'action': 'praise', 'probability': 0.8})
        
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)

    def generate_personalized_response(self, interaction_history: List[Dict]) -> Dict:
        """Generate personalized response strategy"""
        if not interaction_history:
            return {'strategy': 'welcome', 'tone': 'friendly'}
            
        recent_interactions = interaction_history[-5:]
        behavior_scores = self.identify_behavior_type(recent_interactions)
        message_patterns = self.analyze_message_patterns(recent_interactions[-1]['message'])
        engagement_quality = self.calculate_engagement_quality(recent_interactions)
        
        # Determine response strategy
        if message_patterns['frustration'] > 0.7:
            strategy = 'resolve_issue'
            tone = 'empathetic'
        elif message_patterns['urgency'] > 0.7:
            strategy = 'quick_assist'
            tone = 'efficient'
        elif behavior_scores.get('researcher', 0) > 0.7:
            strategy = 'informative'
            tone = 'professional'
        elif behavior_scores.get('social_engager', 0) > 0.7:
            strategy = 'engage'
            tone = 'conversational'
        else:
            strategy = 'assist'
            tone = 'friendly'
        
        # Determine response timing
        if engagement_quality['quality_score'] > 0.8:
            timing = 'immediate'
        elif message_patterns['urgency'] > 0.5:
            timing = 'priority'
        else:
            timing = 'normal'
        
        return {
            'strategy': strategy,
            'tone': tone,
            'timing': timing,
            'personalization_level': engagement_quality['quality_score'],
            'suggested_actions': self.predict_next_actions(recent_interactions)
        } 