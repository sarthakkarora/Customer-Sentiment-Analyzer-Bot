from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional
import json
from loguru import logger

class PersonalizedTracker:
    def __init__(self):
        self.customer_profiles = {}
        self.engagement_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        self.loyalty_levels = {
            'platinum': 0.9,
            'gold': 0.7,
            'silver': 0.5,
            'bronze': 0.3
        }

    def update_customer_profile(self, customer_id: str, interaction_data: Dict):
        """Update or create customer profile with new interaction data"""
        if customer_id not in self.customer_profiles:
            self.customer_profiles[customer_id] = {
                'join_date': datetime.now(),
                'interactions': [],
                'sentiment_history': [],
                'activity_days': set(),
                'total_spent': 0,
                'returns_count': 0,
                'loyalty_score': 0.5,
                'engagement_score': 0.5,
                'frustration_level': 0,
                'last_interaction': datetime.now(),
                'discount_offered': False
            }

        profile = self.customer_profiles[customer_id]
        
        # Update interaction history
        profile['interactions'].append({
            'timestamp': datetime.now(),
            'type': interaction_data.get('type', 'chat'),
            'sentiment': interaction_data.get('sentiment', 0),
            'message': interaction_data.get('message', ''),
            'response_time': interaction_data.get('response_time', 0)
        })

        # Update sentiment history
        profile['sentiment_history'].append(interaction_data.get('sentiment', 0))
        
        # Track activity days
        profile['activity_days'].add(datetime.now().date())
        
        # Update frustration level
        self._update_frustration_level(profile, interaction_data)
        
        # Update engagement score
        self._update_engagement_score(profile)
        
        # Update loyalty score
        self._update_loyalty_score(profile)
        
        # Update last interaction
        profile['last_interaction'] = datetime.now()

    def _update_frustration_level(self, profile: Dict, interaction_data: Dict):
        """Calculate frustration level based on recent interactions"""
        recent_interactions = [i for i in profile['interactions'][-10:] if i['type'] == 'chat']
        if not recent_interactions:
            return

        # Factors affecting frustration:
        # 1. Negative sentiment in messages
        # 2. Short response times (might indicate rushed interactions)
        # 3. Repeated similar messages
        # 4. High frequency of interactions in short time
        
        sentiment_score = np.mean([i['sentiment'] for i in recent_interactions])
        response_times = [i['response_time'] for i in recent_interactions]
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Calculate frustration score (0-1)
        frustration_score = (
            0.4 * (1 - (sentiment_score + 1) / 2) +  # Negative sentiment contribution
            0.3 * (1 - min(avg_response_time / 300, 1)) +  # Response time contribution
            0.3 * self._calculate_message_repetition(recent_interactions)  # Message repetition contribution
        )
        
        profile['frustration_level'] = min(1, max(0, frustration_score))

    def _update_engagement_score(self, profile: Dict):
        """Calculate engagement score based on activity patterns"""
        # Factors affecting engagement:
        # 1. Frequency of interactions
        # 2. Regularity of usage
        # 3. Time spent in app
        # 4. Response to previous interactions
        
        days_since_join = (datetime.now() - profile['join_date']).days
        if days_since_join == 0:
            days_since_join = 1
            
        activity_days = len(profile['activity_days'])
        interaction_count = len(profile['interactions'])
        
        # Calculate engagement score (0-1)
        engagement_score = (
            0.4 * (activity_days / days_since_join) +  # Regularity
            0.3 * min(interaction_count / (days_since_join * 5), 1) +  # Frequency
            0.3 * (1 - profile['frustration_level'])  # Positive experience
        )
        
        profile['engagement_score'] = min(1, max(0, engagement_score))

    def _update_loyalty_score(self, profile: Dict):
        """Calculate loyalty score based on customer behavior"""
        # Factors affecting loyalty:
        # 1. Time since joining
        # 2. Purchase history
        # 3. Return rate
        # 4. Engagement level
        
        days_since_join = (datetime.now() - profile['join_date']).days
        if days_since_join == 0:
            days_since_join = 1
            
        return_rate = profile['returns_count'] / len(profile['interactions']) if profile['interactions'] else 0
        
        # Calculate loyalty score (0-1)
        loyalty_score = (
            0.3 * min(days_since_join / 365, 1) +  # Time as customer
            0.2 * (1 - return_rate) +  # Return rate
            0.3 * profile['engagement_score'] +  # Engagement
            0.2 * (1 - profile['frustration_level'])  # Satisfaction
        )
        
        profile['loyalty_score'] = min(1, max(0, loyalty_score))

    def _calculate_message_repetition(self, interactions: List[Dict]) -> float:
        """Calculate how repetitive recent messages are"""
        if len(interactions) < 2:
            return 0
            
        messages = [i['message'].lower() for i in interactions]
        unique_messages = len(set(messages))
        return 1 - (unique_messages / len(messages))

    def get_customer_status(self, customer_id: str) -> Dict:
        """Get current status and recommendations for a customer"""
        if customer_id not in self.customer_profiles:
            return {
                'status': 'new_customer',
                'recommendations': ['Welcome the customer and gather initial information']
            }

        profile = self.customer_profiles[customer_id]
        
        # Determine engagement level
        engagement_level = 'low'
        for level, threshold in self.engagement_thresholds.items():
            if profile['engagement_score'] >= threshold:
                engagement_level = level
                break

        # Determine loyalty level
        loyalty_level = 'bronze'
        for level, threshold in self.loyalty_levels.items():
            if profile['loyalty_score'] >= threshold:
                loyalty_level = level
                break

        # Generate personalized recommendations
        recommendations = self._generate_recommendations(profile, engagement_level, loyalty_level)

        return {
            'customer_id': customer_id,
            'engagement_level': engagement_level,
            'loyalty_level': loyalty_level,
            'frustration_level': profile['frustration_level'],
            'days_since_join': (datetime.now() - profile['join_date']).days,
            'total_interactions': len(profile['interactions']),
            'active_days': len(profile['activity_days']),
            'recommendations': recommendations,
            'metrics': {
                'engagement_score': profile['engagement_score'],
                'loyalty_score': profile['loyalty_score'],
                'sentiment_trend': self._calculate_sentiment_trend(profile),
                'activity_trend': self._calculate_activity_trend(profile)
            }
        }

    def _generate_recommendations(self, profile: Dict, engagement_level: str, loyalty_level: str) -> List[str]:
        """Generate personalized recommendations based on customer status"""
        recommendations = []
        
        # High frustration recommendations
        if profile['frustration_level'] > 0.7:
            recommendations.append("Offer immediate assistance and personalized support")
            recommendations.append("Consider offering a discount or special offer")
            if not profile['discount_offered']:
                recommendations.append("Send a personalized discount code")
                profile['discount_offered'] = True

        # Engagement-based recommendations
        if engagement_level == 'low':
            recommendations.append("Send re-engagement email with personalized content")
            recommendations.append("Offer exclusive content or early access")
        elif engagement_level == 'medium':
            recommendations.append("Continue providing value through regular updates")
            recommendations.append("Consider implementing a loyalty program")

        # Loyalty-based recommendations
        if loyalty_level in ['gold', 'platinum']:
            recommendations.append("Offer exclusive benefits or early access to new features")
            recommendations.append("Consider implementing a referral program")
        elif loyalty_level == 'silver':
            recommendations.append("Send personalized thank you message")
            recommendations.append("Offer loyalty points for continued engagement")

        return recommendations

    def _calculate_sentiment_trend(self, profile: Dict) -> Dict:
        """Calculate sentiment trend over time"""
        if len(profile['sentiment_history']) < 2:
            return {'direction': 'neutral', 'strength': 0}
            
        recent_sentiment = np.mean(profile['sentiment_history'][-5:])
        previous_sentiment = np.mean(profile['sentiment_history'][-10:-5])
        
        difference = recent_sentiment - previous_sentiment
        direction = 'positive' if difference > 0 else 'negative' if difference < 0 else 'neutral'
        strength = abs(difference)
        
        return {
            'direction': direction,
            'strength': min(strength, 1)
        }

    def _calculate_activity_trend(self, profile: Dict) -> Dict:
        """Calculate activity trend over time"""
        if len(profile['activity_days']) < 2:
            return {'direction': 'neutral', 'strength': 0}
            
        recent_days = sorted(profile['activity_days'])[-7:]
        previous_days = sorted(profile['activity_days'])[-14:-7]
        
        recent_count = len(recent_days)
        previous_count = len(previous_days)
        
        difference = (recent_count - previous_count) / 7
        direction = 'positive' if difference > 0 else 'negative' if difference < 0 else 'neutral'
        strength = abs(difference)
        
        return {
            'direction': direction,
            'strength': min(strength, 1)
        }

    def get_customer_history(self, customer_id: str) -> Dict:
        """Get detailed customer interaction history"""
        if customer_id not in self.customer_profiles:
            return {}
            
        profile = self.customer_profiles[customer_id]
        return {
            'interactions': profile['interactions'],
            'sentiment_history': profile['sentiment_history'],
            'activity_days': list(profile['activity_days']),
            'metrics_over_time': {
                'engagement': self._calculate_metric_over_time(profile, 'engagement_score'),
                'loyalty': self._calculate_metric_over_time(profile, 'loyalty_score'),
                'frustration': self._calculate_metric_over_time(profile, 'frustration_level')
            }
        }

    def _calculate_metric_over_time(self, profile: Dict, metric: str) -> List[Dict]:
        """Calculate how a metric has changed over time"""
        # This is a simplified version - in a real implementation,
        # you would track metric values over time
        return [
            {
                'timestamp': interaction['timestamp'].isoformat(),
                'value': profile[metric]
            }
            for interaction in profile['interactions']
        ] 