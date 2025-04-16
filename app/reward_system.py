from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
from loguru import logger

class RewardSystem:
    def __init__(self):
        self.reward_types = {
            'discount': {
                'bronze': {'min': 5, 'max': 10},
                'silver': {'min': 10, 'max': 15},
                'gold': {'min': 15, 'max': 20},
                'platinum': {'min': 20, 'max': 25}
            },
            'free_shipping': {
                'bronze': 0.1,  # 10% chance
                'silver': 0.3,  # 30% chance
                'gold': 0.5,    # 50% chance
                'platinum': 0.7  # 70% chance
            },
            'early_access': {
                'bronze': 0.0,  # No access
                'silver': 0.2,  # 20% chance
                'gold': 0.4,    # 40% chance
                'platinum': 0.6  # 60% chance
            },
            'exclusive_content': {
                'bronze': 0.0,  # No access
                'silver': 0.1,  # 10% chance
                'gold': 0.3,    # 30% chance
                'platinum': 0.5  # 50% chance
            }
        }
        
        self.reward_cooldowns = {
            'discount': timedelta(days=30),
            'free_shipping': timedelta(days=15),
            'early_access': timedelta(days=7),
            'exclusive_content': timedelta(days=3)
        }
        
        self.customer_rewards = {}

    def update_customer_rewards(self, customer_id: str, loyalty_level: str, engagement_score: float):
        """Update customer's reward eligibility"""
        if customer_id not in self.customer_rewards:
            self.customer_rewards[customer_id] = {
                'last_rewards': {},
                'total_rewards_claimed': 0,
                'reward_points': 0
            }
        
        # Add reward points based on engagement
        points_to_add = int(engagement_score * 10)
        self.customer_rewards[customer_id]['reward_points'] += points_to_add
        
        # Check for milestone rewards
        self._check_milestone_rewards(customer_id)

    def _check_milestone_rewards(self, customer_id: str):
        """Check and award milestone rewards"""
        rewards = self.customer_rewards[customer_id]
        points = rewards['reward_points']
        
        milestones = {
            100: {'type': 'discount', 'value': 15},
            250: {'type': 'free_shipping', 'value': 1},
            500: {'type': 'early_access', 'value': 1},
            1000: {'type': 'exclusive_content', 'value': 1}
        }
        
        for milestone, reward in milestones.items():
            if points >= milestone and f'milestone_{milestone}' not in rewards['last_rewards']:
                self._award_reward(customer_id, reward['type'], reward['value'])
                rewards['last_rewards'][f'milestone_{milestone}'] = datetime.now()

    def generate_personalized_rewards(self, customer_id: str, loyalty_level: str, 
                                    frustration_level: float, engagement_score: float) -> List[Dict]:
        """Generate personalized rewards based on customer status"""
        if customer_id not in self.customer_rewards:
            self.customer_rewards[customer_id] = {
                'last_rewards': {},
                'total_rewards_claimed': 0,
                'reward_points': 0
            }
        
        rewards = []
        current_time = datetime.now()
        
        # Check each reward type
        for reward_type, level_chances in self.reward_types.items():
            # Skip if on cooldown
            last_reward = self.customer_rewards[customer_id]['last_rewards'].get(reward_type)
            if last_reward and (current_time - last_reward) < self.reward_cooldowns[reward_type]:
                continue
            
            # Calculate chance based on loyalty level and current status
            base_chance = level_chances.get(loyalty_level, 0)
            
            # Increase chance if frustrated
            if frustration_level > 0.7:
                base_chance *= 1.5
            
            # Increase chance based on engagement
            base_chance *= (1 + engagement_score)
            
            # Random chance to award
            if random.random() < min(base_chance, 0.95):  # Cap at 95% chance
                reward = self._generate_reward(reward_type, loyalty_level)
                if reward:
                    rewards.append(reward)
                    self._award_reward(customer_id, reward_type, reward['value'])
        
        return rewards

    def _generate_reward(self, reward_type: str, loyalty_level: str) -> Optional[Dict]:
        """Generate a specific reward based on type and loyalty level"""
        if reward_type == 'discount':
            range_config = self.reward_types['discount'][loyalty_level]
            value = random.randint(range_config['min'], range_config['max'])
            return {
                'type': 'discount',
                'value': value,
                'code': f'DISC{random.randint(1000, 9999)}',
                'expires_in': 7  # days
            }
        elif reward_type == 'free_shipping':
            return {
                'type': 'free_shipping',
                'value': 1,
                'code': f'SHIP{random.randint(1000, 9999)}',
                'expires_in': 3  # days
            }
        elif reward_type == 'early_access':
            return {
                'type': 'early_access',
                'value': 1,
                'code': f'EA{random.randint(1000, 9999)}',
                'expires_in': 1  # day
            }
        elif reward_type == 'exclusive_content':
            return {
                'type': 'exclusive_content',
                'value': 1,
                'code': f'EXCL{random.randint(1000, 9999)}',
                'expires_in': 1  # day
            }
        return None

    def _award_reward(self, customer_id: str, reward_type: str, value: float):
        """Record the awarded reward"""
        self.customer_rewards[customer_id]['last_rewards'][reward_type] = datetime.now()
        self.customer_rewards[customer_id]['total_rewards_claimed'] += 1

    def get_reward_history(self, customer_id: str) -> Dict:
        """Get customer's reward history"""
        if customer_id not in self.customer_rewards:
            return {
                'total_rewards_claimed': 0,
                'reward_points': 0,
                'last_rewards': {},
                'available_rewards': []
            }
        
        rewards = self.customer_rewards[customer_id]
        return {
            'total_rewards_claimed': rewards['total_rewards_claimed'],
            'reward_points': rewards['reward_points'],
            'last_rewards': {
                k: v.isoformat() for k, v in rewards['last_rewards'].items()
            },
            'available_rewards': self._get_available_rewards(customer_id)
        }

    def _get_available_rewards(self, customer_id: str) -> List[Dict]:
        """Get currently available rewards for the customer"""
        if customer_id not in self.customer_rewards:
            return []
        
        available = []
        current_time = datetime.now()
        
        for reward_type, last_time in self.customer_rewards[customer_id]['last_rewards'].items():
            if (current_time - last_time) >= self.reward_cooldowns[reward_type]:
                available.append({
                    'type': reward_type,
                    'available_since': last_time.isoformat(),
                    'cooldown_remaining': 0
                })
            else:
                remaining = self.reward_cooldowns[reward_type] - (current_time - last_time)
                available.append({
                    'type': reward_type,
                    'available_since': None,
                    'cooldown_remaining': remaining.total_seconds()
                })
        
        return available 