-- Create database
CREATE DATABASE customer_service;

-- Connect to database
\c customer_service;

-- Create tables
CREATE TABLE customer_interactions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    ticket_id VARCHAR(50) UNIQUE,
    complaint TEXT,
    response TEXT,
    analysis JSONB,
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE customer_profiles (
    customer_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    is_vip BOOLEAN DEFAULT FALSE,
    lifetime_value DECIMAL(10,2),
    preferred_language VARCHAR(10),
    timezone VARCHAR(50),
    communication_channel VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE interaction_metrics (
    id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(50) REFERENCES customer_interactions(ticket_id),
    sentiment_score DECIMAL(5,2),
    emotion_scores JSONB,
    resolution_time INTERVAL,
    customer_satisfaction_score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_customer_interactions_customer_id ON customer_interactions(customer_id);
CREATE INDEX idx_customer_interactions_timestamp ON customer_interactions(timestamp);
CREATE INDEX idx_customer_profiles_email ON customer_profiles(email);
CREATE INDEX idx_interaction_metrics_ticket_id ON interaction_metrics(ticket_id);

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_customer_profiles_updated_at
    BEFORE UPDATE ON customer_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 