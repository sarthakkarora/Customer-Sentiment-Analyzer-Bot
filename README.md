# AI Customer Experience Specialist

An intelligent system for handling customer complaints and generating personalized responses using advanced AI capabilities.

## Features

- **Advanced AI Integration**
  - OpenAI-powered response generation
  - NLP-based complaint analysis
  - Context-aware compensation strategies
  - Sentiment analysis and emotional intelligence

- **Smart Response Generation**
  - Dynamic tone adjustment based on customer sentiment
  - Predictive resolutions for common issues
  - LTV-based compensation and actions
  - Personalized response generation
  - VIP customer recognition
  - Automated ticket ID generation

- **Natural Language Processing**
  - Entity extraction
  - Key phrase identification
  - Contextual understanding
  - Conversation history tracking

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from customer_experience_specialist import CustomerInfo, CustomerExperienceSpecialist

# Create a customer instance
customer = CustomerInfo(
    name="John Doe",
    complaint="My package was delivered late and the product was defective",
    sentiment="angry",
    is_vip=True,
    lifetime_value=1500.00,
    order_history={"total_orders": 10, "return_rate": 0.05},
    previous_complaints=["Late delivery last month"],
    last_interaction_notes="Customer prefers email communication"
)

# Initialize the specialist
specialist = CustomerExperienceSpecialist(
    company_name="Your Company",
    agent_name="Your Name",
    agent_number="+1-555-123-4567"
)

# Generate the response
response = specialist.generate_response(customer)
print(response)
```

## AI-Enhanced Response Format

The system generates sophisticated responses using AI:

1. **Complaint Analysis**
   - NLP-based entity extraction
   - Sentiment analysis
   - Context understanding

2. **Personalized Response**
   - AI-generated empathetic opening
   - Context-aware solutions
   - Dynamic compensation strategies

3. **Follow-up**
   - Personal contact information
   - Special offers
   - Conversation history tracking

## Customization

You can customize the AI behavior by modifying:

1. **OpenAI Configuration**
   - Model selection (GPT-3.5 or GPT-4)
   - Temperature settings
   - Token limits

2. **Response Templates**
   - System prompts
   - Compensation strategies
   - Follow-up procedures

3. **NLP Settings**
   - Entity recognition rules
   - Key phrase extraction
   - Sentiment analysis parameters

## Advanced Features

- **Conversation History**: Tracks all customer interactions
- **Context Analysis**: Deep understanding of complaint context
- **Dynamic Compensation**: AI-powered compensation strategies
- **Multi-turn Support**: Maintains context across interactions

## License

MIT License 