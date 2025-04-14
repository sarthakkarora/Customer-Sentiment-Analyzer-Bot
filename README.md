# Customer Experience Prediction System

A sophisticated system for analyzing and predicting customer experience using advanced machine learning and real-time processing capabilities.

## Features

- **Real-time Processing**
  - Kafka-based message queuing
  - WebSocket notifications
  - Redis caching
  - Elasticsearch storage

- **Advanced Analytics**
  - Customer segmentation
  - Trend analysis
  - Correlation analysis
  - Time series forecasting
  - Interactive visualizations

- **Automated Reporting**
  - Daily reports
  - Weekly reports
  - Custom reports
  - PDF generation
  - Interactive dashboards

- **Machine Learning Models**
  - Sentiment analysis
  - Emotion detection
  - Churn prediction
  - Response quality assessment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sarthakkarora/customer-sentiment-analyzer-bot.git
cd customer-sentiment-analyzer-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Install wkhtmltopdf (required for PDF generation):
```bash
# On macOS
brew install wkhtmltopdf

# On Ubuntu/Debian
sudo apt-get install wkhtmltopdf

# On Windows
# Download from https://wkhtmltopdf.org/downloads.html
```

## Usage

1. Start the real-time processor:
```bash
python real_time_processor.py
```

2. Start the analytics service:
```bash
python analytics.py
```

3. Generate reports:
```bash
python report_generator.py
```

## Project Structure

```
.
├── analytics.py           # Advanced analytics module
├── config.py             # Configuration settings
├── customer_experience_specialist.py  # Main processing logic
├── database_schema.sql   # Database schema
├── ml_models.py          # Machine learning models
├── monitoring.py         # System monitoring
├── real_time_processor.py # Real-time processing
├── report_generator.py   # Report generation
├── requirements.txt      # Dependencies
├── templates/            # HTML templates
│   ├── daily_report.html
│   ├── weekly_report.html
│   └── custom_report.html
└── .env.example         # Environment variables template
```

## API Documentation

### Real-time Processing

- **POST /api/complaints**
  - Process customer complaints in real-time
  - Returns sentiment analysis and risk assessment

- **POST /api/responses**
  - Process agent responses
  - Returns quality assessment and recommendations

### Analytics

- **GET /api/segments**
  - Get customer segments analysis
  - Returns segment characteristics and metrics

- **GET /api/trends**
  - Get trend analysis
  - Returns time series data and forecasts

### Reports

- **GET /api/reports/daily**
  - Generate daily report
  - Returns PDF report

- **GET /api/reports/weekly**
  - Generate weekly report
  - Returns PDF report

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow for machine learning capabilities
- Elasticsearch for data storage and search
- Kafka for real-time message processing
- Plotly for interactive visualizations

