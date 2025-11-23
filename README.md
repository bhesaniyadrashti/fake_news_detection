# Fake News Detection System with Real News Analysis

An advanced machine learning-powered fake news detection system that can analyze both user-provided content and real news articles from NewsAPI.

## Features

- **Real-time News Analysis**: Fetch and analyze real news articles from NewsAPI.org
- **AI-Powered Detection**: Uses machine learning to classify news as REAL or FAKE
- **Confidence Scoring**: Provides detailed probability scores for predictions
- **Multiple Categories**: Filter news by category (business, technology, health, etc.)
- **Country Selection**: Get news from different countries
- **Search Functionality**: Search for specific topics
- **Beautiful UI**: Modern, responsive interface with smooth animations

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get NewsAPI Key

1. Go to [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Open `app.py` and replace the placeholder:

```python
NEWS_API_KEY = "your_newsapi_key_here"  # Replace with your actual API key
```

### 3. Run the Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

## Usage

### Analyze Custom News
1. Go to the "Analyze News" tab
2. Enter a news title and/or content
3. Click "Analyze" to get AI-powered authenticity results

### Real News Analysis
1. Go to the "Real News" tab
2. Select category and country (optional)
3. Enter search terms (optional)
4. Click "📰 Fetch News" to get latest articles
5. Click "🔍 Analyze" on any article to check its authenticity

## API Endpoints

- `GET /api/news` - Fetch news articles
- `POST /api/analyze-news` - Analyze news for authenticity
- `POST /api/predict` - Predict fake/real news
- `GET /api/status` - Get system status and metrics

## Technical Details

- **Backend**: Flask with Python
- **Machine Learning**: scikit-learn Logistic Regression
- **Text Processing**: TF-IDF Vectorization
- **News Source**: NewsAPI.org
- **Frontend**: HTML5, CSS3, JavaScript

## Model Performance

The system is trained on thousands of real and fake news articles and achieves high accuracy in distinguishing between them. Performance metrics are available in the "Performance" tab.

## Privacy

All news analysis is performed locally on your device. No personal data is transmitted to external servers except for fetching news articles from NewsAPI.

## License

This project is for educational and research purposes. Always verify information from multiple reliable sources.
