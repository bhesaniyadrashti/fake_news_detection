import pandas as pd
import re
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# NewsAPI Configuration
NEWS_API_KEY = "77c9c512b29c48769c492b6584503b1d"  # Replace with your actual API key from https://newsapi.org
NEWS_API_BASE_URL = "https://newsapi.org/v2"

class ModelStorage:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.accuracy = None
        self.y_test = None
        self.y_pred = None
        self.report_dict = None
        self.cm = None

model_storage = ModelStorage()

# Add CORS headers
@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    """Clean and normalize input text."""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip())

def fetch_news(category=None, country=None, query=None, page_size=10):
    """Fetch news articles from NewsAPI."""
    try:
        # Use everything endpoint for better coverage
        endpoint = f"{NEWS_API_BASE_URL}/everything"
        params = {
            'apiKey': NEWS_API_KEY,
            'pageSize': page_size
        }
        
        # Build search query
        search_terms = []
        
        if query:
            search_terms.append(query)
        elif country:
            # Convert country code to country name for better search results
            country_names = {
                'us': 'United States',
                'in': 'India', 
                'gb': 'United Kingdom',
                'ca': 'Canada',
                'au': 'Australia',
                'de': 'Germany',
                'fr': 'France',
                'jp': 'Japan',
                'cn': 'China'
            }
            country_name = country_names.get(country.lower(), country.upper())
            search_terms.append(country_name)
        else:
            search_terms.append('news')  # Default search term
        
        if category and not query:
            search_terms.append(category)
        
        params['q'] = ' '.join(search_terms)
        
        # Sort by relevance for better results
        params['sortBy'] = 'relevancy'
        
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'ok':
            return {'success': False, 'error': data.get('message', 'Unknown API error')}
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', ''),
                'publishedAt': article.get('publishedAt', ''),
                'urlToImage': article.get('urlToImage', '')
            })
        
        return {
            'success': True,
            'articles': articles,
            'totalResults': data.get('totalResults', 0)
        }
        
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Network error: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Error fetching news: {str(e)}'}

def get_status_context():
    """Return model performance metrics for system status display."""
    if (model_storage.cm is None or model_storage.accuracy is None or 
        model_storage.report_dict is None or model_storage.y_test is None):
        return None

    # Calculate training data info
    total_samples = len(model_storage.y_test)
    training_samples = total_samples * 4
    report = model_storage.report_dict
    
    # Build comprehensive status response
    return {
        'status': 'Active',
        'model_accuracy': float(model_storage.accuracy),
        'accuracy_percent': f"{model_storage.accuracy:.2%}",
        
        # Confusion Matrix values
        'confusion_matrix': {
            'true_negatives': int(model_storage.cm[0][0]),
            'false_positives': int(model_storage.cm[0][1]), 
            'false_negatives': int(model_storage.cm[1][0]),
            'true_positives': int(model_storage.cm[1][1])
        },
        'tn': int(model_storage.cm[0][0]),
        'fp': int(model_storage.cm[0][1]),
        'fn': int(model_storage.cm[1][0]),
        'tp': int(model_storage.cm[1][1]),
        
        # Classification Report metrics
        'classification_report': {
            'fake': {
                'precision': float(report['0']['precision']),
                'recall': float(report['0']['recall']),
                'f1_score': float(report['0']['f1-score']),
                'support': int(report['0']['support'])
            },
            'real': {
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'f1_score': float(report['1']['f1-score']),
                'support': int(report['1']['support'])
            },
            'macro_avg': {
                'precision': float(report['macro avg']['precision']),
                'recall': float(report['macro avg']['recall']),
                'f1_score': float(report['macro avg']['f1-score'])
            },
            'weighted_avg': {
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'f1_score': float(report['weighted avg']['f1-score'])
            }
        },
        
        # Training Data Information
        'training_data': {
            'total_samples': int(training_samples),
            'test_samples': int(total_samples),
            'training_ratio': '80/20',
            'features_extracted': int(model_storage.vectorizer.get_feature_names_out().shape[0]) if model_storage.vectorizer else 0
        },
        
        # Additional metrics
        'precision': float(report['macro avg']['precision']),
        'recall': float(report['macro avg']['recall']),
        'f1_score': float(report['macro avg']['f1-score']),
        'false_positives': int(model_storage.cm[0][1]),
        'false_negatives': int(model_storage.cm[1][0]),
        'true_positives': int(model_storage.cm[1][1]),
        'true_negatives': int(model_storage.cm[0][0])
    }

def train_model():
    """Train the fake news detection model."""
    try:
        df = pd.read_csv('./test1/fake_news.csv')
        print(f"Dataset loaded: {len(df)} articles")
        
        # Remove missing values
        print("Cleaning data...")
        df = df.dropna(subset=['title', 'text', 'label'])
        df = df.drop_duplicates()
        print(f"Data cleaned: {len(df)} articles after cleanup")
        
        # Clean the data
        print("Extracting content...")
        df['content'] = (df['title'] + ' ' + df['text']).apply(clean_text)
        
        # Train/Test Split
        print("Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['content'], df['label'], test_size=0.2, random_state=42
        )
        print(f"Split data: {len(X_train)} training, {len(X_test)} test")
        
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        print(f"Extracted {X_train_tfidf.shape[1]} features")
        
        # Train the Model
        print("Training Logistic Regression model...")
        model = LogisticRegression(max_iter=300)
        model.fit(X_train_tfidf, y_train)
        print("Model trained successfully")
        
        # Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store in model storage
        model_storage.model = model
        model_storage.vectorizer = vectorizer
        model_storage.accuracy = accuracy
        model_storage.y_test = y_test
        model_storage.y_pred = y_pred
        model_storage.report_dict = report_dict
        model_storage.cm = cm
        
        print(f"Model Accuracy: {accuracy:.2%}")
        return True
    
    except FileNotFoundError:
        print("Make sure './test1/fake_news.csv' exists!")
        return False
    except Exception as e:
        print(f"Training error: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET', 'POST'])
def index():
    """Home page."""
    status = get_status_context()

    if request.method == 'POST':
        # Handle form submission from analyze button
        try:
            title = str(request.form.get('title', '')).strip()
            content = str(request.form.get('content', '')).strip()
            
            if not title and not content:
                return render_template('index.html', error='Please enter title or content', status=status)
            
            if model_storage.model is None or model_storage.vectorizer is None:
                return render_template('index.html', error='Model not initialized', status=status)
            
            # Combine and process
            combined_input = f"{title} {content}"
            cleaned_content = clean_text(combined_input)
            
            # Transform and predict
            content_tfidf = model_storage.vectorizer.transform([cleaned_content])
            prediction = model_storage.model.predict(content_tfidf)[0]
            confidence = model_storage.model.predict_proba(content_tfidf)[0]
            
            result = {
                'prediction': 'FAKE' if prediction == 0 else 'REAL',
                'fake_score': float(confidence[0]),
                'real_score': float(confidence[1]),
                'confidence': float(max(confidence))
            }
            
            return render_template('index.html', result=result, status=status)
            
        except Exception as e:
            return render_template('index.html', error=f'Analysis error: {str(e)}', status=status)
    
    return render_template('index.html', status=status)

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict fake news."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'success': False, 'error': 'Invalid JSON'}), 400
        
        title = str(data.get('title', '')).strip()
        content = str(data.get('content', '')).strip()
        
        print(f"  Title: {title[:50]}...")
        print(f"  Content: {content[:50]}...")
        
        if not title and not content:
            print("  Empty input")
            return jsonify({'success': False, 'error': 'Please enter title or content'}), 400
        
        if model_storage.model is None or model_storage.vectorizer is None:
            print("  Model not initialized")
            return jsonify({'success': False, 'error': 'Model not initialized'}), 500
        
        combined_input = f"{title} {content}"
        cleaned_content = clean_text(combined_input)
        
        content_tfidf = model_storage.vectorizer.transform([cleaned_content])
        prediction = model_storage.model.predict(content_tfidf)[0]
        confidence = model_storage.model.predict_proba(content_tfidf)[0]
        
        result = {
            'success': True,
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'fake_score': float(confidence[0]),
            'real_score': float(confidence[1]),
            'confidence': float(max(confidence))
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/performance', methods=['GET', 'OPTIONS'])
def get_performance():
    """Get model performance metrics - kept for API compatibility."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        status_data = get_status_context()
        if status_data is None:
            return jsonify({'success': False, 'error': 'Model not fully trained yet'}), 500
        
        response_payload = {'success': True}
        response_payload.update(status_data)
        return jsonify(response_payload), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/news', methods=['GET', 'OPTIONS'])
def get_news():
    """Get news articles from NewsAPI."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        category = request.args.get('category')
        country = request.args.get('country')
        query = request.args.get('query')
        page_size = min(int(request.args.get('pageSize', 10)), 20)
        
        result = fetch_news(category=category, country=country, query=query, page_size=page_size)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid pageSize parameter'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/analyze-news', methods=['POST', 'OPTIONS'])
def analyze_news():
    """Analyze a news article for authenticity."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'success': False, 'error': 'Invalid JSON'}), 400
        
        title = str(data.get('title', '')).strip()
        content = str(data.get('content', '')).strip()
        
        if not title and not content:
            return jsonify({'success': False, 'error': 'Please provide title or content'}), 400
        
        if model_storage.model is None or model_storage.vectorizer is None:
            return jsonify({'success': False, 'error': 'Model not initialized'}), 500
        
        combined_input = f"{title} {content}"
        cleaned_content = clean_text(combined_input)
        
        content_tfidf = model_storage.vectorizer.transform([cleaned_content])
        prediction = model_storage.model.predict(content_tfidf)[0]
        confidence = model_storage.model.predict_proba(content_tfidf)[0]
        
        result = {
            'success': True,
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'fake_score': float(confidence[0]),
            'real_score': float(confidence[1]),
            'confidence': float(max(confidence))
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/status', methods=['GET', 'OPTIONS'])
def get_status():
    """Provide the confusion-matrix driven status summary."""
    if request.method == 'OPTIONS':
        return '', 200
    try:
        status_payload = get_status_context()
        if status_payload is None:
            return jsonify({'success': False, 'error': 'Model not fully trained yet'}), 500

        response_payload = {'success': True}
        response_payload.update(status_payload)
        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════
# START APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("FAKE NEWS DETECTION SYSTEM")
    
    if train_model():
        print("Model ready!")
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='localhost', port=5000, use_reloader=True)
    else:
        print("Failed to train model. Check your data file.")
   



