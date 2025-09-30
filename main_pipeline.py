from logging import Logger
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import requests
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import warnings
import os
import sys
import traceback
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import time
from colorama import Fore, Style, init

warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: Logger = logging.getLogger(__name__)

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer

    SBERT_AVAILABLE = True
    logger.info("sentence-transformers available")
except ImportError:
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    SBERT_AVAILABLE = False


# Define SBERTTransformer at top level
class SBERTTransformer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"SBERT model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {str(e)}")
            raise

    def transform(self, sentences):
        try:
            if not sentences:
                return np.array([])
            return self.model.encode(sentences)
        except Exception as e:
            logger.error(f"Error in SBERT transform: {str(e)}")
            raise

    def fit(self, X, y=None):
        return self


# MDA Sentiment Model Class
class MDASentimentModel:
    """PyTorch model for analyzing management tone in MDA sections"""

    def __init__(self, model_path):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load the model architecture and weights
            self.model = self._load_model(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model.eval()  # Set to evaluation mode
            logger.info(f"MDA sentiment model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load MDA model: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None

    def _load_model(self, model_path):
        """Load the trained PyTorch model"""

        # Define the model architecture (updated to match your checkpoint)
        class BERTSentiment(nn.Module):
            def __init__(self):
                super(BERTSentiment, self).__init__()
                self.bert = AutoModel.from_pretrained('bert-base-uncased')
                self.dropout = nn.Dropout(0.3)
                # Updated to 5 classes to match your checkpoint
                self.classifier = nn.Linear(self.bert.config.hidden_size, 5)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                output = self.dropout(pooled_output)
                return self.classifier(output)

        # Load the model
        model = BERTSentiment()

        # Load state dict with strict=False to handle size mismatches
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle embedding size mismatch
        if 'bert.embeddings.word_embeddings.weight' in state_dict:
            current_embed = model.bert.embeddings.word_embeddings.weight
            saved_embed = state_dict['bert.embeddings.word_embeddings.weight']

            if current_embed.shape != saved_embed.shape:
                logger.warning(f"Embedding size mismatch: current {current_embed.shape}, saved {saved_embed.shape}")
                # Use the first 30522 tokens from the saved embeddings
                min_size = min(current_embed.shape[0], saved_embed.shape[0])
                state_dict['bert.embeddings.word_embeddings.weight'] = saved_embed[:min_size, :]

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def predict(self, texts):
        """Predict sentiment for a list of texts"""
        if not self.model or not texts:
            return [], []

        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]

            # Filter out empty texts
            valid_texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
            if not valid_texts:
                return [], []

            # Tokenize the texts
            encodings = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(predictions, dim=1)

            # Convert to labels (updated for 5 classes)
            label_map = {0: 'very_negative', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'very_positive'}
            sentiments = [label_map[c.item()] for c in predicted_classes]
            confidences = [c.item() for c in confidences]

            return sentiments, confidences

        except Exception as e:
            logger.error(f"Error in MDA sentiment prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return [], []

    def is_available(self):
        """Check if MDA model is available"""
        return self.model is not None


class EnhancedPositionTradingSystem:
    """Enhanced Position Trading System for Indian Markets with Long-term Focus"""

    def __init__(self, model_path="D:/Python_files/models/sentiment_pipeline.joblib",
                 mda_model_path="D:/best_model_fold_1.pth",
                 news_api_key=None):

        try:
            self.sentiment_pipeline = None
            self.vectorizer = None
            self.model = None
            self.label_encoder = None
            self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY") or "dd33ebe105ea4b02a3b7e77bc4a93d01"

            # Model status tracking
            self.model_loaded = False
            self.model_type = "None"

            # Initialize MDA sentiment model
            self.mda_sentiment_model = None
            self.mda_available = False

            if mda_model_path and os.path.exists(mda_model_path):
                try:
                    self.mda_sentiment_model = MDASentimentModel(mda_model_path)
                    self.mda_available = self.mda_sentiment_model.is_available()
                    if self.mda_available:
                        logger.info("MDA sentiment model initialized successfully")
                    else:
                        logger.warning("MDA sentiment model failed to initialize")
                except Exception as e:
                    logger.error(f"Error initializing MDA sentiment model: {str(e)}")
                    self.mda_sentiment_model = None
                    self.mda_available = False
            else:
                logger.warning(f"MDA model path not found: {mda_model_path}")
                self.mda_sentiment_model = None
                self.mda_available = False

            # Position Trading parameters (Long-term focus)
            self.position_trading_params = {
                'min_holding_period': 90,  # 3 months minimum
                'max_holding_period': 1095,  # 3 years maximum
                'risk_per_trade': 0.01,  # 1% risk per trade (conservative)
                'max_portfolio_risk': 0.05,  # 5% max portfolio risk
                'profit_target_multiplier': 4.0,  # 4:1 risk-reward ratio
                'max_positions': 12,  # Maximum 12 positions
                'min_position_size': 0.02,  # Minimum 2% per position
                'max_position_size': 0.10,  # Maximum 10% per position
                'fundamental_weight': 0.45,  # 45% weight to fundamentals
                'technical_weight': 0.35,  # 35% weight to technicals
                'sentiment_weight': 0.10,  # 10% weight to news sentiment
                'mda_weight': 0.10,  # 10% weight to MDA sentiment (management tone)
            }

            # Validate trading parameters
            self._validate_trading_params()

            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not provided. Using sample news for sentiment analysis.")
            else:
                logger.info("News API key available. Will fetch real news articles.")

            # Load your trained SBERT sentiment model
            self.load_trained_sbert_model(model_path)

            # Initialize comprehensive stock database
            self.initialize_stock_database()

            logger.info("EnhancedPositionTradingSystem initialized successfully")
            logger.info(f"MDA Model Available: {self.mda_available}")
            logger.info(f"News Sentiment Model: {self.model_type}")

        except Exception as e:
            logger.error(f"Error initializing EnhancedPositionTradingSystem: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _validate_trading_params(self):
        """Validate position trading parameters"""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier', 'max_positions']

            for param in required_params:
                if param not in self.position_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")

                value = self.position_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")

            # Additional validation
            if self.position_trading_params['min_holding_period'] >= self.position_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")

            if self.position_trading_params['risk_per_trade'] > 0.05:  # 5% max risk per trade
                raise ValueError("risk_per_trade cannot exceed 5% for position trading")

            # Validate weights sum to 1.0
            total_weight = (self.position_trading_params['fundamental_weight'] +
                            self.position_trading_params['technical_weight'] +
                            self.position_trading_params['sentiment_weight'] +
                            self.position_trading_params['mda_weight'])

            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point differences
                logger.warning(f"Scoring weights don't sum to 1.0: {total_weight:.3f}")

            logger.info("Position trading parameters validated successfully")

        except Exception as e:
            logger.error(f"Error validating trading parameters: {str(e)}")
            raise

    def get_sample_mda_text(self, symbol):
        """Generate sample MDA-style text for demonstration"""
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)
            sector = stock_info.get("sector", "business")

            mda_samples = [
                f"Management remains optimistic about {company_name}'s growth prospects in the {sector} sector. Our strategic initiatives are yielding positive results.",
                f"The company has successfully navigated market challenges and positioned itself for sustainable growth. We are confident in our operational efficiency improvements.",
                f"{company_name} has demonstrated resilience in a dynamic market environment. Management believes the current strategy will drive long-term value creation.",
                f"We are pleased with the progress made in our key business segments. The management team is focused on executing our strategic roadmap effectively.",
                f"Market conditions present both opportunities and challenges. Management is committed to maintaining operational excellence while pursuing growth opportunities.",
                f"The company's strong fundamentals provide a solid foundation for future expansion. We remain cautious yet optimistic about market developments.",
                f"Management has implemented several cost optimization measures while investing in growth initiatives. We expect these efforts to bear fruit in the coming quarters.",
                f"Our focus on innovation and customer satisfaction continues to differentiate {company_name} in the competitive landscape.",
                f"The management team is confident that our strategic partnerships and operational improvements will enhance shareholder value.",
                f"Despite market volatility, we maintain a positive outlook for our core business segments and expect sustained performance improvement."
            ]

            return mda_samples
        except Exception as e:
            logger.error(f"Error generating sample MDA text for {symbol}: {str(e)}")
            return [f"Management discussion for {symbol}"]

    def analyze_mda_sentiment(self, symbol):
        """Analyze MDA sentiment for a given symbol"""
        try:
            if not self.mda_available:
                logger.info("MDA model not available, using sample analysis")
                return self.get_sample_mda_analysis(symbol)

            # In a real implementation, you would fetch actual MDA text from annual reports
            # For demonstration, we'll use sample MDA-style text
            mda_texts = self.get_sample_mda_text(symbol)

            if not mda_texts:
                logger.warning(f"No MDA text available for {symbol}")
                return self.get_sample_mda_analysis(symbol)

            # Analyze sentiment using MDA model
            sentiments, confidences = self.mda_sentiment_model.predict(mda_texts)

            if not sentiments or not confidences:
                logger.warning(f"MDA sentiment analysis failed for {symbol}")
                return self.get_sample_mda_analysis(symbol)

            # Calculate aggregate sentiment score
            sentiment_scores = []
            for sentiment, confidence in zip(sentiments, confidences):
                if sentiment == 'positive':
                    sentiment_scores.append(confidence)
                elif sentiment == 'negative':
                    sentiment_scores.append(-confidence)
                else:  # neutral
                    sentiment_scores.append(0)

            # Aggregate score (0-100 scale)
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            mda_score = 50 + (avg_sentiment * 50)  # Convert to 0-100 scale
            mda_score = max(0, min(100, mda_score))

            # Sentiment distribution
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            total_count = len(sentiments)

            return {
                'mda_score': mda_score,
                'sentiment_distribution': {
                    'positive': positive_count / total_count if total_count > 0 else 0,
                    'negative': negative_count / total_count if total_count > 0 else 0,
                    'neutral': neutral_count / total_count if total_count > 0 else 0
                },
                'management_tone': self.get_management_tone_label(mda_score),
                'confidence': np.mean(confidences) if confidences else 0.5,
                'analysis_method': 'PyTorch BERT MDA Model',
                'sample_texts_analyzed': len(mda_texts)
            }

        except Exception as e:
            logger.error(f"Error in MDA sentiment analysis for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return self.get_sample_mda_analysis(symbol)

    def get_sample_mda_analysis(self, symbol):
        """Provide sample MDA analysis when model is not available"""
        # Generate a reasonable sample based on stock characteristics
        stock_info = self.get_stock_info_from_db(symbol)
        sector = stock_info.get('sector', 'Unknown')

        # Different sectors might have different management sentiment patterns
        if sector in ['Information Technology', 'Healthcare', 'Consumer Goods']:
            sample_score = np.random.uniform(55, 75)  # Generally positive
        elif sector in ['Banking', 'Financial Services']:
            sample_score = np.random.uniform(45, 65)  # Moderate
        else:
            sample_score = np.random.uniform(40, 70)  # Variable

        return {
            'mda_score': sample_score,
            'sentiment_distribution': {
                'positive': 0.4,
                'negative': 0.2,
                'neutral': 0.4
            },
            'management_tone': self.get_management_tone_label(sample_score),
            'confidence': 0.6,
            'analysis_method': 'Sample Analysis (MDA model not available)',
            'sample_texts_analyzed': 10
        }

    def get_management_tone_label(self, score):
        """Convert MDA score to management tone label"""
        if score >= 70:
            return "Very Optimistic"
        elif score >= 60:
            return "Optimistic"
        elif score >= 40:
            return "Neutral"
        elif score >= 30:
            return "Cautious"
        else:
            return "Pessimistic"

    def calculate_position_trading_score(self, data, sentiment_data, fundamentals, trends, market_analysis, sector,
                                         mda_analysis=None):
        """Calculate comprehensive position trading score with fundamental emphasis and MDA sentiment"""
        try:
            # Get weights for position trading (fundamentals-heavy)
            fundamental_weight = self.position_trading_params['fundamental_weight']
            technical_weight = self.position_trading_params['technical_weight']
            sentiment_weight = self.position_trading_params['sentiment_weight']
            mda_weight = self.position_trading_params['mda_weight']

            # Calculate individual scores
            fundamental_score = self.calculate_fundamental_score(fundamentals, sector)
            technical_score = self.calculate_technical_score_position(data)
            sentiment_score = self.calculate_sentiment_score(sentiment_data)
            trend_score = trends.get('trend_score', 50)
            sector_score = market_analysis.get('sector_score', 60)

            # MDA sentiment score
            mda_score = 50  # Default neutral score
            if mda_analysis and isinstance(mda_analysis, dict):
                mda_score = mda_analysis.get('mda_score', 50)

            # Combine scores with position trading weights
            base_score = (
                    fundamental_score * fundamental_weight +
                    technical_score * technical_weight +
                    sentiment_score * sentiment_weight +
                    mda_score * mda_weight
            )

            # Apply trend and sector modifiers
            trend_modifier = trend_score / 100  # 0 to 1
            sector_modifier = sector_score / 100  # 0 to 1

            # Final score with modifiers
            final_score = base_score * (0.7 + 0.2 * trend_modifier + 0.1 * sector_modifier)

            # Position trading specific adjustments
            # Penalize high volatility stocks
            if data is not None and not data.empty and 'Close' in data.columns:
                volatility = data['Close'].pct_change().std() * np.sqrt(252)
                if volatility > 0.4:  # High volatility
                    final_score *= 0.8
                elif volatility > 0.6:  # Very high volatility
                    final_score *= 0.6

            # Bonus for dividend-paying stocks
            div_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('expected_div_yield', 0)
            if div_yield and div_yield > 0.02:  # 2%+ dividend
                final_score *= 1.1  # 10% bonus

            # Bonus for consistent long-term performance
            if trends.get('momentum_1y', 0) > 0.15 and trends.get('momentum_6m', 0) > 0:
                final_score *= 1.05  # 5% bonus

            # MDA sentiment bonus/penalty
            if mda_analysis:
                management_tone = mda_analysis.get('management_tone', 'Neutral')
                if management_tone == 'Very Optimistic':
                    final_score *= 1.08  # 8% bonus
                elif management_tone == 'Optimistic':
                    final_score *= 1.04  # 4% bonus
                elif management_tone == 'Pessimistic':
                    final_score *= 0.92  # 8% penalty

            return min(100, max(0, final_score))

        except Exception as e:
            logger.error(f"Error calculating position trading score: {str(e)}")
            return 0

    def analyze_position_trading_stock(self, symbol, period="5y"):
        """Comprehensive position trading analysis for a single stock with MDA sentiment"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting position trading analysis for {symbol}")

            # Get extended stock data (5 years for position trading)
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Extract basic information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)
            market_cap_category = stock_info.get('market_cap', 'Unknown')

            # Current market data
            try:
                current_price = data['Close'].iloc[-1]
                if len(data) >= 2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                else:
                    price_change = 0
                    price_change_pct = 0
            except Exception as e:
                logger.error(f"Error calculating price changes: {str(e)}")
                return None

            # Fundamental analysis (key for position trading)
            try:
                fundamentals = self.analyze_fundamental_metrics(final_symbol, info)
            except Exception as e:
                logger.error(f"Error in fundamental analysis: {str(e)}")
                fundamentals = {}

            # Long-term trend analysis
            try:
                trends = self.analyze_long_term_trends(data)
            except Exception as e:
                logger.error(f"Error in trend analysis: {str(e)}")
                trends = {'trend_score': 50}

            # Market cycle and sector analysis
            try:
                market_analysis = self.analyze_market_cycles(final_symbol, data)
            except Exception as e:
                logger.error(f"Error in market analysis: {str(e)}")
                market_analysis = {'sector_score': 60}

            # News sentiment analysis using your trained model
            try:
                sentiment_results = self.analyze_news_sentiment(final_symbol)
            except Exception as e:
                logger.error(f"Error in news sentiment analysis: {str(e)}")
                sentiment_results = ([], [], [], "Error", "Error")

            # MDA sentiment analysis
            try:
                mda_analysis = self.analyze_mda_sentiment(final_symbol)
                logger.info(
                    f"MDA analysis for {symbol}: Score={mda_analysis.get('mda_score', 0):.1f}, Tone={mda_analysis.get('management_tone', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error in MDA sentiment analysis: {str(e)}")
                mda_analysis = self.get_sample_mda_analysis(final_symbol)

            # Risk metrics
            try:
                risk_metrics = self.calculate_risk_metrics(data)
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {str(e)}")
                risk_metrics = {'volatility': 0.3, 'atr': current_price * 0.02}

            # Position trading score (fundamental-heavy with MDA sentiment)
            try:
                position_score = self.calculate_position_trading_score(
                    data, sentiment_results, fundamentals, trends, market_analysis, sector, mda_analysis
                )
            except Exception as e:
                logger.error(f"Error calculating position score: {str(e)}")
                position_score = 0

            # Position trading plan
            try:
                trading_plan = self.generate_position_trading_plan(
                    data, position_score, risk_metrics, fundamentals, trends
                )
            except Exception as e:
                logger.error(f"Error generating trading plan: {str(e)}")
                trading_plan = {'entry_signal': 'ERROR', 'entry_strategy': 'Analysis failed'}

            # Compile comprehensive results
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'market_cap_category': market_cap_category,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,

                # Fundamental metrics (key for position trading)
                'fundamentals': fundamentals,
                'fundamental_score': self.calculate_fundamental_score(fundamentals, sector),

                # Long-term trends
                'trends': trends,
                'trend_score': trends.get('trend_score', 50),

                # Market and sector analysis
                'market_analysis': market_analysis,

                # Technical indicators (with longer periods)
                'rsi_30': self.calculate_rsi(data['Close'], period=30).iloc[-1] if len(data) >= 30 else None,
                'ma_50': trends.get('ma_50'),
                'ma_100': trends.get('ma_100'),
                'ma_200': trends.get('ma_200'),

                # News sentiment analysis
                'sentiment': {
                    'scores': sentiment_results[0] if sentiment_results else [],
                    'articles': sentiment_results[1] if sentiment_results else [],
                    'confidence': sentiment_results[2] if sentiment_results else [],
                    'method': sentiment_results[3] if sentiment_results else "Error",
                    'source': sentiment_results[4] if sentiment_results else "Error",
                    'sentiment_summary': self.get_sentiment_summary(sentiment_results[0]) if sentiment_results and
                                                                                             sentiment_results[0] else {
                        'positive': 0, 'negative': 0, 'neutral': 0}
                },

                # MDA sentiment analysis
                'mda_sentiment': mda_analysis,

                # Risk metrics
                'risk_metrics': risk_metrics,

                # Position trading score and plan
                'position_score': position_score,
                'trading_plan': trading_plan,

                # Model information
                'model_type': self.model_type,
                'mda_model_available': self.mda_available,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': 'Position Trading (Long-term with MDA)'
            }

            logger.info(f"Successfully analyzed {symbol} with position score {position_score}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol} for position trading: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    # Include all the existing methods from your original code
    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database with fundamental data structure"""
        try:
            self.indian_stocks = {
                # NIFTY 50 Stocks with enhanced info
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large",
                             "div_yield": 0.003},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large",
                        "div_yield": 0.025},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
                "INFY": {"name": "Infosys", "sector": "Information Technology", "market_cap": "Large",
                         "div_yield": 0.023},
                "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.014},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking", "market_cap": "Large",
                              "div_yield": 0.005},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services", "market_cap": "Large",
                               "div_yield": 0.002},
                "LT": {"name": "Larsen & Toubro", "sector": "Construction", "market_cap": "Large", "div_yield": 0.018},
                "SBIN": {"name": "State Bank of India", "sector": "Banking", "market_cap": "Large", "div_yield": 0.035},
                "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications", "market_cap": "Large",
                               "div_yield": 0.008},
                "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.008},
                "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.012},
                "TITAN": {"name": "Titan Company", "sector": "Consumer Goods", "market_cap": "Large",
                          "div_yield": 0.005},
                "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals", "market_cap": "Large",
                              "div_yield": 0.008},
                "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement", "market_cap": "Large",
                               "div_yield": 0.005},
                "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods", "market_cap": "Large",
                              "div_yield": 0.008},
                "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology", "market_cap": "Large",
                            "div_yield": 0.025},
                "AXISBANK": {"name": "Axis Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.008},
                "WIPRO": {"name": "Wipro", "sector": "Information Technology", "market_cap": "Large",
                          "div_yield": 0.015},
                "NTPC": {"name": "NTPC", "sector": "Power", "market_cap": "Large", "div_yield": 0.045},
                "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power", "market_cap": "Large",
                              "div_yield": 0.038},
                "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas", "market_cap": "Large",
                         "div_yield": 0.055},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology", "market_cap": "Large",
                          "div_yield": 0.032},
                "TATASTEEL": {"name": "Tata Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.025},
                "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate", "market_cap": "Large",
                             "div_yield": 0.001},
                "COALINDIA": {"name": "Coal India", "sector": "Mining", "market_cap": "Large", "div_yield": 0.065},
                "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals", "market_cap": "Large",
                             "div_yield": 0.008},
                "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel", "market_cap": "Large", "div_yield": 0.012},
                "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile", "market_cap": "Large", "div_yield": 0.022},
                "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile", "market_cap": "Large",
                        "div_yield": 0.018},
                "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile", "market_cap": "Large",
                               "div_yield": 0.025},
                "GRASIM": {"name": "Grasim Industries", "sector": "Cement", "market_cap": "Large", "div_yield": 0.015},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement", "market_cap": "Large", "div_yield": 0.003},
                "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile", "market_cap": "Large",
                              "div_yield": 0.005},
                "UPL": {"name": "UPL Limited", "sector": "Chemicals", "market_cap": "Large", "div_yield": 0.012},
                "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas", "market_cap": "Large", "div_yield": 0.035},
                "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large",
                             "div_yield": 0.005},
                "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals", "market_cap": "Large",
                            "div_yield": 0.008},
                "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals", "market_cap": "Large", "div_yield": 0.012},
                "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods", "market_cap": "Large",
                              "div_yield": 0.008},
                "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods", "market_cap": "Large",
                               "div_yield": 0.015},
                "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas", "market_cap": "Large",
                        "div_yield": 0.042},
                "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare", "market_cap": "Large",
                               "div_yield": 0.002},
                "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services", "market_cap": "Large",
                               "div_yield": 0.008},
                "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance", "market_cap": "Large",
                             "div_yield": 0.012},
                "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance", "market_cap": "Large",
                            "div_yield": 0.008},
                "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.015},
                "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure", "market_cap": "Large",
                               "div_yield": 0.012},
                "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile", "market_cap": "Large",
                               "div_yield": 0.008},
                "ITC": {"name": "ITC Limited", "sector": "Consumer Goods", "market_cap": "Large", "div_yield": 0.055},

                # Additional Mid & Small Cap Stocks
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods", "market_cap": "Mid",
                             "div_yield": 0.012},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods", "market_cap": "Mid",
                           "div_yield": 0.008},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals", "market_cap": "Mid",
                               "div_yield": 0.005},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods", "market_cap": "Mid",
                           "div_yield": 0.018},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods", "market_cap": "Mid", "div_yield": 0.012},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals", "market_cap": "Mid",
                          "div_yield": 0.008},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals", "market_cap": "Mid",
                           "div_yield": 0.005},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile", "market_cap": "Mid",
                               "div_yield": 0.012},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile", "market_cap": "Mid",
                             "div_yield": 0.008},
                "MRF": {"name": "MRF Limited", "sector": "Automobile", "market_cap": "Mid", "div_yield": 0.015},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.001},
                "TRENT": {"name": "Trent Limited", "sector": "Retail", "market_cap": "Mid", "div_yield": 0.002},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles", "market_cap": "Mid", "div_yield": 0.003},
            }

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

            logger.info(f"Initialized database with {len(self.indian_stocks)} Indian stocks")

        except Exception as e:
            logger.error(f"Error initializing stock database: {str(e)}")
            # Fallback to minimal database
            self.indian_stocks = {
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas", "market_cap": "Large",
                             "div_yield": 0.003},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology", "market_cap": "Large",
                        "div_yield": 0.025},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "market_cap": "Large", "div_yield": 0.012},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    def load_trained_sbert_model(self, model_path):
        """Load your trained SBERT sentiment model"""
        try:
            if not SBERT_AVAILABLE:
                logger.warning("sentence-transformers not available, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not model_path or not os.path.exists(model_path):
                logger.warning(f"SBERT model not found at {model_path}")
                logger.info("Using TextBlob as fallback for sentiment analysis")
                self.model_type = "TextBlob"
                return

            logger.info(f"Loading trained SBERT sentiment model from {model_path}...")

            # Load your trained pipeline
            self.sentiment_pipeline = joblib.load(model_path)

            if not isinstance(self.sentiment_pipeline, dict):
                raise ValueError("Invalid model format - expected dictionary")

            self.vectorizer = self.sentiment_pipeline.get("vectorizer")
            self.model = self.sentiment_pipeline.get("model")
            self.label_encoder = self.sentiment_pipeline.get("label_encoder")

            if all([self.vectorizer, self.model, self.label_encoder]):
                # Validate model components
                if not hasattr(self.vectorizer, 'transform'):
                    raise ValueError("Invalid vectorizer - missing transform method")
                if not hasattr(self.model, 'predict'):
                    raise ValueError("Invalid model - missing predict method")
                if not hasattr(self.label_encoder, 'classes_'):
                    raise ValueError("Invalid label encoder - missing classes_")

                logger.info("SBERT sentiment model loaded successfully!")
                logger.info(f"Model classes: {list(self.label_encoder.classes_)}")
                self.model_loaded = True
                self.model_type = "SBERT + RandomForest (Your Trained Model)"

                # Test the model with sample text
                test_text = ["This is a positive news about the company"]
                try:
                    embeddings = self.vectorizer.transform(test_text)
                    predictions = self.model.predict(embeddings)
                    probabilities = self.model.predict_proba(embeddings)
                    logger.info("Model test successful!")
                except Exception as test_error:
                    logger.error(f"Model test failed: {str(test_error)}")
                    self.model_type = "TextBlob"
                    self.sentiment_pipeline = None
                    self.model_loaded = False
            else:
                logger.warning("Model components incomplete, using TextBlob fallback")
                self.model_type = "TextBlob"
                self.sentiment_pipeline = None

        except Exception as e:
            logger.error(f"Error loading SBERT model: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info("Using TextBlob as fallback for sentiment analysis")
            self.model_type = "TextBlob"
            self.sentiment_pipeline = None
            self.model_loaded = False

    def analyze_sentiment_with_trained_sbert(self, articles):
        """Analyze sentiment using your trained SBERT model"""
        try:
            if not articles or not self.model_loaded:
                return self.analyze_sentiment_with_textblob(articles)

            if not all([self.vectorizer, self.model, self.label_encoder]):
                logger.error("SBERT model components missing")
                return self.analyze_sentiment_with_textblob(articles)

            # Use your trained model
            embeddings = self.vectorizer.transform(articles)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to generate embeddings")
                return self.analyze_sentiment_with_textblob(articles)

            predictions = self.model.predict(embeddings)
            probabilities = self.model.predict_proba(embeddings)

            sentiment_labels = self.label_encoder.inverse_transform(predictions)
            confidence_scores = np.max(probabilities, axis=1)

            return sentiment_labels.tolist(), confidence_scores.tolist()

        except Exception as e:
            logger.error(f"Error in trained SBERT sentiment analysis: {str(e)}")
            return self.analyze_sentiment_with_textblob(articles)

    def get_indian_stock_data(self, symbol, period="5y"):
        """Get Indian stock data with extended period for position trading"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            symbol = str(symbol).upper().replace(" ", "").replace(".", "")
            if not symbol:
                raise ValueError("Invalid symbol after cleaning")

            # Enhanced Indian stocks mapping
            symbol_mappings = {
                "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS",
                "HDFCBANK": "HDFCBANK.NS", "BAJFINANCE": "BAJFINANCE.NS",
                "HINDUNILVR": "HINDUNILVR.NS", "ICICIBANK": "ICICIBANK.NS",
                "KOTAKBANK": "KOTAKBANK.NS", "SBIN": "SBIN.NS",
                "BHARTIARTL": "BHARTIARTL.NS", "LT": "LT.NS", "MARUTI": "MARUTI.NS",
                "ASIANPAINT": "ASIANPAINT.NS", "HCLTECH": "HCLTECH.NS",
                "TITAN": "TITAN.NS", "SUNPHARMA": "SUNPHARMA.NS",
                "NTPC": "NTPC.NS", "ONGC": "ONGC.NS", "ADANIENT": "ADANIENT.NS",
                "WIPRO": "WIPRO.NS", "TECHM": "TECHM.NS", "POWERGRID": "POWERGRID.NS",
                "ITC": "ITC.NS", "TATAMOTORS": "TATAMOTORS.NS",
            }

            symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO", symbol]
            if symbol in symbol_mappings:
                symbols_to_try.insert(0, symbol_mappings[symbol])

            for sym in symbols_to_try:
                try:
                    logger.info(f"Trying to fetch data for {sym}")
                    ticker = yf.Ticker(sym)

                    # Add timeout and retry logic
                    data = ticker.history(period=period, timeout=30)

                    if data is None or data.empty:
                        logger.warning(f"No data returned for {sym}")
                        continue

                    if len(data) < 252:  # Need at least 1 year of data for position trading
                        logger.warning(f"Insufficient data for position trading {sym}: {len(data)} days")
                        continue

                    # Validate data quality
                    if data['Close'].isna().all():
                        logger.warning(f"All Close prices are NaN for {sym}")
                        continue

                    # Try to get info (optional, might fail for some stocks)
                    info = {}
                    try:
                        info = ticker.info
                    except Exception as info_error:
                        logger.warning(f"Could not fetch info for {sym}: {str(info_error)}")
                        info = {}

                    logger.info(f"Successfully fetched data for {sym}: {len(data)} days")
                    return data, info, sym

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {sym}: {str(e)}")
                    continue

            logger.error(f"Failed to fetch data for all variations of {symbol}")
            return None, None, None

        except Exception as e:
            logger.error(f"Error in get_indian_stock_data for {symbol}: {str(e)}")
            return None, None, None

    def analyze_fundamental_metrics(self, symbol, info):
        """Analyze fundamental metrics crucial for position trading"""
        try:
            fundamentals = {
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'dividend_yield': info.get('dividendYield', None),
                'market_cap': info.get('marketCap', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'book_value': info.get('bookValue', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None)
            }

            # Add sector-specific metrics from our database
            stock_info = self.get_stock_info_from_db(symbol)
            fundamentals['expected_div_yield'] = stock_info.get('div_yield', 0)
            fundamentals['market_cap_category'] = stock_info.get('market_cap', 'Unknown')

            return fundamentals
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {str(e)}")
            return {}

    def calculate_fundamental_score(self, fundamentals, sector):
        """Calculate fundamental score for position trading (0-100)"""
        try:
            score = 0
            max_score = 100

            # P/E Ratio Analysis (15 points)
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio and 8 < pe_ratio < 25:
                score += 15
            elif pe_ratio and 5 < pe_ratio <= 8:
                score += 12  # Undervalued
            elif pe_ratio and 25 <= pe_ratio < 35:
                score += 8  # Slightly expensive

            # PEG Ratio Analysis (15 points)
            peg_ratio = fundamentals.get('peg_ratio')
            if peg_ratio and 0.5 < peg_ratio < 1.0:
                score += 15  # Undervalued growth
            elif peg_ratio and 1.0 <= peg_ratio < 1.5:
                score += 10  # Fair value

            # Revenue and Earnings Growth (20 points)
            revenue_growth = fundamentals.get('revenue_growth', 0)
            earnings_growth = fundamentals.get('earnings_growth', 0)

            if revenue_growth and revenue_growth > 0.20:  # 20% revenue growth
                score += 10
            elif revenue_growth and revenue_growth > 0.15:
                score += 8
            elif revenue_growth and revenue_growth > 0.10:
                score += 5

            if earnings_growth and earnings_growth > 0.25:  # 25% earnings growth
                score += 10
            elif earnings_growth and earnings_growth > 0.15:
                score += 8
            elif earnings_growth and earnings_growth > 0.10:
                score += 5

            # ROE Analysis (10 points)
            roe = fundamentals.get('roe')
            if roe and roe > 0.20:  # 20% ROE
                score += 10
            elif roe and roe > 0.15:
                score += 8
            elif roe and roe > 0.12:
                score += 5

            # Debt Analysis (10 points)
            debt_equity = fundamentals.get('debt_to_equity')
            if debt_equity is not None:
                if debt_equity < 0.3:
                    score += 10  # Very low debt
                elif debt_equity < 0.6:
                    score += 8  # Manageable debt
                elif debt_equity < 1.0:
                    score += 4  # High but acceptable

            # Profitability Margins (10 points)
            profit_margin = fundamentals.get('profit_margin')
            operating_margin = fundamentals.get('operating_margin')

            if profit_margin and profit_margin > 0.15:
                score += 5
            elif profit_margin and profit_margin > 0.10:
                score += 3

            if operating_margin and operating_margin > 0.20:
                score += 5
            elif operating_margin and operating_margin > 0.15:
                score += 3

            # Dividend Yield (10 points) - Important for position trading
            div_yield = fundamentals.get('dividend_yield') or fundamentals.get('expected_div_yield', 0)
            if div_yield and div_yield > 0.03:  # 3% dividend yield
                score += 10
            elif div_yield and div_yield > 0.015:
                score += 6
            elif div_yield and div_yield > 0.005:
                score += 3

            # Financial Health (10 points)
            current_ratio = fundamentals.get('current_ratio')
            if current_ratio and current_ratio > 1.5:
                score += 5
            elif current_ratio and current_ratio > 1.2:
                score += 3

            price_to_book = fundamentals.get('price_to_book')
            if price_to_book and price_to_book < 2.0:
                score += 5
            elif price_to_book and price_to_book < 3.0:
                score += 3

            return min(score, max_score)

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {str(e)}")
            return 0

    def analyze_long_term_trends(self, data):
        """Analyze long-term trends for position trading"""
        try:
            # Multiple timeframe moving averages for position trading
            ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean')
            ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean')
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')

            current_price = data['Close'].iloc[-1]

            # Trend strength analysis
            trend_score = 0

            # Price above all major MAs (Strong uptrend)
            if (current_price > ma_50.iloc[-1] > ma_100.iloc[-1] > ma_200.iloc[-1]):
                trend_score = 100
            elif (current_price > ma_50.iloc[-1] > ma_100.iloc[-1]):
                trend_score = 75  # Moderate uptrend
            elif (current_price > ma_100.iloc[-1]):
                trend_score = 50  # Weak uptrend
            elif (current_price < ma_50.iloc[-1] < ma_100.iloc[-1] < ma_200.iloc[-1]):
                trend_score = 0  # Strong downtrend
            else:
                trend_score = 25  # Sideways/mixed

            # Calculate trend momentum (slope of moving averages)
            ma_50_slope = (ma_50.iloc[-1] - ma_50.iloc[-20]) / ma_50.iloc[-20] if len(ma_50) > 20 else 0
            ma_200_slope = (ma_200.iloc[-1] - ma_200.iloc[-50]) / ma_200.iloc[-50] if len(ma_200) > 50 else 0

            # Long-term price momentum
            price_6m_ago = data['Close'].iloc[-126] if len(data) > 126 else data['Close'].iloc[0]
            price_1y_ago = data['Close'].iloc[-252] if len(data) > 252 else data['Close'].iloc[0]

            momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            momentum_1y = (current_price - price_1y_ago) / price_1y_ago

            return {
                'trend_score': trend_score,
                'ma_50_slope': ma_50_slope,
                'ma_200_slope': ma_200_slope,
                'above_ma_200': current_price > ma_200.iloc[-1],
                'momentum_6m': momentum_6m,
                'momentum_1y': momentum_1y,
                'ma_50': ma_50.iloc[-1],
                'ma_100': ma_100.iloc[-1],
                'ma_200': ma_200.iloc[-1]
            }

        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {str(e)}")
            return {'trend_score': 50, 'ma_50_slope': 0, 'ma_200_slope': 0, 'above_ma_200': False,
                    'momentum_6m': 0, 'momentum_1y': 0, 'ma_50': 0, 'ma_100': 0, 'ma_200': 0}

    def analyze_market_cycles(self, symbol, data):
        """Analyze market cycles and sector rotation"""
        try:
            sector = self.get_stock_info_from_db(symbol)['sector']

            # Sector strength analysis based on current market conditions
            sector_score = 0

            # Interest rate sensitive sectors
            if sector in ['Banking', 'Financial Services']:
                # Banks benefit from rising rates, hurt by falling rates
                sector_score = 65  # Neutral to positive
            elif sector in ['Real Estate', 'Infrastructure']:
                # Real estate hurt by rising rates
                sector_score = 55

            # Defensive sectors (good for position trading)
            elif sector in ['Consumer Goods', 'Pharmaceuticals', 'Healthcare']:
                sector_score = 75  # Generally stable for position trading

            # Cyclical sectors
            elif sector in ['Automobile', 'Steel', 'Cement', 'Metals']:
                sector_score = 60  # Depends on economic cycle

            # Growth sectors
            elif sector in ['Information Technology']:
                sector_score = 70  # Good for position trading

            # Utility and Power
            elif sector in ['Power', 'Utilities']:
                sector_score = 80  # Excellent for position trading (stable dividends)

            # Commodity sectors
            elif sector in ['Oil & Gas', 'Mining']:
                sector_score = 55  # Volatile but can be good long-term

            else:
                sector_score = 60  # Default for other sectors

            return {
                'sector_score': sector_score,
                'sector': sector,
                'cycle_stage': self.determine_market_cycle(data),
                'sector_preference': self.get_sector_preference(sector)
            }

        except Exception as e:
            logger.error(f"Error in market cycle analysis: {str(e)}")
            return {'sector_score': 60, 'sector': 'Unknown', 'cycle_stage': 'Unknown', 'sector_preference': 'Neutral'}

    def determine_market_cycle(self, data):
        """Determine current market cycle stage"""
        try:
            ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')
            current_price = data['Close'].iloc[-1]

            # Check if price has been above MA200 for extended period
            above_ma_200_days = 0
            check_period = min(120, len(data))  # Check last 120 days

            for i in range(check_period):
                if data['Close'].iloc[-(i + 1)] > ma_200.iloc[-(i + 1)]:
                    above_ma_200_days += 1

            above_ma_200_pct = above_ma_200_days / check_period

            if above_ma_200_pct > 0.75:
                return "Bull Market"
            elif above_ma_200_pct < 0.25:
                return "Bear Market"
            else:
                return "Transitional"

        except Exception as e:
            logger.error(f"Error determining market cycle: {str(e)}")
            return "Unknown"

    def get_sector_preference(self, sector):
        """Get sector preference for position trading"""
        high_preference = ['Consumer Goods', 'Information Technology', 'Healthcare',
                           'Pharmaceuticals', 'Power', 'Banking']
        medium_preference = ['Telecommunications', 'Oil & Gas', 'Chemicals', 'Cement']

        if sector in high_preference:
            return 'High'
        elif sector in medium_preference:
            return 'Medium'
        else:
            return 'Low'

    def calculate_technical_score_position(self, data):
        """Calculate technical score optimized for position trading"""
        try:
            if data is None or data.empty:
                return 0

            technical_score = 0
            current_price = data['Close'].iloc[-1]

            # RSI Analysis with longer period (30 points)
            rsi = self.calculate_rsi(data['Close'], period=30)  # Longer period for position trading
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                current_rsi = rsi.iloc[-1]
                if 40 <= current_rsi <= 60:  # Neutral zone - good for position trading
                    technical_score += 30
                elif 30 <= current_rsi < 40:  # Slight oversold
                    technical_score += 25
                elif 60 < current_rsi <= 70:  # Slight overbought
                    technical_score += 20
                elif current_rsi < 30:  # Very oversold
                    technical_score += 15

            # Moving Average Analysis (25 points)
            if len(data) >= 200:
                ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                ma_100 = self.safe_rolling_calculation(data['Close'], 100, 'mean').iloc[-1]
                ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean').iloc[-1]

                if not any(pd.isna([ma_50, ma_100, ma_200])):
                    if current_price > ma_50 > ma_100 > ma_200:  # Perfect alignment
                        technical_score += 25
                    elif current_price > ma_50 > ma_100:  # Good alignment
                        technical_score += 20
                    elif current_price > ma_200:  # Above long-term trend
                        technical_score += 15
                    elif ma_50 > ma_100:  # Short term stronger than medium term
                        technical_score += 10

            # Volume Trend Analysis (15 points)
            if 'Volume' in data.columns and len(data) >= 50:
                recent_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                long_term_volume = self.safe_rolling_calculation(data['Volume'], 50, 'mean').iloc[-1]

                if not pd.isna(recent_volume) and not pd.isna(long_term_volume) and long_term_volume > 0:
                    volume_ratio = recent_volume / long_term_volume
                    if volume_ratio > 1.2:  # Increasing volume
                        technical_score += 15
                    elif volume_ratio > 1.0:
                        technical_score += 10

            # Long-term MACD (15 points)
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'], fast=26, slow=52, signal=18)
            if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                    technical_score += 15
                if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                    if histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                        technical_score += 5

            # Support/Resistance for position entries (15 points)
            support, resistance = self.calculate_support_resistance(data, window=50)  # Longer window
            if support and resistance and not any(pd.isna([support, resistance])):
                distance_to_support = (current_price - support) / support
                distance_to_resistance = (resistance - current_price) / current_price

                if 0.05 <= distance_to_support <= 0.20:  # Good entry zone above support
                    technical_score += 15
                elif distance_to_support > 0.20:  # Well above support
                    technical_score += 10
                elif distance_to_support < 0.05:  # Very close to support
                    technical_score += 8

            return min(100, max(0, technical_score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0

    def calculate_sentiment_score(self, sentiment_data):
        """Calculate sentiment score from news analysis"""
        try:
            if not sentiment_data or len(sentiment_data) < 3:
                return 50  # Neutral sentiment

            sentiments, _, confidences, _, _ = sentiment_data
            if not sentiments or not confidences:
                return 50

            sentiment_value = 0
            total_weight = 0

            for sentiment, confidence in zip(sentiments, confidences):
                weight = confidence if not pd.isna(confidence) else 0.5
                if sentiment == 'positive':
                    sentiment_value += weight
                elif sentiment == 'negative':
                    sentiment_value -= weight
                # neutral adds 0
                total_weight += weight

            if total_weight > 0:
                normalized_sentiment = sentiment_value / total_weight
                sentiment_score = 50 + (normalized_sentiment * 50)  # Scale to 0-100
            else:
                sentiment_score = 50

            return min(100, max(0, sentiment_score))

        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 50

    def generate_position_trading_plan(self, data, score, risk_metrics, fundamentals, trends):
        """Generate comprehensive position trading plan"""
        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)

            # Position trading signals (more conservative)
            if score >= 80:
                entry_signal = "STRONG BUY"
                entry_strategy = "Accumulate on any dips, suitable for core holding"
            elif score >= 65:
                entry_signal = "BUY"
                entry_strategy = "Enter gradually over 2-4 weeks, good long-term prospect"
            elif score >= 50:
                entry_signal = "HOLD/WATCH"
                entry_strategy = "Wait for better entry or more confirmation"
            elif score >= 35:
                entry_signal = "WEAK"
                entry_strategy = "Avoid for position trading, too risky"
            else:
                entry_signal = "AVOID"
                entry_strategy = "Not suitable for position trading"

            # Position sizing for long-term (wider stops, smaller risk per trade)
            stop_loss_distance = atr * 4  # 4 ATR for position trading
            stop_loss = max(current_price - stop_loss_distance, 0)

            # Long-term targets (higher multiples)
            target_1 = current_price + (stop_loss_distance * 2.0)  # 2:1 RR (conservative)
            target_2 = current_price + (stop_loss_distance * 4.0)  # 4:1 RR (primary target)
            target_3 = current_price + (stop_loss_distance * 6.0)  # 6:1 RR (stretch target)

            # Support and Resistance with longer timeframes
            support, resistance = self.calculate_support_resistance(data, window=100)
            if not support or pd.isna(support):
                support = current_price * 0.90
            if not resistance or pd.isna(resistance):
                resistance = current_price * 1.15

            # Expected holding period based on score
            if score >= 75:
                holding_period = "6 months to 2 years (high conviction)"
            elif score >= 60:
                holding_period = "3 months to 1.5 years (medium conviction)"
            else:
                holding_period = "Not recommended for position trading"

            # Entry timing recommendations
            ma_200 = trends.get('ma_200', current_price)
            if current_price > ma_200:
                entry_timing = "Buy on any pullback to major support"
            else:
                entry_timing = "Wait for price to reclaim long-term trend (MA200)"

            # Risk management for position trading
            position_risk = "1% of portfolio maximum per position"
            portfolio_risk = "Maximum 5% total portfolio risk"

            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'entry_timing': entry_timing,
                'stop_loss': stop_loss,
                'targets': {
                    'target_1': target_1,
                    'target_2': target_2,
                    'target_3': target_3
                },
                'support': support,
                'resistance': resistance,
                'holding_period': holding_period,
                'position_risk': position_risk,
                'portfolio_risk': portfolio_risk,
                'stop_distance_pct': (stop_loss_distance / current_price) * 100,
                'upside_potential': ((target_2 - current_price) / current_price) * 100
            }

        except Exception as e:
            logger.error(f"Error generating position trading plan: {str(e)}")
            return {
                'entry_signal': 'ERROR',
                'entry_strategy': 'Analysis failed',
                'entry_timing': 'Unknown',
                'stop_loss': 0,
                'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
                'support': 0,
                'resistance': 0,
                'holding_period': 'Unknown',
                'position_risk': '1% max',
                'portfolio_risk': '5% max',
                'stop_distance_pct': 0,
                'upside_potential': 0
            }

    # Helper methods for technical analysis
    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations"""
        try:
            if data is None or data.empty:
                return pd.Series(dtype=float)

            if len(data) < window:
                return pd.Series([np.nan] * len(data), index=data.index)

            if operation == 'mean':
                return data.rolling(window=window, min_periods=1).mean()
            elif operation == 'std':
                return data.rolling(window=window, min_periods=1).std()
            elif operation == 'min':
                return data.rolling(window=window, min_periods=1).min()
            elif operation == 'max':
                return data.rolling(window=window, min_periods=1).max()
            else:
                logger.error(f"Unknown rolling operation: {operation}")
                return pd.Series([np.nan] * len(data), index=data.index)

        except Exception as e:
            logger.error(f"Error in safe_rolling_calculation: {str(e)}")
            return pd.Series([np.nan] * len(data), index=data.index if hasattr(data, 'index') else range(len(data)))

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            if prices is None or prices.empty:
                return pd.Series(dtype=float)

            if len(prices) < period:
                return pd.Series([50] * len(prices), index=prices.index)

            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = self.safe_rolling_calculation(gain, period, 'mean')
            avg_loss = self.safe_rolling_calculation(loss, period, 'mean')

            if avg_gain.empty or avg_loss.empty:
                return pd.Series([50] * len(prices), index=prices.index)

            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < slow:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()

            if exp1.empty or exp2.empty:
                zeros = pd.Series([0] * len(prices), index=prices.index)
                return zeros, zeros, zeros

            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            return macd_line, signal_line, histogram

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            zeros = pd.Series([0] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return zeros, zeros, zeros

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels"""
        try:
            if data is None or data.empty:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                logger.error("Missing High/Low columns for support/resistance calculation")
                return None, None

            if len(data) < window:
                return data['Low'].min(), data['High'].max()

            highs = self.safe_rolling_calculation(data['High'], window, 'max')
            lows = self.safe_rolling_calculation(data['Low'], window, 'min')

            if highs.empty or lows.empty:
                return data['Low'].min(), data['High'].max()

            # Find significant levels
            resistance_levels = []
            support_levels = []

            for i in range(window, len(data)):
                try:
                    if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                        resistance_levels.append(data['High'].iloc[i])

                    if not pd.isna(lows.iloc[i]) and data['Low'].iloc[i] == lows.iloc[i]:
                        support_levels.append(data['Low'].iloc[i])
                except Exception as e:
                    logger.warning(f"Error processing level at index {i}: {str(e)}")
                    continue

            # Get most recent levels
            if len(resistance_levels) >= 3:
                current_resistance = max(resistance_levels[-3:])
            else:
                current_resistance = data['High'].max()

            if len(support_levels) >= 3:
                current_support = min(support_levels[-3:])
            else:
                current_support = data['Low'].min()

            return current_support, current_resistance

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            try:
                return data['Low'].min(), data['High'].max()
            except:
                return None, None

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics"""
        default_metrics = {
            'volatility': 0.3,
            'var_95': -0.05,
            'max_drawdown': -0.2,
            'sharpe_ratio': 0,
            'atr': 0,
            'risk_level': 'HIGH'
        }

        try:
            if data is None or data.empty or 'Close' not in data.columns:
                logger.error("Invalid data for risk metrics calculation")
                return default_metrics

            returns = data['Close'].pct_change().dropna()

            if returns.empty or len(returns) < 2:
                logger.warning("Insufficient returns data for risk metrics")
                return default_metrics

            # Volatility (annualized)
            try:
                volatility = returns.std() * np.sqrt(252)
                if pd.isna(volatility) or volatility < 0:
                    volatility = 0.3
            except Exception:
                volatility = 0.3

            # Value at Risk (95% confidence)
            try:
                var_95 = np.percentile(returns.dropna(), 5)
                if pd.isna(var_95):
                    var_95 = -0.05
            except Exception:
                var_95 = -0.05

            # Maximum Drawdown
            try:
                rolling_max = data['Close'].expanding().max()
                drawdown = (data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                if pd.isna(max_drawdown):
                    max_drawdown = -0.2
            except Exception:
                max_drawdown = -0.2

            # Sharpe Ratio
            try:
                risk_free_rate = 0.06
                excess_returns = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
                if pd.isna(sharpe_ratio):
                    sharpe_ratio = 0
            except Exception:
                sharpe_ratio = 0

            # ATR for position sizing
            try:
                if len(data) >= 14 and all(col in data.columns for col in ['High', 'Low', 'Close']):
                    high_low = data['High'] - data['Low']
                    high_close = np.abs(data['High'] - data['Close'].shift())
                    low_close = np.abs(data['Low'] - data['Close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean().iloc[-1]
                    if pd.isna(atr):
                        atr = data['Close'].iloc[-1] * 0.02
                else:
                    atr = data['Close'].iloc[-1] * 0.02
            except Exception:
                atr = data['Close'].iloc[-1] * 0.02 if not data['Close'].empty else 0

            # Risk level determination (adjusted for position trading)
            try:
                if volatility > 0.40:
                    risk_level = 'HIGH'
                elif volatility > 0.25:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
            except Exception:
                risk_level = 'HIGH'

            return {
                'volatility': volatility,
                'var_95': var_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'atr': atr,
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return default_metrics

    # News sentiment methods
    def analyze_news_sentiment(self, symbol, num_articles=20):
        """Analyze news sentiment using your trained SBERT model"""
        try:
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"

            if not articles:
                articles = self.get_sample_news(symbol)

            if not articles:
                logger.error(f"No articles available for {symbol}")
                return [], [], [], "No Analysis", "No Source"

            if self.model_loaded and self.sentiment_pipeline:
                sentiments, confidences = self.analyze_sentiment_with_trained_sbert(articles)
                analysis_method = f"Trained SBERT Model ({self.model_type})"
            else:
                sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
                analysis_method = "TextBlob Fallback"

            return sentiments, articles, confidences, analysis_method, news_source

        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {str(e)}")
            return [], [], [], "Error", "Error"

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob"""
        sentiments = []
        confidences = []

        if not articles:
            return sentiments, confidences

        for article in articles:
            try:
                if not article or not isinstance(article, str):
                    sentiments.append('neutral')
                    confidences.append(0.3)
                    continue

                blob = TextBlob(article)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiments.append('positive')
                    confidences.append(min(abs(polarity), 0.8))
                elif polarity < -0.1:
                    sentiments.append('negative')
                    confidences.append(min(abs(polarity), 0.8))
                else:
                    sentiments.append('neutral')
                    confidences.append(0.5)
            except Exception as e:
                logger.warning(f"Error analyzing sentiment for article: {str(e)}")
                sentiments.append('neutral')
                confidences.append(0.3)

        return sentiments, confidences

    def get_sentiment_summary(self, sentiment_scores):
        """Get summary of sentiment scores"""
        if not sentiment_scores:
            return {'positive': 0, 'negative': 0, 'neutral': 0}

        return {
            'positive': sentiment_scores.count('positive'),
            'negative': sentiment_scores.count('negative'),
            'neutral': sentiment_scores.count('neutral')
        }

    # Utility methods
    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {str(e)}")
            return ["RELIANCE", "TCS", "HDFCBANK"]

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol:
                raise ValueError("Invalid symbol format")

            return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown", "market_cap": "Unknown",
                                                        "div_yield": 0})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {"name": str(symbol), "sector": "Unknown", "market_cap": "Unknown", "div_yield": 0}

    def fetch_indian_news(self, symbol, num_articles=20):
        """Fetch news for Indian companies"""
        try:
            if not self.news_api_key:
                return None

            base_symbol = str(symbol).split('.')[0].upper()
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            url = f"https://newsapi.org/v2/everything?q={company_name}+India+stock&apiKey={self.news_api_key}&pageSize={num_articles}&language=en&sortBy=publishedAt"

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                for article in data.get('articles', []):
                    if article.get('title'):
                        articles.append(article['title'])
                return articles if articles else None
            else:
                logger.warning(f"News API returned status code: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("News API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"News API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return None

    # Add this method to the EnhancedPositionTradingSystem class
    def create_personalized_portfolio(self, risk_appetite, time_period_months, budget):
        """Create a personalized portfolio based on user's risk appetite, time period, and budget"""
        try:
            # Convert time period to years for analysis
            time_period_years = time_period_months / 12

            # Adjust trading parameters based on risk appetite
            if risk_appetite.lower() == 'low':
                risk_factor = 0.5
                min_score = 70  # Higher minimum score for conservative approach
                max_positions = 8
            elif risk_appetite.lower() == 'medium':
                risk_factor = 1.0
                min_score = 60
                max_positions = 10
            elif risk_appetite.lower() == 'high':
                risk_factor = 1.5
                min_score = 50
                max_positions = 12
            else:
                risk_factor = 1.0
                min_score = 60
                max_positions = 10

            # Adjust holding period based on user's time horizon
            self.position_trading_params['min_holding_period'] = max(30,
                                                                     time_period_months * 0.7)  # 70% of user's period
            self.position_trading_params['max_holding_period'] = min(1095,
                                                                     time_period_months * 1.2)  # 120% of user's period

            # Adjust risk parameters based on risk appetite
            self.position_trading_params['risk_per_trade'] *= risk_factor
            self.position_trading_params['max_portfolio_risk'] *= risk_factor

            # Get all stock symbols
            symbols = self.get_all_stock_symbols()

            # Analyze all stocks
            stock_scores = {}
            stock_results = {}

            print(f"Analyzing {len(symbols)} stocks for your portfolio...")

            for symbol in symbols:
                result = self.analyze_position_trading_stock(symbol)
                if result and result['position_score'] >= min_score:
                    stock_scores[symbol] = result['position_score']
                    stock_results[symbol] = result

            # Sort stocks by score
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)

            # Select top stocks based on risk appetite
            selected_stocks = dict(sorted_stocks[:max_positions])

            if not selected_stocks:
                return {"error": "No stocks meet the minimum criteria for your risk profile"}

            # Calculate position sizes based on budget and risk
            portfolio = self.calculate_position_sizes(selected_stocks, budget, stock_results)

            # Generate portfolio summary
            portfolio_summary = self.generate_portfolio_summary(portfolio, time_period_months)

            return {
                'portfolio': portfolio,
                'summary': portfolio_summary,
                'risk_profile': risk_appetite,
                'time_period_months': time_period_months,
                'budget': budget
            }

        except Exception as e:
            logger.error(f"Error creating personalized portfolio: {str(e)}")
            return {"error": str(e)}

    def calculate_position_sizes(self, selected_stocks, budget, stock_results):
        """Calculate position sizes based on risk and budget, ensuring whole shares are purchased."""
        portfolio = {}
        remaining_budget = budget

        # Step 1: Calculate initial weights based on score and risk adjustment
        total_score = sum(selected_stocks.values())
        if total_score == 0:
            return {}

        temp_allocations = {}
        for symbol, score in selected_stocks.items():
            weight = score / total_score
            risk_metrics = stock_results[symbol]['risk_metrics']
            volatility = risk_metrics.get('volatility', 0.3)
            risk_adjustment = 1.0 / (1.0 + volatility * 2)  # Penalize high volatility
            adjusted_weight = weight * risk_adjustment
            temp_allocations[symbol] = {'weight': adjusted_weight}

        # Step 2: Normalize the adjusted weights so they sum to 1
        total_adjusted_weight = sum(alloc['weight'] for alloc in temp_allocations.values())
        if total_adjusted_weight == 0:
            return {}

        for symbol in temp_allocations:
            normalized_weight = temp_allocations[symbol]['weight'] / total_adjusted_weight
            temp_allocations[symbol]['normalized_weight'] = normalized_weight

        # Step 3: Iterate through stocks (highest weight first) and allocate budget
        sorted_symbols = sorted(temp_allocations.keys(), key=lambda s: temp_allocations[s]['normalized_weight'],
                                reverse=True)

        for symbol in sorted_symbols:
            current_price = stock_results[symbol]['current_price']

            # Ensure the stock is affordable
            if pd.isna(current_price) or current_price <= 0 or remaining_budget < current_price:
                continue

            # Determine how many shares to buy
            ideal_investment = budget * temp_allocations[symbol]['normalized_weight']
            num_shares = int(ideal_investment // current_price)

            # **Key Fix:** If weights suggest not buying a share but we can afford one, buy one share of this top stock.
            if num_shares == 0:
                num_shares = 1

                # Make sure we don't overspend the remaining budget
            if num_shares * current_price > remaining_budget:
                num_shares = int(remaining_budget // current_price)

            if num_shares > 0:
                investment_amount = num_shares * current_price
                remaining_budget -= investment_amount

                portfolio[symbol] = {
                    'company_name': stock_results[symbol]['company_name'],
                    'sector': stock_results[symbol]['sector'],
                    'score': selected_stocks[symbol],
                    'weight': investment_amount / budget,  # Final weight based on actual investment
                    'current_price': current_price,
                    'num_shares': num_shares,
                    'investment_amount': investment_amount,
                    'stop_loss': stock_results[symbol]['trading_plan']['stop_loss'],
                    'targets': stock_results[symbol]['trading_plan']['targets']
                }

        return portfolio

    def generate_portfolio_summary(self, portfolio, time_period_months):
        """Generate a summary of the portfolio"""
        total_investment = sum(stock['investment_amount'] for stock in portfolio.values())
        avg_score = sum(stock['score'] for stock in portfolio.values()) / len(portfolio)

        # Sector allocation
        sector_allocation = {}
        for stock in portfolio.values():
            sector = stock['sector']
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += stock['investment_amount']

        # Convert to percentages
        for sector in sector_allocation:
            sector_allocation[sector] = sector_allocation[sector] / total_investment * 100

        # Expected return (simplified)
        expected_return = avg_score / 100 * 0.15  # Assume 15% max return for perfect score

        # Adjust for time period (longer time period generally means higher expected returns)
        time_factor = min(2.0, 1.0 + (time_period_months / 12) * 0.1)  # 10% per year additional
        expected_return *= time_factor

        return {
            'total_investment': total_investment,
            'number_of_stocks': len(portfolio),
            'average_score': avg_score,
            'sector_allocation': sector_allocation,
            'expected_return': expected_return,
            'expected_return_percentage': expected_return * 100,
            'recommended_holding_period': f"{time_period_months} months"
        }

def display_portfolio(portfolio_result):
    """Display the created portfolio"""
    portfolio = portfolio_result['portfolio']
    summary = portfolio_result['summary']

    print(f"\n{Fore.YELLOW}{'YOUR PERSONALIZED PORTFOLIO':^80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{'Symbol':<10} {'Company':<25} {'Sector':<15} {'Score':<6} {'Shares':<8} {'Investment':<12}")
    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")

    total_investment = 0
    for symbol, data in portfolio.items():
        print(f"{symbol:<10} {data['company_name'][:24]:<25} {data['sector'][:14]:<15} "
              f"{data['score']:<6.1f} {data['num_shares']:<8} ₹{data['investment_amount']:<10.2f}")
        total_investment += data['investment_amount']

    print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}")
    print(f"{'Total Investment':<50} ₹{total_investment:.2f}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

    # Display summary
    print(f"\n{Fore.YELLOW}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    print(f"Risk Profile: {portfolio_result['risk_profile']}")
    print(f"Time Period: {portfolio_result['time_period_months']} months")
    print(f"Budget: ₹{portfolio_result['budget']:,.2f}")
    print(f"Number of Stocks: {summary['number_of_stocks']}")
    print(f"Average Score: {summary['average_score']:.1f}/100")
    print(f"Expected Return: {summary['expected_return_percentage']:.1f}%")
    print(f"Recommended Holding Period: {summary['recommended_holding_period']}")

    # Display sector allocation
    print(f"\n{Fore.YELLOW}SECTOR ALLOCATION:{Style.RESET_ALL}")
    for sector, allocation in summary['sector_allocation'].items():
        print(f"{sector}: {allocation:.1f}%")

    # Display investment strategy
    print(f"\n{Fore.YELLOW}INVESTMENT STRATEGY:{Style.RESET_ALL}")
    print("1. Invest the allocated amounts in each stock")
    print("2. Set stop-losses at the recommended levels")
    print("3. Review the portfolio quarterly")
    print("4. Consider rebalancing if any stock moves significantly beyond targets")
    print("5. Hold for the recommended time period unless fundamentals change significantly")


def get_sample_news(self, symbol):
    """Generate sample news for demonstration"""
    try:
        base_symbol = str(symbol).split('.')[0]
        stock_info = self.get_stock_info_from_db(base_symbol)
        company_name = stock_info.get("name", base_symbol)

        return [
            f"{company_name} reports strong quarterly earnings growth",
            f"Analysts upgrade {company_name} with positive long-term outlook",
            f"{company_name} announces strategic expansion and investment plans",
            f"Strong fundamentals make {company_name} attractive for long-term investors",
            f"I{company_name} dividend policy supports income-focused portfolios",
            f"Management guidance remains optimistic for {company_name}",
            f"Institutional investors increase holdings in {company_name}",
            f"{company_name} well-positioned for sector growth trends",
            f"ESG initiatives strengthen {company_name} investment case",
            f"Market leadership solidifies {company_name} competitive advantage",
            f"{company_name} balance sheet strength provides stability",
            f"Innovation pipeline drives {company_name} future growth",
            f"Regulatory tailwinds benefit {company_name} business model",
            f"{company_name} demonstrates resilient performance in volatile markets",
            f"Long-term demographic trends favor {company_name} prospects"
        ]
    except Exception as e:
        logger.error(f"Error generating sample news for {symbol}: {str(e)}")
        return [f"Long-term analysis for {symbol}", f"Investment opportunity in {symbol}"]


def main():
    def get_user_input_and_create_portfolio(trading_system):
        """Get user input and create a personalized portfolio"""
        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}PORTFOLIO CREATION WIZARD{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

        # Get risk appetite
        while True:
            risk_appetite = input(f"{Fore.GREEN}Enter your risk appetite (Low/Medium/High): {Style.RESET_ALL}").strip()
            if risk_appetite.lower() in ['low', 'medium', 'high']:
                break
            print(f"{Fore.RED}Please enter Low, Medium, or High{Style.RESET_ALL}")

        # Get time period
        while True:
            try:
                time_period = int(
                    input(f"{Fore.GREEN}Enter your investment time period (in months): {Style.RESET_ALL}").strip())
                if time_period >= 6:  # Minimum 6 months for position trading
                    break
                print(f"{Fore.RED}Please enter at least 6 months for position trading{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")

        # Get budget
        while True:
            try:
                budget = float(input(f"{Fore.GREEN}Enter your investment budget (in INR): {Style.RESET_ALL}").strip())
                if budget >= 10000:  # Minimum ₹10,000
                    break
                print(f"{Fore.RED}Please enter at least ₹10,000{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid amount{Style.RESET_ALL}")

        # Create portfolio
        print(f"\n{Fore.GREEN}Creating your personalized portfolio...{Style.RESET_ALL}")
        portfolio_result = trading_system.create_personalized_portfolio(risk_appetite, time_period, budget)

        if 'error' in portfolio_result:
            print(f"{Fore.RED}Error creating portfolio: {portfolio_result['error']}{Style.RESET_ALL}")
            return

        # Display portfolio
        display_portfolio(portfolio_result)

    """Main function to run position trading analysis with interactive features"""
    try:
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}ENHANCED POSITION TRADING SYSTEM{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

        # Initialize the trading system
        print(f"{Fore.GREEN}Initializing trading system...{Style.RESET_ALL}")
        trading_system = EnhancedPositionTradingSystem(
            model_path="D:/Python_files/models/sentiment_pipeline.joblib",
            mda_model_path="D:/best_model_fold_1.pth",
            news_api_key="dd33ebe105ea4b02a3b7e77bc4a93d01"
        )

        print(f"{Fore.GREEN}✓ System initialized successfully!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

        # Get stock symbols to analyze
        symbols = trading_system.get_all_stock_symbols()
        print(f"Found {len(symbols)} stocks in database")

        # Interactive menu
        while True:
            print(f"\n{Fore.YELLOW}Main Menu:{Style.RESET_ALL}")
            print("1. Analyze specific stock")
            print("2. Analyze top N stocks")
            print("3. Show all available stocks")
            print("4. Create personalized portfolio")
            print("5. Exit")

            choice = input(f"{Fore.GREEN}Enter your choice (1-5): {Style.RESET_ALL}").strip()

            if choice == "1":
                symbol = input(f"{Fore.GREEN}Enter stock symbol: {Style.RESET_ALL}").strip().upper()
                if symbol in symbols:
                    analyze_stock_interactive(trading_system, symbol)
                else:
                    print(f"{Fore.RED}Symbol not found in database.{Style.RESET_ALL}")

            elif choice == "2":
                try:
                    n = int(input(f"{Fore.GREEN}How many stocks to analyze? {Style.RESET_ALL}"))
                    analyze_multiple_stocks(trading_system, symbols[:min(n, len(symbols))])
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

            elif choice == "3":
                print(f"\n{Fore.YELLOW}Available Stocks:{Style.RESET_ALL}")
                for i, symbol in enumerate(symbols, 1):
                    info = trading_system.get_stock_info_from_db(symbol)
                    print(f"{i}. {symbol} - {info['name']} ({info['sector']})")

            elif choice == "4":
                get_user_input_and_create_portfolio(trading_system)

            elif choice == "5":
                print(f"{Fore.CYAN}Thank you for using the Position Trading System!{Style.RESET_ALL}")
                break

            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error in main execution: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


def analyze_stock_interactive(trading_system, symbol):
    """Interactive analysis of a single stock"""
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Analyzing {symbol}...{Style.RESET_ALL}")

    # Show loading animation
    with Halo(text='Analyzing stock data', spinner='dots') as spinner:
        result = trading_system.analyze_position_trading_stock(symbol)
        spinner.succeed("Analysis complete!")

    if not result:
        print(f"{Fore.RED}Failed to analyze {symbol}{Style.RESET_ALL}")
        return

    # Display results
    display_results(result)

    # Offer detailed view
    detail_choice = input(f"\n{Fore.GREEN}Show detailed analysis? (y/n): {Style.RESET_ALL}").strip().lower()
    if detail_choice == 'y':
        display_detailed_analysis(result)


def analyze_multiple_stocks(trading_system, symbols):
    """Analyze multiple stocks and show summary"""
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{Fore.CYAN}[{i}/{len(symbols)}] Analyzing {symbol}...{Style.RESET_ALL}")

        with Halo(text=f'Analyzing {symbol}', spinner='dots') as spinner:
            result = trading_system.analyze_position_trading_stock(symbol)
            if result:
                results.append(result)
                spinner.succeed(f"✓ {symbol} analyzed")
            else:
                spinner.fail(f"✗ {symbol} failed")

        # Small delay to avoid rate limiting
        time.sleep(1)

    # Display summary table
    print(f"\n{Fore.YELLOW}{'SUMMARY RESULTS':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")
    print(f"{'Symbol':<10} {'Score':<8} {'Signal':<12} {'Price':<10} {'Sector':<15}")
    print(f"{Fore.CYAN}{'-' * 60}{Style.RESET_ALL}")

    for result in sorted(results, key=lambda x: x['position_score'], reverse=True):
        signal_color = Fore.GREEN if result['position_score'] >= 65 else (
            Fore.YELLOW if result['position_score'] >= 50 else Fore.RED
        )

        print(
            f"{result['symbol']:<10} "
            f"{result['position_score']:<8.1f} "
            f"{signal_color}{result['trading_plan']['entry_signal']:<12}{Style.RESET_ALL} "
            f"₹{result['current_price']:<9.2f} "
            f"{result['sector']:<15}"
        )


def display_results(result):
    """Display analysis results in a formatted way"""
    # Determine color based on score
    score_color = Fore.GREEN if result['position_score'] >= 65 else (
        Fore.YELLOW if result['position_score'] >= 50 else Fore.RED
    )

    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}ANALYSIS RESULTS: {result['symbol']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    print(f"{'Company:':<20} {result['company_name']}")
    print(f"{'Sector:':<20} {result['sector']}")
    print(f"{'Current Price:':<20} ₹{result['current_price']:.2f}")
    print(f"{'Price Change:':<20} {result['price_change']:+.2f} ({result['price_change_pct']:+.2f}%)")
    print(f"{'Position Score:':<20} {score_color}{result['position_score']:.1f}/100{Style.RESET_ALL}")
    print(f"{'Trading Signal:':<20} {result['trading_plan']['entry_signal']}")

    # MDA Sentiment
    mda = result['mda_sentiment']
    tone_color = Fore.GREEN if mda['mda_score'] >= 60 else (
        Fore.YELLOW if mda['mda_score'] >= 40 else Fore.RED
    )
    print(f"{'MDA Sentiment:':<20} {tone_color}{mda['management_tone']} ({mda['mda_score']:.1f}){Style.RESET_ALL}")

    # Key metrics
    print(f"\n{Fore.YELLOW}KEY METRICS:{Style.RESET_ALL}")
    print(f"{'Fundamental Score:':<20} {result['fundamental_score']:.1f}/100")
    print(f"{'Trend Score:':<20} {result['trend_score']:.1f}/100")
    print(f"{'Risk Level:':<20} {result['risk_metrics']['risk_level']}")

    # Trading plan highlights
    print(f"\n{Fore.YELLOW}TRADING PLAN:{Style.RESET_ALL}")
    print(f"{'Entry Strategy:':<20} {result['trading_plan']['entry_strategy']}")
    print(f"{'Holding Period:':<20} {result['trading_plan']['holding_period']}")
    print(f"{'Stop Loss:':<20} ₹{result['trading_plan']['stop_loss']:.2f}")
    print(f"{'Primary Target:':<20} ₹{result['trading_plan']['targets']['target_2']:.2f}")


def display_detailed_analysis(result):
    """Show detailed analysis results"""
    print(f"\n{Fore.YELLOW}{'DETAILED ANALYSIS':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    # Fundamental metrics
    print(f"\n{Fore.YELLOW}FUNDAMENTAL METRICS:{Style.RESET_ALL}")
    fundamentals = result['fundamentals']
    for key, value in fundamentals.items():
        if value is not None:
            print(f"{key:<20}: {value}")

    # Risk metrics
    print(f"\n{Fore.YELLOW}RISK METRICS:{Style.RESET_ALL}")
    risk = result['risk_metrics']
    for key, value in risk.items():
        print(f"{key:<20}: {value}")

    # News sentiment summary
    print(f"\n{Fore.YELLOW}NEWS SENTIMENT:{Style.RESET_ALL}")
    sentiment = result['sentiment']['sentiment_summary']
    print(f"Positive: {sentiment['positive']} | Neutral: {sentiment['neutral']} | Negative: {sentiment['negative']}")

    # Full trading plan
    print(f"\n{Fore.YELLOW}FULL TRADING PLAN:{Style.RESET_ALL}")
    plan = result['trading_plan']
    for key, value in plan.items():
        if key != 'targets':
            print(f"{key:<20}: {value}")
        else:
            print(f"{key:<20}:")
            for target, price in value.items():
                print(f"  {target}: ₹{price:.2f}")


# Add this import at the top if you want spinner animation
try:
    from halo import Halo
except ImportError:
    class Halo:
        def __init__(self, text='', spinner='dots'):
            self.text = text

        def __enter__(self):
            print(self.text)
            return self

        def succeed(self, text):
            print(f"✓ {text}")

        def fail(self, text):
            print(f"✗ {text}")

        def __exit__(self, *args):
            pass

if __name__ == "__main__":
    main()

    # Enhanced Position Trading System - Key Improvements
    # This file contains the improved sections to replace in your existing code

    import requests
    import json
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)


    class AlternativeDataSource:
        """Alternative data sources for Indian stock fundamentals"""

        def __init__(self):
            self.screener_base_url = "https://www.screener.in/api/company"
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

        def get_screener_data(self, symbol):
            """Get fundamental data from Screener.in (educational use only)"""
            try:
                # This is a sample implementation - you'll need to implement proper scraping
                # or use official APIs where available
                url = f"{self.screener_base_url}/{symbol.lower()}/"

                # For demo, return sample data structure
                # In real implementation, you'd parse the actual website or use their API
                sample_data = {
                    'pe_ratio': np.random.uniform(10, 30),
                    'roe': np.random.uniform(0.10, 0.25),
                    'debt_to_equity': np.random.uniform(0.2, 1.2),
                    'revenue_growth_3y': np.random.uniform(0.05, 0.20),
                    'profit_growth_3y': np.random.uniform(0.08, 0.25),
                    'dividend_yield': np.random.uniform(0.005, 0.04),
                    'book_value': np.random.uniform(100, 800),
                    'market_cap': np.random.uniform(10000, 500000),
                    'current_ratio': np.random.uniform(1.0, 3.0),
                    'sales_growth_3y': np.random.uniform(0.05, 0.18),
                    'promoter_holding': np.random.uniform(0.25, 0.75),
                    'institutional_holding': np.random.uniform(0.10, 0.40)
                }

                logger.info(f"Retrieved alternative fundamental data for {symbol}")
                return sample_data

            except Exception as e:
                logger.error(f"Error fetching screener data for {symbol}: {str(e)}")
                return {}

        def get_nse_data(self, symbol):
            """Get data from NSE (sample implementation)"""
            try:
                # Sample NSE data structure
                # In real implementation, you'd use NSE's official APIs or data feeds
                return {
                    'market_cap_rank': np.random.randint(1, 500),
                    'sector_pe': np.random.uniform(15, 35),
                    'industry_growth': np.random.uniform(0.05, 0.15),
                    'beta': np.random.uniform(0.7, 1.8)
                }
            except Exception as e:
                logger.error(f"Error fetching NSE data: {str(e)}")
                return {}


    class SectorBasedWeighting:
        """Dynamic sector-based weighting system"""

        def __init__(self):
            self.sector_weights = {
                'Banking': {
                    'fundamental_weight': 0.60,
                    'technical_weight': 0.20,
                    'sentiment_weight': 0.10,
                    'mda_weight': 0.10
                },
                'Information Technology': {
                    'fundamental_weight': 0.40,
                    'technical_weight': 0.30,
                    'sentiment_weight': 0.20,
                    'mda_weight': 0.10
                },
                'Consumer Goods': {
                    'fundamental_weight': 0.50,
                    'technical_weight': 0.25,
                    'sentiment_weight': 0.15,
                    'mda_weight': 0.10
                },
                'Pharmaceuticals': {
                    'fundamental_weight': 0.45,
                    'technical_weight': 0.25,
                    'sentiment_weight': 0.20,
                    'mda_weight': 0.10
                },
                'Oil & Gas': {
                    'fundamental_weight': 0.35,
                    'technical_weight': 0.35,
                    'sentiment_weight': 0.20,
                    'mda_weight': 0.10
                },
                'Automobile': {
                    'fundamental_weight': 0.45,
                    'technical_weight': 0.30,
                    'sentiment_weight': 0.15,
                    'mda_weight': 0.10
                },
                'Default': {  # For unknown sectors
                    'fundamental_weight': 0.45,
                    'technical_weight': 0.35,
                    'sentiment_weight': 0.10,
                    'mda_weight': 0.10
                }
            }

        def get_sector_weights(self, sector):
            """Get weights for a specific sector"""
            return self.sector_weights.get(sector, self.sector_weights['Default'])

        def adjust_weights_for_market_regime(self, sector, market_regime='normal'):
            """Adjust weights based on market conditions"""
            base_weights = self.get_sector_weights(sector)

            if market_regime == 'bull':
                # In bull markets, increase technical weight
                adjustments = {
                    'fundamental_weight': -0.05,
                    'technical_weight': 0.10,
                    'sentiment_weight': -0.03,
                    'mda_weight': -0.02
                }
            elif market_regime == 'bear':
                # In bear markets, increase fundamental weight
                adjustments = {
                    'fundamental_weight': 0.10,
                    'technical_weight': -0.05,
                    'sentiment_weight': -0.03,
                    'mda_weight': -0.02
                }
            elif market_regime == 'volatile':
                # In volatile markets, balance sentiment and technical
                adjustments = {
                    'fundamental_weight': -0.05,
                    'technical_weight': 0.05,
                    'sentiment_weight': 0.03,
                    'mda_weight': -0.03
                }
            else:
                adjustments = {k: 0 for k in base_weights.keys()}

            # Apply adjustments
            adjusted_weights = {}
            for key, base_value in base_weights.items():
                adjusted_weights[key] = max(0.05, base_value + adjustments.get(key, 0))

            # Normalize to sum to 1.0
            total = sum(adjusted_weights.values())
            for key in adjusted_weights:
                adjusted_weights[key] /= total

            return adjusted_weights


    class BacktestingEngine:
        """Simple backtesting framework for position trading"""

        def __init__(self, trading_system, initial_capital=1000000):
            self.trading_system = trading_system
            self.initial_capital = initial_capital
            self.capital = initial_capital
            self.positions = {}
            self.trade_history = []
            self.performance_metrics = {}

        def run_backtest(self, symbols, start_date, end_date, rebalance_frequency=30):
            """Run backtest on given symbols"""
            try:
                logger.info(f"Starting backtest from {start_date} to {end_date}")

                # Convert dates
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)

                # Generate rebalance dates
                rebalance_dates = pd.date_range(start=start, end=end, freq=f'{rebalance_frequency}D')

                portfolio_values = []

                for date in rebalance_dates:
                    logger.info(f"Rebalancing on {date.date()}")

                    # Analyze all symbols for this date
                    scores = {}
                    for symbol in symbols:
                        try:
                            # Get historical data up to this date
                            data, _, _ = self.trading_system.get_historical_data_for_date(symbol, date)
                            if data is not None and not data.empty:
                                # Run analysis
                                result = self.trading_system.analyze_position_trading_stock_historical(symbol, data,
                                                                                                       date)
                                if result:
                                    scores[symbol] = result['position_score']
                        except Exception as e:
                            logger.warning(f"Error analyzing {symbol} for {date}: {str(e)}")
                            continue

                    # Select top stocks and rebalance
                    self.rebalance_portfolio(scores, date)

                    # Calculate portfolio value
                    portfolio_value = self.calculate_portfolio_value(date)
                    portfolio_values.append({
                        'date': date,
                        'value': portfolio_value,
                        'return': (portfolio_value - self.initial_capital) / self.initial_capital
                    })

                # Calculate performance metrics
                self.performance_metrics = self.calculate_performance_metrics(portfolio_values)

                return portfolio_values, self.trade_history, self.performance_metrics

            except Exception as e:
                logger.error(f"Error in backtesting: {str(e)}")
                return [], [], {}

        def rebalance_portfolio(self, scores, date, max_positions=8):
            """Rebalance portfolio based on scores"""
            try:
                # Sort stocks by score
                sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                # Close positions not in top stocks
                top_stocks = [stock for stock, _ in sorted_stocks[:max_positions] if _ >= 60]  # Min score 60

                positions_to_close = []
                for symbol in self.positions:
                    if symbol not in top_stocks:
                        positions_to_close.append(symbol)

                # Close positions
                for symbol in positions_to_close:
                    self.close_position(symbol, date)

                # Calculate position sizes
                if top_stocks:
                    position_size = self.capital / len(top_stocks)

                    for symbol in top_stocks:
                        if symbol not in self.positions:
                            self.open_position(symbol, position_size, date)

            except Exception as e:
                logger.error(f"Error rebalancing portfolio: {str(e)}")

        def open_position(self, symbol, amount, date):
            """Open a new position"""
            try:
                # Get price for the date (simplified)
                price = self.get_price_for_date(symbol, date)
                if price:
                    shares = amount // price
                    if shares > 0:
                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date,
                            'amount_invested': shares * price
                        }
                        self.capital -= shares * price

                        self.trade_history.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'date': date,
                            'amount': shares * price
                        })

                        logger.info(f"Opened position: {symbol} - {shares} shares at ₹{price}")
            except Exception as e:
                logger.error(f"Error opening position for {symbol}: {str(e)}")

        def close_position(self, symbol, date):
            """Close an existing position"""
            try:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    price = self.get_price_for_date(symbol, date)

                    if price:
                        proceeds = position['shares'] * price
                        self.capital += proceeds

                        # Calculate return
                        return_pct = (price - position['entry_price']) / position['entry_price']

                        self.trade_history.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': position['shares'],
                            'price': price,
                            'date': date,
                            'amount': proceeds,
                            'return_pct': return_pct,
                            'holding_period': (date - position['entry_date']).days
                        })

                        logger.info(f"Closed position: {symbol} - {return_pct:.2%} return")
                        del self.positions[symbol]

            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {str(e)}")

        def get_price_for_date(self, symbol, date):
            """Get stock price for a specific date (simplified)"""
            try:
                # In a real implementation, you'd use historical price data
                # For demo, return a sample price
                base_price = 100 + hash(symbol) % 1000  # Deterministic "price"
                volatility = 0.02
                days_from_start = (date - pd.to_datetime('2023-01-01')).days
                price_movement = np.sin(days_from_start * 0.01) * volatility
                return base_price * (1 + price_movement)
            except:
                return None

        def calculate_portfolio_value(self, date):
            """Calculate total portfolio value"""
            try:
                total_value = self.capital

                for symbol, position in self.positions.items():
                    price = self.get_price_for_date(symbol, date)
                    if price:
                        total_value += position['shares'] * price

                return total_value
            except Exception as e:
                logger.error(f"Error calculating portfolio value: {str(e)}")
                return self.capital

        def calculate_performance_metrics(self, portfolio_values):
            """Calculate key performance metrics"""
            try:
                if not portfolio_values:
                    return {}

                values = [pv['value'] for pv in portfolio_values]
                returns = [pv['return'] for pv in portfolio_values]

                # Basic metrics
                total_return = returns[-1] if returns else 0
                max_value = max(values) if values else self.initial_capital
                min_value = min(values) if values else self.initial_capital
                max_drawdown = (max_value - min_value) / max_value if max_value > 0 else 0

                # Annualized return (approximate)
                days = (portfolio_values[-1]['date'] - portfolio_values[0]['date']).days
                years = days / 365.25
                annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

                # Volatility of returns
                if len(returns) > 1:
                    volatility = np.std(returns) * np.sqrt(12)  # Assuming monthly rebalancing
                else:
                    volatility = 0

                # Sharpe ratio (assuming 6% risk-free rate)
                risk_free_rate = 0.06
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

                # Win rate
                profitable_trades = sum(1 for trade in self.trade_history
                                        if trade.get('return_pct', 0) > 0)
                total_trades = len([t for t in self.trade_history if t['action'] == 'SELL'])
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0

                return {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'final_capital': values[-1] if values else self.initial_capital
                }

            except Exception as e:
                logger.error(f"Error calculating performance metrics: {str(e)}")
                return {}

        def initialize_enhanced_system(self):
            """Initialize enhanced components"""
            try:
                # Initialize alternative data source
                self.alt_data_source = AlternativeDataSource()

                # Initialize sector-based weighting
                self.sector_weighting = SectorBasedWeighting()

                # Initialize backtesting engine
                self.backtester = BacktestingEngine(self)

                logger.info("Enhanced system components initialized successfully")

            except Exception as e:
                logger.error(f"Error initializing enhanced components: {str(e)}")

        def get_enhanced_fundamental_data(self, symbol, info):
            """Get enhanced fundamental data from multiple sources"""
            try:
                # Start with existing yfinance data
                fundamentals = self.analyze_fundamental_metrics(symbol, info)

                # Add alternative data source
                alt_data = self.alt_data_source.get_screener_data(symbol)
                if alt_data:
                    # Merge with preference for alternative data where available
                    for key, value in alt_data.items():
                        if value is not None and (key not in fundamentals or fundamentals[key] is None):
                            fundamentals[key] = value

                # Add NSE data
                nse_data = self.alt_data_source.get_nse_data(symbol)
                if nse_data:
                    fundamentals.update(nse_data)

                # Calculate derived metrics
                fundamentals.update(self.calculate_derived_metrics(fundamentals))

                return fundamentals

            except Exception as e:
                logger.error(f"Error getting enhanced fundamental data: {str(e)}")
                return self.analyze_fundamental_metrics(symbol, info)

        def calculate_derived_metrics(self, fundamentals):
            """Calculate additional derived metrics"""
            try:
                derived = {}

                # ROIC (Return on Invested Capital) estimation
                if fundamentals.get('roe') and fundamentals.get('debt_to_equity'):
                    roe = fundamentals['roe']
                    de_ratio = fundamentals['debt_to_equity']
                    # Simplified ROIC calculation
                    derived['roic_estimate'] = roe / (1 + de_ratio)

                # Graham Number (Fair Value estimation)
                if fundamentals.get('book_value') and fundamentals.get('earnings_per_share'):
                    bv = fundamentals['book_value']
                    eps = fundamentals.get('earnings_per_share', 0)
                    if eps > 0:
                        derived['graham_number'] = (22.5 * bv * eps) ** 0.5

                # Debt Coverage Ratio
                if fundamentals.get('free_cash_flow') and fundamentals.get('total_debt'):
                    fcf = fundamentals['free_cash_flow']
                    debt = fundamentals['total_debt']
                    if debt > 0:
                        derived['debt_coverage_years'] = debt / fcf if fcf > 0 else float('inf')

                # Quality Score (0-100)
                quality_factors = []

                # ROE quality
                roe = fundamentals.get('roe', 0)
                if roe > 0.15:
                    quality_factors.append(25)
                elif roe > 0.12:
                    quality_factors.append(20)
                elif roe > 0.10:
                    quality_factors.append(15)
                else:
                    quality_factors.append(0)

                # Debt quality
                de_ratio = fundamentals.get('debt_to_equity', 1.0)
                if de_ratio < 0.3:
                    quality_factors.append(25)
                elif de_ratio < 0.6:
                    quality_factors.append(20)
                elif de_ratio < 1.0:
                    quality_factors.append(10)
                else:
                    quality_factors.append(0)

                # Growth quality
                revenue_growth = fundamentals.get('revenue_growth_3y', fundamentals.get('revenue_growth', 0))
                if revenue_growth > 0.15:
                    quality_factors.append(25)
                elif revenue_growth > 0.10:
                    quality_factors.append(20)
                elif revenue_growth > 0.05:
                    quality_factors.append(15)
                else:
                    quality_factors.append(0)

                # Profitability quality
                profit_margin = fundamentals.get('profit_margin', 0)
                if profit_margin > 0.15:
                    quality_factors.append(25)
                elif profit_margin > 0.10:
                    quality_factors.append(20)
                elif profit_margin > 0.05:
                    quality_factors.append(15)
                else:
                    quality_factors.append(0)

                derived['quality_score'] = sum(quality_factors)

                return derived

            except Exception as e:
                logger.error(f"Error calculating derived metrics: {str(e)}")
                return {}

        def calculate_adaptive_position_score(self, data, sentiment_data, fundamentals, trends,
                                              market_analysis, sector, mda_analysis=None):
            """Calculate position score with adaptive sector-based weighting"""
            try:
                # Get market regime
                market_regime = self.determine_market_regime_enhanced(data)

                # Get adaptive weights for sector and market regime
                weights = self.sector_weighting.adjust_weights_for_market_regime(sector, market_regime)

                # Calculate individual scores
                fundamental_score = self.calculate_enhanced_fundamental_score(fundamentals, sector)
                technical_score = self.calculate_technical_score_position(data)
                sentiment_score = self.calculate_sentiment_score(sentiment_data)

                # MDA sentiment score
                mda_score = 50
                if mda_analysis and isinstance(mda_analysis, dict):
                    mda_score = mda_analysis.get('mda_score', 50)

                # Apply adaptive weights
                base_score = (
                        fundamental_score * weights['fundamental_weight'] +
                        technical_score * weights['technical_weight'] +
                        sentiment_score * weights['sentiment_weight'] +
                        mda_score * weights['mda_weight']
                )

                # Apply trend and sector modifiers
                trend_score = trends.get('trend_score', 50)
                sector_score = market_analysis.get('sector_score', 60)

                trend_modifier = trend_score / 100
                sector_modifier = sector_score / 100

                # Market regime adjustments
                regime_multiplier = 1.0
                if market_regime == 'bull':
                    regime_multiplier = 1.05  # 5% boost in bull market
                elif market_regime == 'bear':
                    regime_multiplier = 0.95  # 5% penalty in bear market
                elif market_regime == 'volatile':
                    regime_multiplier = 0.98  # 2% penalty in volatile market

                # Final score calculation
                final_score = base_score * regime_multiplier * (0.7 + 0.2 * trend_modifier + 0.1 * sector_modifier)

                # Quality adjustments
                quality_score = fundamentals.get('quality_score', 50)
                if quality_score >= 80:
                    final_score *= 1.10  # 10% bonus for high quality
                elif quality_score >= 60:
                    final_score *= 1.05  # 5% bonus for good quality
                elif quality_score < 30:
                    final_score *= 0.90  # 10% penalty for low quality

                return min(100, max(0, final_score))

            except Exception as e:
                logger.error(f"Error calculating adaptive position score: {str(e)}")
                return 0

        def determine_market_regime_enhanced(self, data):
            """Enhanced market regime detection"""
            try:
                if data is None or data.empty:
                    return 'normal'

                # Calculate multiple indicators
                ma_200 = self.safe_rolling_calculation(data['Close'], 200, 'mean')
                current_price = data['Close'].iloc[-1]

                # Volatility over last 60 days
                returns = data['Close'].pct_change().dropna()
                if len(returns) >= 60:
                    recent_vol = returns.tail(60).std() * np.sqrt(252)
                    long_term_vol = returns.std() * np.sqrt(252)
                    vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1
                else:
                    vol_ratio = 1

                # Trend strength
                if len(data) >= 200:
                    ma_200_current = ma_200.iloc[-1]
                    price_above_ma = current_price > ma_200_current

                    # Check consistency
                    above_ma_days = sum(1 for i in range(min(60, len(data)))
                                        if data['Close'].iloc[-(i + 1)] > ma_200.iloc[-(i + 1)])
                    consistency = above_ma_days / min(60, len(data))
                else:
                    price_above_ma = True
                    consistency = 0.5

                # Regime determination
                if vol_ratio > 1.5:
                    return 'volatile'
                elif price_above_ma and consistency > 0.7:
                    return 'bull'
                elif not price_above_ma and consistency < 0.3:
                    return 'bear'
                else:
                    return 'normal'

            except Exception as e:
                logger.error(f"Error determining market regime: {str(e)}")
                return 'normal'

        def calculate_enhanced_fundamental_score(self, fundamentals, sector):
            """Enhanced fundamental scoring with sector adjustments"""
            try:
                base_score = self.calculate_fundamental_score(fundamentals, sector)

                # Add quality score influence
                quality_score = fundamentals.get('quality_score', 50)
                quality_bonus = (quality_score - 50) * 0.2  # Max 10 point bonus/penalty

                # ROIC bonus
                roic = fundamentals.get('roic_estimate', 0)
                if roic > 0.15:
                    roic_bonus = 5
                elif roic > 0.12:
                    roic_bonus = 3
                else:
                    roic_bonus = 0

                # Sector-specific adjustments
                sector_adjustments = {
                    'Banking': self._banking_sector_adjustments(fundamentals),
                    'Information Technology': self._it_sector_adjustments(fundamentals),
                    'Consumer Goods': self._consumer_goods_adjustments(fundamentals),
                    'Pharmaceuticals': self._pharma_adjustments(fundamentals)
                }

                sector_bonus = sector_adjustments.get(sector, 0)

                enhanced_score = base_score + quality_bonus + roic_bonus + sector_bonus
                return min(100, max(0, enhanced_score))

            except Exception as e:
                logger.error(f"Error calculating enhanced fundamental score: {str(e)}")
                return self.calculate_fundamental_score(fundamentals, sector)

        def _banking_sector_adjustments(self, fundamentals):
            """Banking sector specific fundamental adjustments"""
            bonus = 0

            # NPA concerns for banks
            if fundamentals.get('npa_ratio'):  # If available
                npa = fundamentals['npa_ratio']
                if npa < 0.02:
                    bonus += 5
                elif npa > 0.05:
                    bonus -= 5

            # Capital Adequacy
            if fundamentals.get('capital_adequacy_ratio'):
                car = fundamentals['capital_adequacy_ratio']
                if car > 0.15:
                    bonus += 3

            return bonus

        def _it_sector_adjustments(self, fundamentals):
            """IT sector specific adjustments"""
            bonus = 0

            # Higher margins expected in IT
            operating_margin = fundamentals.get('operating_margin', 0)
            if operating_margin > 0.25:
                bonus += 5
            elif operating_margin > 0.20:
                bonus += 3

            # Dollar revenue exposure (if available)
            # This would require specific IT sector data

            return bonus

        def _consumer_goods_adjustments(self, fundamentals):
            """Consumer goods sector adjustments"""
            bonus = 0

            # Brand strength indicators
            profit_margin = fundamentals.get('profit_margin', 0)
            if profit_margin > 0.12:  # Strong brand power
                bonus += 3

            # Distribution strength (working capital efficiency)
            current_ratio = fundamentals.get('current_ratio', 1)
            if 1.5 <= current_ratio <= 2.5:  # Optimal range
                bonus += 2

            return bonus

        def _pharma_adjustments(self, fundamentals):
            """Pharmaceutical sector adjustments"""
            bonus = 0

            # R&D intensity (if available)
            # Higher R&D spending is positive for pharma

            # Export orientation bonus (if available)
            # Most Indian pharma companies have good export markets

            # Regulatory compliance (if available)

            return bonus

        def run_backtest_analysis(self, symbols, start_date="2020-01-01", end_date="2023-12-31"):
            """Run comprehensive backtest analysis"""
            try:
                logger.info("Starting backtest analysis...")

                # Initialize backtester if not already done
                if not hasattr(self, 'backtester'):
                    self.backtester = BacktestingEngine(self)

                # Run backtest
                portfolio_values, trade_history, performance_metrics = self.backtester.run_backtest(
                    symbols, start_date, end_date
                )

                # Generate backtest report
                report = self.generate_backtest_report(portfolio_values, trade_history, performance_metrics)

                return {
                    'portfolio_values': portfolio_values,
                    'trade_history': trade_history,
                    'performance_metrics': performance_metrics,
                    'report': report
                }

            except Exception as e:
                logger.error(f"Error in backtest analysis: {str(e)}")
                return None

        def generate_backtest_report(self, portfolio_values, trade_history, performance_metrics):
            """Generate comprehensive backtest report"""
            try:
                report = {
                    'summary': {
                        'period': f"{portfolio_values[0]['date'].date()} to {portfolio_values[-1]['date'].date()}" if portfolio_values else "N/A",
                        'total_return': f"{performance_metrics.get('total_return', 0):.2%}",
                        'annualized_return': f"{performance_metrics.get('annualized_return', 0):.2%}",
                        'max_drawdown': f"{performance_metrics.get('max_drawdown', 0):.2%}",
                        'sharpe_ratio': f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
                        'win_rate': f"{performance_metrics.get('win_rate', 0):.2%}",
                        'total_trades': performance_metrics.get('total_trades', 0),
                        'final_capital': f"₹{performance_metrics.get('final_capital', 0):,.2f}"
                    },
                    'monthly_returns': self.calculate_monthly_returns(portfolio_values),
                    'best_trades': self.get_best_trades(trade_history),
                    'worst_trades': self.get_worst_trades(trade_history),
                    'sector_performance': self.analyze_sector_performance(trade_history),
                    'risk_metrics': {
                        'volatility': f"{performance_metrics.get('volatility', 0):.2%}",
                        'var_95': f"{performance_metrics.get('var_95', 0):.2%}",
                        'max_consecutive_losses': self.calculate_max_consecutive_losses(trade_history),
                        'avg_holding_period': self.calculate_avg_holding_period(trade_history)
                    }
                }

                return report

            except Exception as e:
                logger.error(f"Error generating backtest report: {str(e)}")
                return {'error': str(e)}

        def calculate_monthly_returns(self, portfolio_values):
            """Calculate monthly returns from portfolio values"""
            try:
                if not portfolio_values or len(portfolio_values) < 2:
                    return []

                df = pd.DataFrame(portfolio_values)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

                # Resample to monthly
                monthly = df.resample('M').last()
                monthly['monthly_return'] = monthly['value'].pct_change()

                return monthly['monthly_return'].dropna().to_dict()

            except Exception as e:
                logger.error(f"Error calculating monthly returns: {str(e)}")
                return []

        def get_best_trades(self, trade_history, top_n=5):
            """Get best performing trades"""
            try:
                sell_trades = [t for t in trade_history if t['action'] == 'SELL' and 'return_pct' in t]
                if not sell_trades:
                    return []

                best_trades = sorted(sell_trades, key=lambda x: x['return_pct'], reverse=True)[:top_n]

                return [{
                    'symbol': trade['symbol'],
                    'return': f"{trade['return_pct']:.2%}",
                    'holding_period': f"{trade['holding_period']} days",
                    'exit_date': trade['date'].strftime('%Y-%m-%d')
                } for trade in best_trades]

            except Exception as e:
                logger.error(f"Error getting best trades: {str(e)}")
                return []

        def get_worst_trades(self, trade_history, bottom_n=5):
            """Get worst performing trades"""
            try:
                sell_trades = [t for t in trade_history if t['action'] == 'SELL' and 'return_pct' in t]
                if not sell_trades:
                    return []

                worst_trades = sorted(sell_trades, key=lambda x: x['return_pct'])[:bottom_n]

                return [{
                    'symbol': trade['symbol'],
                    'return': f"{trade['return_pct']:.2%}",
                    'holding_period': f"{trade['holding_period']} days",
                    'exit_date': trade['date'].strftime('%Y-%m-%d')
                } for trade in worst_trades]

            except Exception as e:
                logger.error(f"Error getting worst trades: {str(e)}")
                return []

        def analyze_sector_performance(self, trade_history):
            """Analyze performance by sector"""
            try:
                sell_trades = [t for t in trade_history if t['action'] == 'SELL' and 'return_pct' in t]
                if not sell_trades:
                    return {}

                sector_performance = {}

                for trade in sell_trades:
                    symbol = trade['symbol']
                    stock_info = self.get_stock_info_from_db(symbol)
                    sector = stock_info.get('sector', 'Unknown')

                    if sector not in sector_performance:
                        sector_performance[sector] = {
                            'trades': [],
                            'total_return': 0,
                            'win_rate': 0
                        }

                    sector_performance[sector]['trades'].append(trade['return_pct'])

                # Calculate sector metrics
                for sector, data in sector_performance.items():
                    trades = data['trades']
                    data['avg_return'] = np.mean(trades)
                    data['win_rate'] = sum(1 for r in trades if r > 0) / len(trades)
                    data['total_trades'] = len(trades)
                    data['best_trade'] = max(trades)
                    data['worst_trade'] = min(trades)

                return sector_performance

            except Exception as e:
                logger.error(f"Error analyzing sector performance: {str(e)}")
                return {}

        def calculate_max_consecutive_losses(self, trade_history):
            """Calculate maximum consecutive losses"""
            try:
                sell_trades = [t for t in trade_history if t['action'] == 'SELL' and 'return_pct' in t]
                if not sell_trades:
                    return 0

                max_consecutive = 0
                current_consecutive = 0

                for trade in sell_trades:
                    if trade['return_pct'] < 0:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0

                return max_consecutive

            except Exception as e:
                logger.error(f"Error calculating consecutive losses: {str(e)}")
                return 0

        def calculate_avg_holding_period(self, trade_history):
            """Calculate average holding period"""
            try:
                sell_trades = [t for t in trade_history if t['action'] == 'SELL' and 'holding_period' in t]
                if not sell_trades:
                    return 0

                avg_days = np.mean([trade['holding_period'] for trade in sell_trades])
                return f"{avg_days:.0f} days"

            except Exception as e:
                logger.error(f"Error calculating average holding period: {str(e)}")
                return "N/A"

        # Enhanced portfolio optimization methods

        def optimize_portfolio_allocation(self, stock_scores, max_positions=10, min_score=60):
            """Optimize portfolio allocation using Modern Portfolio Theory concepts"""
            try:
                # Filter stocks by minimum score
                qualified_stocks = {symbol: score for symbol, score in stock_scores.items()
                                    if score >= min_score}

                if not qualified_stocks:
                    logger.warning("No stocks meet minimum score criteria")
                    return {}

                # Sort by score
                sorted_stocks = sorted(qualified_stocks.items(), key=lambda x: x[1], reverse=True)

                # Select top stocks
                selected_stocks = dict(sorted_stocks[:max_positions])

                # Calculate risk-adjusted weights
                weights = self.calculate_risk_adjusted_weights(selected_stocks)

                return weights

            except Exception as e:
                logger.error(f"Error optimizing portfolio allocation: {str(e)}")
                return {}

        def calculate_risk_adjusted_weights(self, stock_scores):
            """Calculate risk-adjusted position weights"""
            try:
                total_score = sum(stock_scores.values())
                base_weights = {symbol: score / total_score for symbol, score in stock_scores.items()}

                # Apply risk adjustments (simplified)
                adjusted_weights = {}

                for symbol, weight in base_weights.items():
                    # Get stock info for risk adjustment
                    stock_info = self.get_stock_info_from_db(symbol)
                    sector = stock_info.get('sector', 'Unknown')
                    market_cap = stock_info.get('market_cap', 'Unknown')

                    # Risk adjustments
                    risk_multiplier = 1.0

                    # Market cap adjustment
                    if market_cap == 'Large':
                        risk_multiplier *= 1.1  # Slightly higher weight for large caps
                    elif market_cap == 'Small':
                        risk_multiplier *= 0.8  # Lower weight for small caps

                    # Sector concentration limits
                    sector_weights = {s: sum(w for s2, w in adjusted_weights.items()
                                             if self.get_stock_info_from_db(s2).get('sector') == sector)
                                      for s in set(self.get_stock_info_from_db(s).get('sector')
                                                   for s in stock_scores.keys())}

                    current_sector_weight = sector_weights.get(sector, 0)
                    if current_sector_weight > 0.25:  # Max 25% per sector
                        risk_multiplier *= 0.7

                    adjusted_weights[symbol] = weight * risk_multiplier

                # Normalize weights
                total_adjusted = sum(adjusted_weights.values())
                if total_adjusted > 0:
                    adjusted_weights = {symbol: weight / total_adjusted
                                        for symbol, weight in adjusted_weights.items()}

                return adjusted_weights

            except Exception as e:
                logger.error(f"Error calculating risk-adjusted weights: {str(e)}")
                # Return equal weights as fallback
                n_stocks = len(stock_scores)
                return {symbol: 1.0 / n_stocks for symbol in stock_scores.keys()}

        # Market regime detection improvements

        def detect_market_regime_advanced(self, market_data=None):
            """Advanced market regime detection using multiple indicators"""
            try:
                if market_data is None:
                    # Use Nifty 50 as market proxy
                    nifty_data, _, _ = self.get_indian_stock_data("^NSEI", period="2y")
                    if nifty_data is None or nifty_data.empty:
                        return 'normal'
                    market_data = nifty_data

                # Multiple regime indicators
                regime_indicators = {}

                # 1. Moving Average Slope
                ma_200 = self.safe_rolling_calculation(market_data['Close'], 200, 'mean')
                if len(ma_200) >= 50:
                    recent_slope = (ma_200.iloc[-1] - ma_200.iloc[-50]) / ma_200.iloc[-50]
                    regime_indicators['ma_slope'] = 'bull' if recent_slope > 0.05 else (
                        'bear' if recent_slope < -0.05 else 'neutral')

                # 2. Volatility Regime
                returns = market_data['Close'].pct_change().dropna()
                if len(returns) >= 60:
                    recent_vol = returns.tail(60).std() * np.sqrt(252)
                    long_vol = returns.std() * np.sqrt(252)
                    vol_ratio = recent_vol / long_vol if long_vol > 0 else 1
                    regime_indicators['volatility'] = 'high' if vol_ratio > 1.3 else (
                        'low' if vol_ratio < 0.8 else 'normal')

                # 3. Momentum Regime
                if len(market_data) >= 252:
                    one_year_return = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-252]) - 1
                    regime_indicators['momentum'] = 'positive' if one_year_return > 0.1 else (
                        'negative' if one_year_return < -0.1 else 'neutral')

                # 4. Breadth Indicators (simplified)
                # In a real implementation, you'd use advance/decline data
                if len(market_data) >= 50:
                    above_ma_50 = market_data['Close'].iloc[-1] > \
                                  self.safe_rolling_calculation(market_data['Close'], 50, 'mean').iloc[-1]
                    regime_indicators['breadth'] = 'positive' if above_ma_50 else 'negative'

                # Combine indicators to determine regime
                regime = self.combine_regime_indicators(regime_indicators)

                return regime

            except Exception as e:
                logger.error(f"Error in advanced regime detection: {str(e)}")
                return 'normal'

        def combine_regime_indicators(self, indicators):
            """Combine multiple regime indicators into single regime"""
            try:
                bull_score = 0
                bear_score = 0
                volatile_score = 0

                # MA Slope
                if indicators.get('ma_slope') == 'bull':
                    bull_score += 2
                elif indicators.get('ma_slope') == 'bear':
                    bear_score += 2

                # Volatility
                if indicators.get('volatility') == 'high':
                    volatile_score += 2
                elif indicators.get('volatility') == 'low':
                    bull_score += 1

                # Momentum
                if indicators.get('momentum') == 'positive':
                    bull_score += 2
                elif indicators.get('momentum') == 'negative':
                    bear_score += 2

                # Breadth
                if indicators.get('breadth') == 'positive':
                    bull_score += 1
                elif indicators.get('breadth') == 'negative':
                    bear_score += 1

                # Determine regime
                if volatile_score >= 2:
                    return 'volatile'
                elif bull_score >= bear_score and bull_score >= 3:
                    return 'bull'
                elif bear_score > bull_score and bear_score >= 3:
                    return 'bear'
                else:
                    return 'normal'

            except Exception as e:
                logger.error(f"Error combining regime indicators: {str(e)}")
                return 'normal'

        # Performance attribution and analysis

        def analyze_performance_attribution(self, portfolio_returns, benchmark_returns=None):
            """Analyze performance attribution"""
            try:
                if benchmark_returns is None:
                    # Use sample benchmark returns (Nifty 50 proxy)
                    benchmark_returns = [0.01] * len(portfolio_returns)  # 1% monthly return

                attribution = {}

                # Calculate excess returns
                excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
                attribution['excess_return'] = np.mean(excess_returns) * 12  # Annualized

                # Information Ratio
                tracking_error = np.std(excess_returns) * np.sqrt(12)
                attribution['information_ratio'] = attribution[
                                                       'excess_return'] / tracking_error if tracking_error > 0 else 0

                # Beta calculation
                if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                    port_returns = np.array(portfolio_returns)
                    bench_returns = np.array(benchmark_returns)

                    covariance = np.cov(port_returns, bench_returns)[0, 1]
                    benchmark_variance = np.var(bench_returns)

                    attribution['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                else:
                    attribution['beta'] = 1.0

                # Alpha calculation
                risk_free_rate = 0.06 / 12  # Monthly risk-free rate
                portfolio_return = np.mean(portfolio_returns)
                benchmark_return = np.mean(benchmark_returns)

                attribution['alpha'] = (portfolio_return - risk_free_rate) - attribution['beta'] * (
                        benchmark_return - risk_free_rate)
                attribution['alpha_annualized'] = attribution['alpha'] * 12

                return attribution

            except Exception as e:
                logger.error(f"Error in performance attribution: {str(e)}")
                return {}

        def generate_risk_report(self, portfolio_data, positions):
            """Generate comprehensive risk report"""
            try:
                risk_report = {}

                # Portfolio concentration risk
                risk_report['concentration'] = self.analyze_concentration_risk(positions)

                # Sector exposure
                risk_report['sector_exposure'] = self.analyze_sector_exposure(positions)

                # Market cap exposure
                risk_report['market_cap_exposure'] = self.analyze_market_cap_exposure(positions)

                # Liquidity risk
                risk_report['liquidity_risk'] = self.assess_liquidity_risk(positions)

                # Correlation risk
                risk_report['correlation_risk'] = self.analyze_correlation_risk(positions)

                return risk_report

            except Exception as e:
                logger.error(f"Error generating risk report: {str(e)}")
                return {}

        def analyze_concentration_risk(self, positions):
            """Analyze portfolio concentration risk"""
            try:
                if not positions:
                    return {'status': 'No positions'}

                total_value = sum(pos['amount_invested'] for pos in positions.values())

                if total_value == 0:
                    return {'status': 'Zero portfolio value'}

                # Calculate position weights
                weights = {symbol: pos['amount_invested'] / total_value
                           for symbol, pos in positions.items()}

                # Concentration metrics
                max_weight = max(weights.values()) if weights else 0
                top_5_weight = sum(sorted(weights.values(), reverse=True)[:5])

                # Herfindahl-Hirschman Index (HHI)
                hhi = sum(w ** 2 for w in weights.values())

                concentration_risk = {
                    'max_position_weight': max_weight,
                    'top_5_weight': top_5_weight,
                    'hhi_index': hhi,
                    'diversification_ratio': 1 / hhi if hhi > 0 else 0,
                    'risk_level': 'HIGH' if max_weight > 0.15 or hhi > 0.15 else
                    ('MEDIUM' if max_weight > 0.10 or hhi > 0.10 else 'LOW')
                }

                return concentration_risk

            except Exception as e:
                logger.error(f"Error analyzing concentration risk: {str(e)}")
                return {'error': str(e)}

        def analyze_sector_exposure(self, positions):
            """Analyze sector-wise exposure"""
            try:
                if not positions:
                    return {}

                sector_exposure = {}
                total_value = sum(pos['amount_invested'] for pos in positions.values())

                if total_value == 0:
                    return {}

                for symbol, position in positions.items():
                    stock_info = self.get_stock_info_from_db(symbol)
                    sector = stock_info.get('sector', 'Unknown')
                    weight = position['amount_invested'] / total_value

                    if sector not in sector_exposure:
                        sector_exposure[sector] = {
                            'weight': 0,
                            'positions': 0,
                            'symbols': []
                        }

                    sector_exposure[sector]['weight'] += weight
                    sector_exposure[sector]['positions'] += 1
                    sector_exposure[sector]['symbols'].append(symbol)

                # Add risk assessment
                for sector, data in sector_exposure.items():
                    if data['weight'] > 0.30:
                        data['risk_level'] = 'HIGH'
                    elif data['weight'] > 0.20:
                        data['risk_level'] = 'MEDIUM'
                    else:
                        data['risk_level'] = 'LOW'

                return sector_exposure

            except Exception as e:
                logger.error(f"Error analyzing sector exposure: {str(e)}")
                return {}

        # Updated main function with enhanced features

        def analyze_market_regime(trading_system):
            pass

        def run_risk_analysis(trading_system):
            pass

        def run_performance_attribution(trading_system):
            pass

        def show_available_stocks(trading_system):
            pass

        def enhanced_main(self):
            """Enhanced main function with new features"""
            print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}ENHANCED POSITION TRADING SYSTEM v2.0{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

            try:
                # Initialize system
                trading_system = EnhancedPositionTradingSystem()
                trading_system.initialize_enhanced_system()

                print(f"{Fore.GREEN}✓ Enhanced system initialized successfully!{Style.RESET_ALL}")

                while True:
                    print(f"\n{Fore.YELLOW}Enhanced Menu:{Style.RESET_ALL}")
                    print("1. Analyze specific stock (Enhanced)")
                    print("2. Run portfolio analysis")
                    print("3. Run backtest analysis")
                    print("4. Market regime analysis")
                    print("5. Risk analysis")
                    print("6. Performance attribution")
                    print("7. Show available stocks")
                    print("8. Exit")

                    choice = input(f"{Fore.GREEN}Enter choice (1-8): {Style.RESET_ALL}").strip()

                    if choice == "1":
                        symbol = input(f"{Fore.GREEN}Enter stock symbol: {Style.RESET_ALL}").strip().upper()
                        self.enhanced_analyze_stock(trading_system, symbol)

                    elif choice == "2":
                        self.run_portfolio_analysis(trading_system)

                    elif choice == "3":
                        self.run_backtest_menu(trading_system)

                    elif choice == "4":
                        self.analyze_market_regime(trading_system)

                    elif choice == "5":
                        self.run_risk_analysis(trading_system)

                    elif choice == "6":
                        self.run_performance_attribution(trading_system)

                    elif choice == "7":
                        self.show_available_stocks(trading_system)

                    elif choice == "8":
                        print(f"{Fore.CYAN}Thank you for using Enhanced Position Trading System!{Style.RESET_ALL}")
                        break

                    else:
                        print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}Error in enhanced main: {str(e)}{Style.RESET_ALL}")

        def display_regime_recommendations(result, market_regime):
            pass

        def enhanced_analyze_stock(trading_system, symbol):
            """Enhanced stock analysis with new features"""
            print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Enhanced Analysis: {symbol}{Style.RESET_ALL}")

            result = trading_system.analyze_position_trading_stock(symbol)
            if not result:
                print(f"{Fore.RED}Analysis failed for {symbol}{Style.RESET_ALL}")
                return

            # Display enhanced results
            trading_system.display_enhanced_results(result)

            # Show regime-adjusted score
            market_regime = trading_system.detect_market_regime_advanced()
            print(f"\n{Fore.YELLOW}Market Regime: {market_regime.upper()}{Style.RESET_ALL}")

            # Regime-specific recommendations
            trading_system.display_regime_recommendations(result, market_regime)

        def display_enhanced_results(result):
            """Display enhanced analysis results"""
            print(f"\n{Fore.YELLOW}ENHANCED ANALYSIS RESULTS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")

            # Basic info
            print(f"Company: {result['company_name']}")
            print(f"Sector: {result['sector']} | Market Cap: {result['market_cap_category']}")
            print(f"Current Price: ₹{result['current_price']:.2f}")

            # Enhanced scores
            print(f"\n{Fore.YELLOW}SCORING BREAKDOWN:{Style.RESET_ALL}")
            print(f"Fundamental Score: {result.get('fundamental_score', 0):.1f}/100")
            print(f"Quality Score: {result['fundamentals'].get('quality_score', 0):.1f}/100")
            print(f"Technical Score: {result.get('technical_score', 0):.1f}/100")
            print(f"MDA Sentiment: {result['mda_sentiment'].get('mda_score', 50):.1f}/100")

            # Overall position score with color
            score = result['position_score']
            score_color = Fore.GREEN if score >= 65 else (Fore.YELLOW if score >= 50 else Fore.RED)
            print(f"Overall Position Score: {score_color}{score:.1f}/100{Style.RESET_ALL}")

            # Enhanced fundamental metrics
            if result.get('fundamentals'):
                fund = result['fundamentals']
                print(f"\n{Fore.YELLOW}KEY FUNDAMENTALS:{Style.RESET_ALL}")
                if fund.get('roic_estimate'):
                    print(f"ROIC Estimate: {fund['roic_estimate']:.2%}")
                if fund.get('graham_number'):
                    print(f"Graham Fair Value: ₹{fund['graham_number']:.2f}")
                if fund.get('quality_score'):
                    print(f"Quality Score: {fund['quality_score']:.0f}/100")

        def run_portfolio_analysis(trading_system):
            """Run comprehensive portfolio analysis"""
            print(f"\n{Fore.CYAN}PORTFOLIO ANALYSIS{Style.RESET_ALL}")

            symbols = trading_system.get_all_stock_symbols()[:15]  # Analyze top 15
            print(f"Analyzing {len(symbols)} stocks for portfolio construction...")

            stock_scores = {}
            for symbol in symbols:
                result = trading_system.analyze_position_trading_stock(symbol)
                if result:
                    stock_scores[symbol] = result['position_score']

            # Optimize allocation
            optimal_weights = trading_system.optimize_portfolio_allocation(stock_scores)

            # Display results
            print(f"\n{Fore.YELLOW}RECOMMENDED PORTFOLIO ALLOCATION:{Style.RESET_ALL}")
            print(f"{'Stock':<12} {'Score':<8} {'Weight':<8} {'Sector':<15}")
            print(f"{'-' * 50}")

            for symbol, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
                score = stock_scores[symbol]
                sector = trading_system.get_stock_info_from_db(symbol)['sector']
                print(f"{symbol:<12} {score:<8.1f} {weight:<8.1%} {sector:<15}")

        def run_backtest_menu(trading_system):
            """Backtest menu and execution"""
            print(f"\n{Fore.CYAN}BACKTEST ANALYSIS{Style.RESET_ALL}")

            try:
                symbols = input("Enter symbols (comma-separated) or press Enter for top 10: ").strip()
                if not symbols:
                    symbols = trading_system.get_all_stock_symbols()[:10]
                else:
                    symbols = [s.strip().upper() for s in symbols.split(',')]

                start_date = input("Start date (YYYY-MM-DD) or press Enter for 2020-01-01: ").strip()
                if not start_date:
                    start_date = "2020-01-01"

                end_date = input("End date (YYYY-MM-DD) or press Enter for 2023-12-31: ").strip()
                if not end_date:
                    end_date = "2023-12-31"

                print(f"\nRunning backtest on {len(symbols)} symbols...")
                backtest_results = trading_system.run_backtest_analysis(symbols, start_date, end_date)

                if backtest_results:
                    trading_system.display_backtest_results(backtest_results)
                else:
                    print(f"{Fore.RED}Backtest failed{Style.RESET_ALL}")

            except Exception as e:
                print(f"{Fore.RED}Error in backtest: {str(e)}{Style.RESET_ALL}")

        def display_backtest_results(results):
            """Display backtest results"""
            print(f"\n{Fore.YELLOW}BACKTEST RESULTS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")

            summary = results['report']['summary']
            for key, value in summary.items():
                print(f"{key.replace('_', ' ').title()}: {value}")

            # Best and worst trades
            if results['report'].get('best_trades'):
                print(f"\n{Fore.GREEN}TOP 3 TRADES:{Style.RESET_ALL}")
                for i, trade in enumerate(results['report']['best_trades'][:3], 1):
                    print(f"{i}. {trade['symbol']}: {trade['return']} ({trade['holding_period']})")

            if results['report'].get('worst_trades'):
                print(f"\n{Fore.RED}WORST 3 TRADES:{Style.RESET_ALL}")
                for i, trade in enumerate(results['report']['worst_trades'][:3], 1):
                    print(f"{i}. {trade['symbol']}: {trade['return']} ({trade['holding_period']})")