# unified_trading_platform.py
import sys
import os
import logging
from logging import Logger
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import warnings
import traceback
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import time
from colorama import Fore, Style, init

# Import your position trading code
from position_trading_code import EnhancedPositionTradingSystem, MDASentimentModel, SBERTTransformer, SBERT_AVAILABLE


class EnhancedPositionTradingSystem:
    """Enhanced Position Trading System for Indian Markets with Long-term Focus"""

    def __init__(self, model_path="D:\Python_files\trading_platforms\models\sentiment_pipeline_chunking.joblib",
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

    class ImprovedMDAExtractor:
        """Enhanced MD&A text extractor for Indian companies"""

        def __init__(self):
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })

        def get_mda_text(self, symbol: str, max_reports: int = 3) -> List[str]:
            """
            Extract real MD&A text from multiple sources
            Returns a list of MD&A text sections
            """
            try:
                symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
                logger.info(f"Extracting MD&A text for {symbol}")

                mda_texts = []

                # Try multiple extraction methods
                methods = [
                    self._extract_from_bse_announcements,
                    self._extract_from_nse_reports,
                    self._extract_from_company_website,
                    self._extract_from_annual_reports,
                    self._extract_from_yahoo_finance_filings
                ]

                for method in methods:
                    try:
                        texts = method(symbol)
                        if texts:
                            mda_texts.extend(texts)
                            logger.info(f"Successfully extracted {len(texts)} MD&A sections using {method.__name__}")

                            # If we have enough content, break
                            if len(mda_texts) >= max_reports:
                                break

                    except Exception as e:
                        logger.warning(f"Method {method.__name__} failed for {symbol}: {str(e)}")
                        continue

                    # Rate limiting
                    time.sleep(1)

                # Clean and filter the extracted texts
                cleaned_texts = self._clean_and_validate_mda_texts(mda_texts)

                if cleaned_texts:
                    logger.info(f"Successfully extracted {len(cleaned_texts)} MD&A texts for {symbol}")
                    return cleaned_texts[:max_reports]  # Return up to max_reports
                else:
                    logger.warning(f"No valid MD&A text found for {symbol}")
                    return []

            except Exception as e:
                logger.error(f"Error extracting MD&A text for {symbol}: {str(e)}")
                return []

        def _extract_from_bse_announcements(self, symbol: str) -> List[str]:
            """Extract MD&A from BSE announcements"""
            try:
                # BSE announcement URL pattern
                bse_url = f"https://www.bseindia.com/stock-share-price/{symbol}/announcements/"

                response = self.session.get(bse_url, timeout=10)
                if response.status_code != 200:
                    return []

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for annual report links or MD&A related announcements
                announcements = soup.find_all('a', href=re.compile(r'annual|report|mda|management.*discussion',
                                                                   re.IGNORECASE))

                mda_texts = []
                for link in announcements[:3]:  # Check first 3 relevant links
                    try:
                        href = link.get('href')
                        if href and not href.startswith('http'):
                            href = 'https://www.bseindia.com' + href

                        doc_response = self.session.get(href, timeout=10)
                        if doc_response.status_code == 200:
                            # Extract text from PDF or HTML
                            text = self._extract_text_from_response(doc_response)
                            mda_section = self._extract_mda_section(text)
                            if mda_section:
                                mda_texts.append(mda_section)

                    except Exception as e:
                        logger.debug(f"Error processing BSE link {href}: {str(e)}")
                        continue

                return mda_texts

            except Exception as e:
                logger.error(f"Error extracting from BSE for {symbol}: {str(e)}")
                return []

        def _extract_from_nse_reports(self, symbol: str) -> List[str]:
            """Extract MD&A from NSE corporate reports"""
            try:
                # NSE doesn't have a direct API, but we can try their corporate section
                nse_search_url = f"https://www.nseindia.com/companies-listing/corporate-filings-company-wise"

                # This would require more complex scraping with session management
                # For now, we'll implement a basic version

                return []  # Placeholder - NSE requires complex session handling

            except Exception as e:
                logger.error(f"Error extracting from NSE for {symbol}: {str(e)}")
                return []

        def _extract_from_company_website(self, symbol: str) -> List[str]:
            """Try to extract MD&A from company's official website"""
            try:
                # Get company website from yfinance
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info

                website = info.get('website', '')
                if not website:
                    return []

                # Look for investor relations section
                ir_urls = [
                    f"{website}/investor-relations",
                    f"{website}/investors",
                    f"{website}/annual-reports",
                    f"{website}/financial-reports"
                ]

                mda_texts = []
                for url in ir_urls:
                    try:
                        response = self.session.get(url, timeout=10)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Look for annual report links
                            report_links = soup.find_all('a', href=re.compile(r'annual.*report|financial.*report',
                                                                              re.IGNORECASE))

                            for link in report_links[:2]:  # Check first 2 reports
                                try:
                                    href = link.get('href')
                                    if href and not href.startswith('http'):
                                        href = website + href

                                    doc_response = self.session.get(href, timeout=15)
                                    if doc_response.status_code == 200:
                                        text = self._extract_text_from_response(doc_response)
                                        mda_section = self._extract_mda_section(text)
                                        if mda_section:
                                            mda_texts.append(mda_section)

                                except Exception as e:
                                    logger.debug(f"Error processing company website link: {str(e)}")
                                    continue

                    except Exception as e:
                        logger.debug(f"Error accessing {url}: {str(e)}")
                        continue

                return mda_texts

            except Exception as e:
                logger.error(f"Error extracting from company website for {symbol}: {str(e)}")
                return []

        def _extract_from_annual_reports(self, symbol: str) -> List[str]:
            """Extract MD&A from publicly available annual reports"""
            try:
                # Search for annual reports using Google Search API or web scraping
                search_query = f"{symbol} annual report filetype:pdf site:bseindia.com OR site:nseindia.com"

                # This is a simplified version - in practice, you'd use Google Search API
                # or implement more sophisticated web scraping

                return []  # Placeholder

            except Exception as e:
                logger.error(f"Error extracting from annual reports for {symbol}: {str(e)}")
                return []

        def _extract_from_yahoo_finance_filings(self, symbol: str) -> List[str]:
            """Extract MD&A information from Yahoo Finance filings data"""
            try:
                ticker = yf.Ticker(f"{symbol}.NS")

                # Get recent financial data and news
                info = ticker.info
                news = ticker.news

                # Extract relevant information from company description and recent news
                mda_like_texts = []

                # Company description often contains management perspective
                if 'longBusinessSummary' in info and info['longBusinessSummary']:
                    business_summary = info['longBusinessSummary']
                    if len(business_summary) > 200:  # Only if substantial content
                        mda_like_texts.append(business_summary)

                # Recent news articles that might contain management quotes
                for article in news[:5]:  # Check recent 5 articles
                    try:
                        if 'summary' in article and article['summary']:
                            summary = article['summary']
                            # Look for management-related content
                            if any(keyword in summary.lower() for keyword in
                                   ['management', 'ceo', 'outlook', 'strategy', 'expects', 'guidance']):
                                mda_like_texts.append(summary)
                    except Exception:
                        continue

                return mda_like_texts

            except Exception as e:
                logger.error(f"Error extracting from Yahoo Finance for {symbol}: {str(e)}")
                return []

        def _extract_text_from_response(self, response) -> str:
            """Extract text from HTTP response (HTML or PDF)"""
            try:
                content_type = response.headers.get('content-type', '').lower()

                if 'application/pdf' in content_type:
                    # Extract text from PDF
                    return self._extract_text_from_pdf(response.content)
                else:
                    # Extract text from HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    return soup.get_text()

            except Exception as e:
                logger.error(f"Error extracting text from response: {str(e)}")
                return ""

        def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
            """Extract text from PDF content"""
            try:
                import PyPDF2
                from io import BytesIO

                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                text = ""

                for page in pdf_reader.pages:
                    text += page.extract_text()

                return text

            except ImportError:
                logger.warning("PyPDF2 not available for PDF text extraction")
                return ""
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
                return ""

        def _extract_mda_section(self, full_text: str) -> Optional[str]:
            """Extract MD&A section from full document text"""
            try:
                if not full_text or len(full_text) < 100:
                    return None

                # Common MD&A section headers in Indian reports
                mda_patterns = [
                    r"management.*discussion.*and.*analysis",
                    r"management.*discussion",
                    r"directors.*report",
                    r"management.*analysis",
                    r"business.*outlook",
                    r"management.*commentary",
                    r"operational.*review",
                    r"management.*perspective"
                ]

                text_lower = full_text.lower()

                for pattern in mda_patterns:
                    matches = list(re.finditer(pattern, text_lower))

                    if matches:
                        # Find the start of MD&A section
                        start_pos = matches[0].start()

                        # Find the end (look for next major section or end of document)
                        end_patterns = [
                            r"financial.*statements",
                            r"notes.*to.*accounts",
                            r"auditor.*report",
                            r"corporate.*governance",
                            r"annexure",
                            r"schedule"
                        ]

                        end_pos = len(full_text)
                        for end_pattern in end_patterns:
                            end_matches = list(re.finditer(end_pattern, text_lower[start_pos:]))
                            if end_matches:
                                end_pos = start_pos + end_matches[0].start()
                                break

                        # Extract the section
                        mda_section = full_text[start_pos:end_pos]

                        # Clean and validate
                        if len(mda_section) > 500:  # Minimum length for meaningful MD&A
                            return self._clean_extracted_text(mda_section)

                # If no specific MD&A section found, look for management-related content
                management_content = self._extract_management_content(full_text)
                if management_content and len(management_content) > 300:
                    return management_content

                return None

            except Exception as e:
                logger.error(f"Error extracting MD&A section: {str(e)}")
                return None

        def _extract_management_content(self, text: str) -> Optional[str]:
            """Extract management-related content from text"""
            try:
                sentences = text.split('.')
                management_sentences = []

                management_keywords = [
                    'management', 'strategy', 'outlook', 'expects', 'believes',
                    'anticipates', 'guidance', 'performance', 'operations',
                    'future', 'growth', 'investment', 'market', 'business'
                ]

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 50:  # Minimum sentence length
                        sentence_lower = sentence.lower()
                        keyword_count = sum(1 for keyword in management_keywords if keyword in sentence_lower)

                        if keyword_count >= 2:  # At least 2 management-related keywords
                            management_sentences.append(sentence)

                if len(management_sentences) >= 5:  # At least 5 relevant sentences
                    return '. '.join(management_sentences[:20])  # Limit to 20 sentences

                return None

            except Exception as e:
                logger.error(f"Error extracting management content: {str(e)}")
                return None

        def _clean_extracted_text(self, text: str) -> str:
            """Clean and format extracted MD&A text"""
            try:
                # Remove excessive whitespace
                text = re.sub(r'\s+', ' ', text)

                # Remove page numbers and headers/footers
                text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
                text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)

                # Remove common report artifacts
                artifacts = [
                    r'annual report \d{4}',
                    r'page \d+',
                    r'www\.\w+\.com',
                    r'tel:?\s*\+?\d+[\d\s\-\(\)]+',
                    r'email:\s*\S+@\S+',
                ]

                for artifact in artifacts:
                    text = re.sub(artifact, '', text, flags=re.IGNORECASE)

                # Clean up spacing
                text = re.sub(r'\s+', ' ', text).strip()

                return text

            except Exception as e:
                logger.error(f"Error cleaning extracted text: {str(e)}")
                return text

        def _clean_and_validate_mda_texts(self, mda_texts: List[str]) -> List[str]:
            """Clean and validate extracted MD&A texts"""
            try:
                cleaned_texts = []

                for text in mda_texts:
                    if not text or not isinstance(text, str):
                        continue

                    # Clean the text
                    cleaned_text = self._clean_extracted_text(text)

                    # Validate quality
                    if self._validate_mda_text_quality(cleaned_text):
                        cleaned_texts.append(cleaned_text)

                # Remove duplicates (texts that are too similar)
                unique_texts = self._remove_similar_texts(cleaned_texts)

                return unique_texts

            except Exception as e:
                logger.error(f"Error cleaning and validating MD&A texts: {str(e)}")
                return mda_texts  # Return original if cleaning fails

        def _validate_mda_text_quality(self, text: str) -> bool:
            """Validate if the extracted text is meaningful MD&A content"""
            try:
                if not text or len(text) < 200:
                    return False

                # Check for minimum management-related keywords
                management_keywords = [
                    'management', 'performance', 'business', 'operations',
                    'growth', 'strategy', 'market', 'revenue', 'profit',
                    'outlook', 'expects', 'believes', 'future'
                ]

                text_lower = text.lower()
                keyword_count = sum(1 for keyword in management_keywords if keyword in text_lower)

                # Require at least 3 management-related keywords
                if keyword_count < 3:
                    return False

                # Check for reasonable sentence structure
                sentences = text.split('.')
                valid_sentences = [s for s in sentences if len(s.strip()) > 20]

                if len(valid_sentences) < 5:
                    return False

                return True

            except Exception as e:
                logger.error(f"Error validating MD&A text quality: {str(e)}")
                return False

        def _remove_similar_texts(self, texts: List[str]) -> List[str]:
            """Remove texts that are too similar to each other"""
            try:
                if len(texts) <= 1:
                    return texts

                unique_texts = []

                for text in texts:
                    is_unique = True

                    for existing_text in unique_texts:
                        # Simple similarity check based on common words
                        similarity = self._calculate_text_similarity(text, existing_text)
                        if similarity > 0.7:  # 70% similarity threshold
                            is_unique = False
                            break

                    if is_unique:
                        unique_texts.append(text)

                return unique_texts

            except Exception as e:
                logger.error(f"Error removing similar texts: {str(e)}")
                return texts

        def _calculate_text_similarity(self, text1: str, text2: str) -> float:
            """Calculate similarity between two texts"""
            try:
                # Simple word-based similarity
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())

                if not words1 and not words2:
                    return 1.0

                if not words1 or not words2:
                    return 0.0

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                return intersection / union if union > 0 else 0.0

            except Exception as e:
                logger.error(f"Error calculating text similarity: {str(e)}")
                return 0.0

    # Update the analyze_mda_sentiment method in your main class
    def updated_analyze_mda_sentiment(self, symbol):
        """
        Updated analyze_mda_sentiment method that prioritizes real MD&A extraction
        """
        try:
            if not self.mda_available:
                logger.info("MDA model not available, using sample analysis")
                return self.get_sample_mda_analysis(symbol)

            # ✅ Step 1: Try to fetch real MD&A text using improved extractor
            try:
                extractor = self.ImprovedMDAExtractor()
                mda_texts = extractor.get_mda_text(symbol, max_reports=3)

                if mda_texts:
                    logger.info(f"Successfully extracted {len(mda_texts)} real MD&A texts for {symbol}")
                else:
                    logger.warning(f"No real MD&A text found for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to extract real MDA text for {symbol}: {e}")
                mda_texts = []

            # ✅ Step 2: If no real text found, try alternative sources
            if not mda_texts:
                logger.info(f"Trying alternative sources for {symbol}")
                try:
                    # Try getting recent earnings transcripts or management commentary
                    alternative_texts = self._get_alternative_management_content(symbol)
                    if alternative_texts:
                        mda_texts = alternative_texts
                        logger.info(f"Found {len(mda_texts)} alternative management texts for {symbol}")
                except Exception as e:
                    logger.warning(f"Alternative content extraction failed: {e}")

            # ✅ Step 3: If still no real text, use sample but log it clearly
            if not mda_texts:
                logger.warning(f"No real MDA text found for {symbol}, using sample analysis")
                return self.get_sample_mda_analysis(symbol)

            # ✅ Step 4: Run your MDA sentiment model on real text
            try:
                sentiments, confidences = self.mda_sentiment_model.predict(mda_texts)

                if not sentiments or not confidences:
                    logger.warning(f"MDA sentiment analysis failed for {symbol}")
                    return self.get_sample_mda_analysis(symbol)

            except Exception as e:
                logger.error(f"MDA model prediction failed for {symbol}: {e}")
                return self.get_sample_mda_analysis(symbol)

            # ✅ Step 5: Convert sentiment to scores
            sentiment_scores = []
            for sentiment, confidence in zip(sentiments, confidences):
                if sentiment in ['positive', 'very_positive']:
                    sentiment_scores.append(confidence)
                elif sentiment in ['negative', 'very_negative']:
                    sentiment_scores.append(-confidence)
                else:
                    sentiment_scores.append(0)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            mda_score = 50 + (avg_sentiment * 50)  # scale to 0–100
            mda_score = max(0, min(100, mda_score))

            # Helper to get tone label
            management_tone = "Neutral"
            if mda_score >= 70:
                management_tone = "Very Optimistic"
            elif mda_score >= 60:
                management_tone = "Optimistic"
            elif mda_score <= 40:
                management_tone = "Pessimistic"

            return {
                'mda_score': mda_score,
                'sentiment_distribution': {
                    'positive': sentiments.count('positive') / len(sentiments) if sentiments else 0,
                    'negative': sentiments.count('negative') / len(sentiments) if sentiments else 0,
                    'neutral': sentiments.count('neutral') / len(sentiments) if sentiments else 0,
                },
                'management_tone': management_tone,
                'confidence': np.mean(confidences) if confidences else 0,
                'analysis_method': 'PyTorch BERT MDA Model (REAL TEXT)',
                'sample_texts_analyzed': len(mda_texts),
                'text_sources': 'Real MD&A extraction successful'
            }

        except Exception as e:
            logger.error(f"Error in MDA sentiment analysis for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return self.get_sample_mda_analysis(symbol)

    def _get_alternative_management_content(self, symbol):
        """
        Get alternative management content from earnings calls, press releases, etc.
        """
        try:
            alternative_texts = []

            # Method 1: Try to get recent earnings call transcripts
            try:
                # This would require integration with services like:
                # - AlphaVantage (has earnings call transcripts)
                # - Financial news APIs
                # - Company press releases

                # For now, we'll try to get management quotes from recent news
                ticker = yf.Ticker(f"{symbol}.NS")
                news = ticker.news

                management_quotes = []
                for article in news[:10]:  # Check recent 10 articles
                    try:
                        if 'summary' in article and article['summary']:
                            summary = article['summary']
                            # Look for quoted management statements
                            if any(keyword in summary.lower() for keyword in [
                                'ceo said', 'management said', 'according to', 'stated',
                                'commented', 'believes', 'expects', 'outlook'
                            ]):
                                management_quotes.append(summary)
                    except Exception:
                        continue

                if management_quotes:
                    alternative_texts.extend(management_quotes)

            except Exception as e:
                logger.debug(f"Error getting earnings content: {e}")

            return alternative_texts if len(alternative_texts) >= 2 else []

        except Exception as e:
            logger.error(f"Error getting alternative management content: {e}")
            return []

    def calculate_position_trading_score(self, data, sentiment_data, fundamentals, trends, market_analysis, sector,
                                         mda_analysis=None):
        """
        Calculate comprehensive position trading score with contextual sentiment
        and other dynamic modifiers.
        """
        try:
            # Get base weights for position trading (fundamentals-heavy)
            fundamental_weight = self.position_trading_params['fundamental_weight']
            technical_weight = self.position_trading_params['technical_weight']
            sentiment_weight = self.position_trading_params['sentiment_weight']
            mda_weight = self.position_trading_params['mda_weight']

            # 1. Calculate all individual base scores
            fundamental_score = self.calculate_fundamental_score(fundamentals, sector)
            technical_score = self.calculate_technical_score_position(data)
            sentiment_score = self.calculate_sentiment_score(sentiment_data)
            trend_score = trends.get('trend_score', 50)
            sector_score = market_analysis.get('sector_score', 60)
            mda_score = mda_analysis.get('mda_score', 50) if mda_analysis and isinstance(mda_analysis, dict) else 50

            # 2. MODIFICATION: Apply the contextual sentiment multiplier based on the sector
            sector_sentiment_multipliers = {
                'Information Technology': 1.2,  # News is highly impactful
                'Consumer Goods': 1.1,
                'Financial Services': 1.1,
                'Pharmaceuticals': 1.2,  # Regulatory news is critical
                'Power': 0.8,  # More stable, less news-driven
                'Oil & Gas': 1.0,
                'Default': 1.0
            }
            sentiment_multiplier = sector_sentiment_multipliers.get(sector, sector_sentiment_multipliers['Default'])
            contextual_sentiment_score = sentiment_score * sentiment_multiplier
            logger.info(f"Applying sentiment multiplier of {sentiment_multiplier} for {sector} sector.")

            # 3. Combine scores using the CONTEXTUAL sentiment score
            base_score = (
                    fundamental_score * fundamental_weight +
                    technical_score * technical_weight +
                    contextual_sentiment_score * sentiment_weight +  # <-- Use the adjusted score here
                    mda_score * mda_weight
            )

            # 4. Apply trend, sector, and other specific modifiers to the final score
            trend_modifier = trend_score / 100
            sector_modifier = sector_score / 100
            final_score = base_score * (0.7 + 0.2 * trend_modifier + 0.1 * sector_modifier)

            # Penalize high volatility stocks
            if data is not None and not data.empty and 'Close' in data.columns:
                volatility = data['Close'].pct_change().std() * np.sqrt(252)
                if volatility > 0.4:
                    final_score *= 0.8
                elif volatility > 0.6:
                    final_score *= 0.6

            # Bonus for dividend-paying stocks
            div_yield = fundamentals.get('dividend_yield', 0) or fundamentals.get('expected_div_yield', 0)
            if div_yield and div_yield > 0.02:
                final_score *= 1.1

            # Bonus for consistent long-term performance
            if trends.get('momentum_1y', 0) > 0.15 and trends.get('momentum_6m', 0) > 0:
                final_score *= 1.05

            # MDA sentiment bonus/penalty
            if mda_analysis:
                management_tone = mda_analysis.get('management_tone', 'Neutral')
                if management_tone == 'Very Optimistic':
                    final_score *= 1.08
                elif management_tone == 'Optimistic':
                    final_score *= 1.04
                elif management_tone == 'Pessimistic':
                    final_score *= 0.92

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
                # *** FIX 1: Corrected method name ***
                mda_analysis = self.updated_analyze_mda_sentiment(final_symbol)
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
    # In class EnhancedPositionTradingSystem:

    def create_personalized_portfolio(self, risk_appetite, time_period_months, budget):
        """Create a personalized portfolio using risk-based position sizing."""
        try:
            min_score = 65  # Set a minimum score for a trade to even be considered

            symbols = self.get_all_stock_symbols()
            stock_results = []

            print(f"\nAnalyzing {len(symbols)} stocks for your portfolio...")
            for symbol in symbols:
                result = self.analyze_position_trading_stock(symbol)
                # Only consider stocks with a BUY signal and a high enough score
                if result and result.get('position_score', 0) >= min_score and \
                        result.get('trading_plan', {}).get('entry_signal') in ['BUY', 'STRONG BUY']:
                    stock_results.append(result)

            # Sort by score to prioritize the best setups first
            sorted_stocks = sorted(stock_results, key=lambda x: x['position_score'], reverse=True)

            if not sorted_stocks:
                return {"error": "No stocks meet the minimum criteria for your risk profile."}

            # --- UPDATED: Pass the entire budget to the new risk-based calculator ---
            portfolio = self.calculate_position_sizes(sorted_stocks, budget)

            if not portfolio:
                return {"error": "Could not create a portfolio with the given risk parameters and budget."}

            summary = self.generate_portfolio_summary(portfolio, time_period_months)

            return {
                'portfolio': portfolio,
                'summary': summary,
                'risk_profile': risk_appetite,
                'time_period_months': time_period_months,
                'budget': budget
            }

        except Exception as e:
            logger.error(f"Error creating personalized portfolio: {str(e)}")
            return {"error": str(e)}

    def calculate_position_sizes(self, selected_stocks, total_capital):
        """
        --- COMPLETELY REWRITTEN ---
        Calculate position sizes based on a fixed risk percentage of total capital.
        """
        portfolio = {}

        # Get the risk per trade from your class parameters (e.g., 0.01 for 1%)
        risk_per_trade_pct = self.position_trading_params['risk_per_trade']
        capital_at_risk_per_trade = total_capital * risk_per_trade_pct

        total_allocated = 0

        for stock_data in selected_stocks:
            try:
                current_price = stock_data.get('current_price', 0)
                trading_plan = stock_data.get('trading_plan', {})
                stop_loss = trading_plan.get('stop_loss', 0)

                # Validate data for this trade
                if current_price <= 0 or stop_loss <= 0 or current_price <= stop_loss:
                    continue

                # --- Core Risk-Based Calculation ---
                risk_per_share = current_price - stop_loss
                num_shares = int(capital_at_risk_per_trade / risk_per_share)

                if num_shares == 0:
                    continue  # Cannot afford even one share with this risk model

                investment_amount = num_shares * current_price

                # Ensure we don't allocate more than the total available capital
                if total_allocated + investment_amount > total_capital:
                    continue  # Skip trade if it exceeds total budget

                total_allocated += investment_amount

                symbol = stock_data.get('symbol', 'Unknown')
                portfolio[symbol] = {
                    'company_name': stock_data.get('company_name', 'Unknown'),
                    'sector': stock_data.get('sector', 'Unknown'),
                    'score': stock_data.get('position_score', 0),
                    'num_shares': num_shares,
                    'investment_amount': investment_amount,
                    'stop_loss': stop_loss,
                    'targets': trading_plan.get('targets')
                }

            except Exception as e:
                logger.error(f"Error sizing position for {stock_data.get('symbol')}: {e}")
                continue

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

    # *** FIX 2: Added the missing method ***
    def get_sample_mda_analysis(self, symbol):
        """Generate sample MDA analysis for demonstration or fallback"""
        try:
            # Generate a score based on a hash of the symbol for consistency
            base_score = 50 + (hash(symbol) % 25)
            tone_map = {
                (0, 45): "Pessimistic",
                (45, 55): "Neutral",
                (55, 65): "Optimistic",
                (65, 100): "Very Optimistic"
            }
            management_tone = "Neutral"
            for (lower, upper), tone in tone_map.items():
                if lower <= base_score < upper:
                    management_tone = tone
                    break

            return {
                'mda_score': base_score,
                'sentiment_distribution': {'positive': 0.4, 'negative': 0.1, 'neutral': 0.5},
                'management_tone': management_tone,
                'confidence': 0.75,
                'analysis_method': 'Sample MDA Analysis (Fallback)',
                'sample_texts_analyzed': 0,
                'text_sources': 'No real text found; using sample data.'
            }
        except Exception as e:
            logger.error(f"Error generating sample MDA analysis for {symbol}: {str(e)}")
            return {'mda_score': 50, 'management_tone': 'Neutral', 'analysis_method': 'Error'}

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: Logger = logging.getLogger(__name__)


# ======================================================================================
# FIX: The entire 'EnhancedSwingTradingSystem' class block below has been un-indented
# to fix the 'Unexpected indent' error.
# ======================================================================================

# Swing Trading System Class
class EnhancedSwingTradingSystem:
    """Enhanced Swing Trading System for Indian Markets with Budget & Risk Management"""

    def __init__(self, model_path="D:\Python_files\trading_platforms\models\sentiment_pipeline_chunking.joblib", news_api_key=None):
        """Initialize the swing trading system with comprehensive error handling"""
        try:
            self.sentiment_pipeline = None
            self.vectorizer = None
            self.model = None
            self.label_encoder = None
            self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY") or "dd33ebe105ea4b02a3b7e77bc4a93d01"

            self.model_loaded = False
            self.model_type = "None"

            # --- STRATEGY IMPROVEMENT: RISK PARAMETER ---
            # This parameter is now central to position sizing to control drawdown.
            self.swing_trading_params = {
                'min_holding_period': 3,  # days
                'max_holding_period': 30,  # days
                'risk_per_trade': 0.02,  # CRITICAL: Risk only 2% of total capital per trade.
                'max_portfolio_risk': 0.10,  # 10% max portfolio risk
                'profit_target_multiplier': 2.5,  # Risk-reward ratio
            }

            self._validate_trading_params()

            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not provided. Using sample news for sentiment analysis.")
            else:
                logger.info("News API key available. Will fetch real news articles.")

            self.load_sbert_model(model_path)
            self.initialize_stock_database()

            logger.info("EnhancedSwingTradingSystem initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing EnhancedSwingTradingSystem: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _validate_trading_params(self):
        """Validate trading parameters"""
        try:
            required_params = ['min_holding_period', 'max_holding_period', 'risk_per_trade',
                               'max_portfolio_risk', 'profit_target_multiplier']

            for param in required_params:
                if param not in self.swing_trading_params:
                    raise ValueError(f"Missing required trading parameter: {param}")

                value = self.swing_trading_params[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Invalid trading parameter {param}: {value}")

            # Additional validation
            if self.swing_trading_params['min_holding_period'] >= self.swing_trading_params['max_holding_period']:
                raise ValueError("min_holding_period must be less than max_holding_period")

            if self.swing_trading_params['risk_per_trade'] > 0.1:  # 10% max risk per trade
                raise ValueError("risk_per_trade cannot exceed 10%")

            logger.info("Trading parameters validated successfully")

        except Exception as e:
            logger.error(f"Error validating trading parameters: {str(e)}")
            raise

    def initialize_stock_database(self):
        """Initialize comprehensive Indian stock database (BSE + NSE) with backtest-based improvements"""
        try:
            self.indian_stocks = {
                # NIFTY 50 Stocks
                "RELIANCE": {"name": "Reliance Industries", "sector": "Oil & Gas"},
                "TCS": {"name": "Tata Consultancy Services", "sector": "Information Technology"},
                "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking"},
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
                "LT": {"name": "Larsen & Toubro", "sector": "Construction"},
                "SBIN": {"name": "State Bank of India", "sector": "Banking"},
                "BHARTIARTL": {"name": "Bharti Airtel", "sector": "Telecommunications"},
                "ASIANPAINT": {"name": "Asian Paints", "sector": "Consumer Goods"},
                "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile"},
                "TITAN": {"name": "Titan Company", "sector": "Consumer Goods"},
                "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals"},
                "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "Cement"},
                "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods"},
                "HCLTECH": {"name": "HCL Technologies", "sector": "Information Technology"},
                "AXISBANK": {"name": "Axis Bank", "sector": "Banking"},
                "WIPRO": {"name": "Wipro", "sector": "Information Technology"},
                "NTPC": {"name": "NTPC", "sector": "Power"},
                "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power"},
                "ONGC": {"name": "Oil & Natural Gas Corporation", "sector": "Oil & Gas"},
                "TECHM": {"name": "Tech Mahindra", "sector": "Information Technology"},
                "TATASTEEL": {"name": "Tata Steel", "sector": "Steel"},
                "ADANIENT": {"name": "Adani Enterprises", "sector": "Conglomerate"},
                "COALINDIA": {"name": "Coal India", "sector": "Mining"},
                "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals"},
                "JSWSTEEL": {"name": "JSW Steel", "sector": "Steel"},
                "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile"},
                "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile"},
                "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile"},
                "GRASIM": {"name": "Grasim Industries", "sector": "Cement"},
                "SHREECEM": {"name": "Shree Cement", "sector": "Cement"},
                "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile"},
                "UPL": {"name": "UPL Limited", "sector": "Chemicals"},
                "BPCL": {"name": "Bharat Petroleum", "sector": "Oil & Gas"},
                "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals"},
                "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Pharmaceuticals"},
                "CIPLA": {"name": "Cipla", "sector": "Pharmaceuticals"},
                "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods"},
                "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods"},
                "IOC": {"name": "Indian Oil Corporation", "sector": "Oil & Gas"},
                "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Healthcare"},
                "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services"},
                "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Insurance"},
                "SBILIFE": {"name": "SBI Life Insurance", "sector": "Insurance"},
                "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking"},
                "ADANIPORTS": {"name": "Adani Ports", "sector": "Infrastructure"},
                "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile"},
                "ITC": {"name": "ITC Limited", "sector": "Consumer Goods"},
                "GODREJCP": {"name": "Godrej Consumer Products", "sector": "Consumer Goods"},
                "COLPAL": {"name": "Colgate-Palmolive India", "sector": "Consumer Goods"},
                "PIDILITIND": {"name": "Pidilite Industries", "sector": "Chemicals"},
                "BAJAJHLDNG": {"name": "Bajaj Holdings", "sector": "Financial Services"},
                "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods"},
                "DABUR": {"name": "Dabur India", "sector": "Consumer Goods"},
                "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals"},
                "CADILAHC": {"name": "Cadila Healthcare", "sector": "Pharmaceuticals"},
                "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals"},
                "ALKEM": {"name": "Alkem Laboratories", "sector": "Pharmaceuticals"},
                "TORNTPHARM": {"name": "Torrent Pharmaceuticals", "sector": "Pharmaceuticals"},
                "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharmaceuticals"},
                "MOTHERSUMI": {"name": "Motherson Sumi Systems", "sector": "Automobile"},
                "BOSCHLTD": {"name": "Bosch Limited", "sector": "Automobile"},
                "EXIDEIND": {"name": "Exide Industries", "sector": "Automobile"},
                "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile"},
                "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile"},
                "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile"},
                "MRF": {"name": "MRF Limited", "sector": "Automobile"},
                "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile"},
                "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile"},
                "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking"},
                "BANDHANBNK": {"name": "Bandhan Bank", "sector": "Banking"},
                "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Banking"},
                "RBLBANK": {"name": "RBL Bank", "sector": "Banking"},
                "YESBANK": {"name": "Yes Bank", "sector": "Banking"},
                "PNB": {"name": "Punjab National Bank", "sector": "Banking"},
                "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking"},
                "CANBK": {"name": "Canara Bank", "sector": "Banking"},
                "UNIONBANK": {"name": "Union Bank of India", "sector": "Banking"},
                "CHOLAFIN": {"name": "Cholamandalam Investment", "sector": "Financial Services"},
                "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services"},
                "MANAPPURAM": {"name": "Manappuram Finance", "sector": "Financial Services"},
                "MMFIN": {"name": "Mahindra & Mahindra Financial", "sector": "Financial Services"},
                "SRTRANSFIN": {"name": "Shriram Transport Finance", "sector": "Financial Services"},
                "MINDTREE": {"name": "Mindtree Limited", "sector": "Information Technology"},
                "LTTS": {"name": "L&T Technology Services", "sector": "Information Technology"},
                "PERSISTENT": {"name": "Persistent Systems", "sector": "Information Technology"},
                "CYIENT": {"name": "Cyient Limited", "sector": "Information Technology"},
                "NIITTECH": {"name": "NIIT Technologies", "sector": "Information Technology"},
                "ROLTA": {"name": "Rolta India", "sector": "Information Technology"},
                "HEXATECHNO": {"name": "Hexa Technologies", "sector": "Information Technology"},
                "COFORGE": {"name": "Coforge Limited", "sector": "Information Technology"},
                "DMART": {"name": "Avenue Supermarts", "sector": "Retail"},
                "TRENT": {"name": "Trent Limited", "sector": "Retail"},
                "PAGEIND": {"name": "Page Industries", "sector": "Textiles"},
                "RAYMOND": {"name": "Raymond Limited", "sector": "Textiles"},
                "VBL": {"name": "Varun Beverages", "sector": "Consumer Goods"},
                "EMAMILTD": {"name": "Emami Limited", "sector": "Consumer Goods"},
                "JUBLFOOD": {"name": "Jubilant FoodWorks", "sector": "Consumer Goods"},
            }

            # --- STRATEGY IMPROVEMENT BASED ON BACKTEST ---
            # The backtest for the Swing Trading strategy showed consistent losses
            # on these specific low-volatility, blue-chip stocks. They are being
            # excluded from this strategy's universe to improve performance.
            symbols_to_exclude = ['RELIANCE', 'HDFCBANK', 'TCS']

            original_count = len(self.indian_stocks)

            self.indian_stocks = {
                symbol: info
                for symbol, info in self.indian_stocks.items()
                if symbol not in symbols_to_exclude
            }

            logger.info(
                f"Excluded {len(symbols_to_exclude)} underperforming symbols based on backtest. "
                f"Universe size reduced from {original_count} to {len(self.indian_stocks)}."
            )

            if not self.indian_stocks:
                raise ValueError("Stock database initialization failed - empty database")

        except Exception as e:
            logger.error(f"Error initializing stock database: {str(e)}")
            # Fallback to minimal database
            self.indian_stocks = {
                "INFY": {"name": "Infosys", "sector": "Information Technology"},
                "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking"},
                "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services"},
            }
            logger.warning(f"Using fallback database with {len(self.indian_stocks)} stocks")

    def get_all_stock_symbols(self):
        """Get all stock symbols for analysis with error handling"""
        try:
            if not self.indian_stocks:
                raise ValueError("Stock database is empty")
            return list(self.indian_stocks.keys())
        except Exception as e:
            logger.error(f"Error getting stock symbols: {str(e)}")
            return ["RELIANCE", "TCS", "HDFCBANK"]  # Fallback symbols

    def get_stock_info_from_db(self, symbol):
        """Get stock information from internal database with error handling"""
        try:
            if not symbol:
                raise ValueError("Empty symbol provided")

            base_symbol = str(symbol).split('.')[0].upper().strip()
            if not base_symbol:
                raise ValueError("Invalid symbol format")

            return self.indian_stocks.get(base_symbol, {"name": symbol, "sector": "Unknown"})
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {"name": str(symbol), "sector": "Unknown"}

    def load_sbert_model(self, model_path):
        """Load trained SBERT sentiment model with comprehensive error handling"""
        try:
            if not SBERT_AVAILABLE:
                logger.warning("sentence-transformers not available, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not model_path:
                logger.warning("No model path provided, using TextBlob fallback")
                self.model_type = "TextBlob"
                return

            if not os.path.exists(model_path):
                logger.warning(f"SBERT model not found at {model_path}")
                logger.info("Using TextBlob as fallback for sentiment analysis")
                self.model_type = "TextBlob"
                return

            logger.info(f"Loading SBERT sentiment model from {model_path}...")

            # Load with timeout protection
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
                self.model_type = "SBERT + RandomForest"
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

    def get_sector_weights(self, sector):
        """Get dynamic weights based on sector for swing trading with error handling"""
        try:
            if not sector:
                logger.warning("Empty sector provided, using default weights")
                return 0.55, 0.45

            sector = str(sector).lower().strip()

            # Swing trading weights (balanced approach)
            tech_weight, sentiment_weight = 0.55, 0.45

            weights_map = {
                "technology": (0.45, 0.55),  # Tech more sentiment driven
                "information technology": (0.45, 0.55),
                "tech": (0.45, 0.55),
                "it": (0.45, 0.55),
                "financial": (0.60, 0.40),  # Finance more technical
                "financial services": (0.60, 0.40),
                "banking": (0.60, 0.40),
                "finance": (0.60, 0.40),
                "consumer staples": (0.65, 0.35),
                "staples": (0.65, 0.35),
                "consumer goods": (0.65, 0.35),
                "food & staples retailing": (0.65, 0.35),
                "energy": (0.55, 0.45),
                "oil & gas": (0.55, 0.45),
                "utilities": (0.70, 0.30),
                "electric": (0.70, 0.30),
                "power": (0.70, 0.30),
                "healthcare": (0.50, 0.50),
                "pharmaceuticals": (0.50, 0.50),
                "health care": (0.50, 0.50),
                "pharma": (0.50, 0.50),
                "consumer discretionary": (0.45, 0.55),
                "consumer cyclicals": (0.45, 0.55),
                "retail": (0.45, 0.55),
                "automobile": (0.45, 0.55),
                "auto": (0.45, 0.55),
            }

            for key, weights in weights_map.items():
                if key in sector:
                    tech_weight, sentiment_weight = weights
                    break

            # Validate weights
            if tech_weight + sentiment_weight != 1.0:
                logger.warning(f"Invalid weights for sector {sector}, using defaults")
                return 0.55, 0.45

            return tech_weight, sentiment_weight

        except Exception as e:
            logger.error(f"Error getting sector weights for {sector}: {str(e)}")
            return 0.55, 0.45  # Default weights

    def get_indian_stock_data(self, symbol, period="6mo"):
        """Get Indian stock data with extended period for swing trading and comprehensive error handling"""
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

                    if len(data) < 30:  # Need more data for swing trading
                        logger.warning(f"Insufficient data for {sym}: {len(data)} days")
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

    def safe_rolling_calculation(self, data, window, operation='mean'):
        """Safely perform rolling calculations with error handling"""
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

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands with comprehensive error handling"""
        try:
            if prices is None or prices.empty:
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series, empty_series

            if len(prices) < period:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            sma = self.safe_rolling_calculation(prices, period, 'mean')
            std = self.safe_rolling_calculation(prices, period, 'std')

            if sma.empty or std.empty:
                nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
                return nan_series, nan_series, nan_series

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return upper_band, sma, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            nan_series = pd.Series([np.nan] * len(prices),
                                   index=prices.index if hasattr(prices, 'index') else range(len(prices)))
            return nan_series, nan_series, nan_series

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                empty_series = pd.Series(dtype=float)
                return empty_series, empty_series

            if len(close) < k_period:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            lowest_low = self.safe_rolling_calculation(low, k_period, 'min')
            highest_high = self.safe_rolling_calculation(high, k_period, 'max')

            if lowest_low.empty or highest_high.empty:
                nan_series = pd.Series([np.nan] * len(close), index=close.index)
                return nan_series, nan_series

            # Avoid division by zero
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, np.nan)

            k_percent = 100 * ((close - lowest_low) / denominator)
            d_percent = self.safe_rolling_calculation(k_percent, d_period, 'mean')

            return k_percent, d_percent

        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            nan_series = pd.Series([np.nan] * len(close),
                                   index=close.index if hasattr(close, 'index') else range(len(close)))
            return nan_series, nan_series

    def calculate_support_resistance(self, data, window=20):
        """Calculate support and resistance levels with error handling"""
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
                    # Check if current high is a local maximum
                    if not pd.isna(highs.iloc[i]) and data['High'].iloc[i] == highs.iloc[i]:
                        resistance_levels.append(data['High'].iloc[i])

                    # Check if current low is a local minimum
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

    def calculate_volume_profile(self, data, bins=20):
        """Calculate Volume Profile with error handling"""
        try:
            if data is None or data.empty or 'Volume' not in data.columns:
                return None, None

            if 'High' not in data.columns or 'Low' not in data.columns:
                return None, None

            price_range = data['High'].max() - data['Low'].min()
            if price_range <= 0:
                return None, None

            bin_size = price_range / bins
            volume_profile = {}

            for i in range(len(data)):
                try:
                    price = (data['High'].iloc[i] + data['Low'].iloc[i]) / 2
                    volume = data['Volume'].iloc[i]

                    if pd.isna(price) or pd.isna(volume) or volume <= 0:
                        continue

                    bin_level = int((price - data['Low'].min()) / bin_size)
                    bin_level = min(bin_level, bins - 1)
                    bin_level = max(bin_level, 0)

                    price_level = data['Low'].min() + (bin_level * bin_size)

                    if price_level not in volume_profile:
                        volume_profile[price_level] = 0
                    volume_profile[price_level] += volume
                except Exception as e:
                    logger.warning(f"Error processing volume at index {i}: {str(e)}")
                    continue

            if not volume_profile:
                return None, None

            # Find Point of Control (POC) - highest volume level
            poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x])
            poc_volume = volume_profile[poc_price]

            return volume_profile, poc_price

        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return None, None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with comprehensive error handling"""
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

            # Avoid division by zero
            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)  # Fill NaN with neutral RSI

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index if hasattr(prices, 'index') else range(len(prices)))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD with error handling"""
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

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range with error handling"""
        try:
            if any(x is None or x.empty for x in [high, low, close]):
                return pd.Series(dtype=float)

            if len(close) < period:
                return pd.Series([np.nan] * len(close), index=close.index)

            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self.safe_rolling_calculation(tr, period, 'mean')

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series([np.nan] * len(close), index=close.index if hasattr(close, 'index') else range(len(close)))

    def fetch_indian_news(self, symbol, num_articles=15):
        """Fetch news for Indian companies with error handling"""
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

    def get_sample_news(self, symbol):
        """Generate sample news for demonstration with error handling"""
        try:
            base_symbol = str(symbol).split('.')[0]
            stock_info = self.get_stock_info_from_db(base_symbol)
            company_name = stock_info.get("name", base_symbol)

            return [
                f"{company_name} reports strong quarterly earnings beating estimates",
                f"Analysts upgrade {company_name} target price citing strong fundamentals",
                f"{company_name} announces major expansion plans and new product launches",
                f"Regulatory approval boosts {company_name} market position",
                f"{company_name} forms strategic partnership with global leader",
                f"Market volatility creates buying opportunity in {company_name}",
                f"{company_name} invests heavily in R&D and digital transformation",
                f"Industry experts bullish on {company_name} long-term prospects",
                f"Competitive pressure intensifies for {company_name} in key markets",
                f"Strong domestic demand drives {company_name} revenue growth",
                f"{company_name} management provides optimistic guidance for next quarter",
                f"Foreign institutional investors increase stake in {company_name}",
                f"Technical breakout signals potential upside for {company_name}",
                f"{company_name} benefits from favorable government policy changes",
                f"Sector rotation favors {company_name} business model"
            ]
        except Exception as e:
            logger.error(f"Error generating sample news for {symbol}: {str(e)}")
            return [f"Market analysis for {symbol}", f"Investment opportunity in {symbol}"]

    def analyze_sentiment_with_sbert(self, articles):
        """Analyze sentiment using trained SBERT model with error handling"""
        try:
            if not articles or not self.model_loaded:
                return self.analyze_sentiment_with_textblob(articles)

            if not all([self.vectorizer, self.model, self.label_encoder]):
                logger.error("SBERT model components missing")
                return self.analyze_sentiment_with_textblob(articles)

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
            logger.error(f"Error in SBERT sentiment analysis: {str(e)}")
            return self.analyze_sentiment_with_textblob(articles)

    def analyze_sentiment_with_textblob(self, articles):
        """Fallback sentiment analysis using TextBlob with error handling"""
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

    def analyze_news_sentiment(self, symbol, num_articles=15):
        """Main sentiment analysis function with comprehensive error handling"""
        try:
            articles = self.fetch_indian_news(symbol, num_articles)
            news_source = "Real news (NewsAPI)" if articles else "Sample news"

            if not articles:
                articles = self.get_sample_news(symbol)

            if not articles:
                logger.error(f"No articles available for {symbol}")
                return [], [], [], "No Analysis", "No Source"

            if self.model_loaded:
                sentiments, confidences = self.analyze_sentiment_with_sbert(articles)
                analysis_method = f"SBERT Model ({self.model_type})"
            else:
                sentiments, confidences = self.analyze_sentiment_with_textblob(articles)
                analysis_method = "TextBlob Fallback"

            return sentiments, articles, confidences, analysis_method, news_source

        except Exception as e:
            logger.error(f"Error in news sentiment analysis for {symbol}: {str(e)}")
            return [], [], [], "Error", "Error"

    def calculate_swing_trading_score(self, data, sentiment_data, sector):
        """Calculate comprehensive swing trading score with error handling"""
        try:
            tech_weight, sentiment_weight = self.get_sector_weights(sector)

            # Initialize components
            technical_score = 0
            sentiment_score = 50  # Default neutral sentiment

            if data is None or data.empty:
                logger.error("No data provided for scoring")
                return 0

            # ===== TECHNICAL ANALYSIS (Enhanced for Swing Trading) =====
            try:
                current_price = data['Close'].iloc[-1]
                if pd.isna(current_price) or current_price <= 0:
                    logger.error("Invalid current price")
                    return 0
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                return 0

            # RSI Analysis (20 points)
            try:
                rsi = self.calculate_rsi(data['Close'])
                if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                    current_rsi = rsi.iloc[-1]
                    if 30 <= current_rsi <= 70:  # Good for swing trading
                        technical_score += 20
                    elif current_rsi < 30:  # Oversold - potential reversal
                        technical_score += 15
                    elif current_rsi > 70:  # Overbought - potential reversal
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")

            # Bollinger Bands Analysis (15 points)
            try:
                bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
                if not bb_upper.empty and not any(pd.isna([bb_upper.iloc[-1], bb_lower.iloc[-1]])):
                    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                    if 0.2 <= bb_position <= 0.8:  # Good swing trading zone
                        technical_score += 15
                    elif bb_position < 0.2:  # Near lower band - potential buy
                        technical_score += 12
                    elif bb_position > 0.8:  # Near upper band - potential sell
                        technical_score += 8
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}")

            # Stochastic Analysis (15 points)
            try:
                stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
                if not stoch_k.empty and not any(pd.isna([stoch_k.iloc[-1], stoch_d.iloc[-1]])):
                    k_val = stoch_k.iloc[-1]
                    d_val = stoch_d.iloc[-1]
                    if k_val > d_val and k_val < 80:  # Bullish crossover
                        technical_score += 15
                    elif 20 <= k_val <= 80:  # Good swing range
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating Stochastic: {str(e)}")

            # MACD Analysis (15 points)
            try:
                macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
                if not macd_line.empty and not any(pd.isna([macd_line.iloc[-1], signal_line.iloc[-1]])):
                    if macd_line.iloc[-1] > signal_line.iloc[-1]:  # Bullish
                        technical_score += 15
                    if len(histogram) > 1 and not any(pd.isna([histogram.iloc[-1], histogram.iloc[-2]])):
                        if histogram.iloc[-1] > histogram.iloc[-2]:  # Increasing momentum
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")

            # Volume Analysis (10 points)
            try:
                if 'Volume' in data.columns:
                    avg_volume = self.safe_rolling_calculation(data['Volume'], 20, 'mean').iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    if not pd.isna(avg_volume) and not pd.isna(current_volume) and avg_volume > 0:
                        if current_volume > avg_volume * 1.2:  # Above average volume
                            technical_score += 10
                        elif current_volume > avg_volume:
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating volume: {str(e)}")

            # Support/Resistance Analysis (10 points)
            try:
                support, resistance = self.calculate_support_resistance(data)
                if support and resistance and not any(pd.isna([support, resistance])):
                    distance_to_support = (current_price - support) / support
                    distance_to_resistance = (resistance - current_price) / current_price

                    if distance_to_support < 0.05:  # Near support
                        technical_score += 8
                    elif distance_to_resistance < 0.05:  # Near resistance
                        technical_score += 5
                    elif 0.05 <= distance_to_support <= 0.15:  # Good swing zone
                        technical_score += 10
            except Exception as e:
                logger.warning(f"Error calculating support/resistance: {str(e)}")

            # Moving Average Analysis (15 points)
            try:
                if len(data) >= 50:
                    ma_20 = self.safe_rolling_calculation(data['Close'], 20, 'mean').iloc[-1]
                    ma_50 = self.safe_rolling_calculation(data['Close'], 50, 'mean').iloc[-1]
                    if not any(pd.isna([ma_20, ma_50])):
                        if current_price > ma_20 > ma_50:  # Strong uptrend
                            technical_score += 15
                        elif current_price > ma_20:  # Above short-term MA
                            technical_score += 10
                        elif ma_20 > ma_50:  # MA alignment positive
                            technical_score += 5
            except Exception as e:
                logger.warning(f"Error calculating moving averages: {str(e)}")

            # Normalize technical score to 0-100
            technical_score = min(100, max(0, technical_score))

            # ===== SENTIMENT ANALYSIS =====
            try:
                if sentiment_data and len(sentiment_data) >= 3:
                    sentiments, _, confidences, _, _ = sentiment_data
                    if sentiments and confidences:
                        sentiment_value = 0
                        total_weight = 0

                        for sentiment, confidence in zip(sentiments, confidences):
                            weight = confidence if not pd.isna(confidence) else 0.5
                            if sentiment == 'positive':
                                sentiment_value += weight
                            elif sentiment == 'negative':
                                sentiment_value -= weight
                            total_weight += weight

                        if total_weight > 0:
                            normalized_sentiment = sentiment_value / total_weight
                            sentiment_score = 50 + (normalized_sentiment * 50)
                        else:
                            sentiment_score = 50
                    else:
                        sentiment_score = 50
                else:
                    sentiment_score = 50
            except Exception as e:
                logger.warning(f"Error calculating sentiment score: {str(e)}")
                sentiment_score = 50

            # ===== COMBINE SCORES =====
            sentiment_score = min(100, max(0, sentiment_score))
            final_score = (technical_score * tech_weight) + (sentiment_score * sentiment_weight)
            final_score = min(100, max(0, final_score))

            return final_score

        except Exception as e:
            logger.error(f"Error calculating swing trading score: {str(e)}")
            return 0

    def calculate_risk_metrics(self, data):
        """Calculate risk management metrics with comprehensive error handling"""
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

            # Sharpe Ratio (assuming 6% risk-free rate)
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
                atr = self.calculate_atr(data['High'], data['Low'], data['Close'])
                current_atr = atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else data['Close'].iloc[
                                                                                                   -1] * 0.02
            except Exception:
                current_atr = data['Close'].iloc[-1] * 0.02 if not data['Close'].empty else 0

            # Risk level determination
            try:
                if volatility > 0.4:
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
                'atr': current_atr,
                'risk_level': risk_level
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return default_metrics

    def generate_trading_plan(self, data, score, risk_metrics):
        """Generate complete trading plan with comprehensive error handling and trade management advice."""
        default_plan = {
            'entry_signal': "HOLD/WATCH",
            'entry_strategy': "Wait for clearer signals",
            'stop_loss': 0,
            'targets': {'target_1': 0, 'target_2': 0, 'target_3': 0},
            'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days",
            'trade_management_note': 'N/A'  # Added default for consistency
        }

        try:
            current_price = data['Close'].iloc[-1]
            atr = risk_metrics.get('atr', current_price * 0.02)
            if pd.isna(atr) or atr <= 0: atr = current_price * 0.02

            # --- STRATEGY IMPROVEMENT BASED ON BACKTEST ---
            trade_management_note = "After hitting Target 1, consider moving Stop Loss to breakeven to let profits run."

            # Entry Strategy
            if score >= 75:
                entry_signal = "STRONG BUY"
            elif score >= 60:
                entry_signal = "BUY"
            elif score >= 45:
                entry_signal = "HOLD/WATCH"
            elif score >= 30:
                entry_signal = "SELL"
            else:
                entry_signal = "STRONG SELL"

            entry_strategy_map = {
                "STRONG BUY": "Enter aggressively on any dip",
                "BUY": "Enter on pullbacks or breakouts",
                "HOLD/WATCH": "Wait for clearer signals",
                "SELL": "Exit longs, consider shorts",
                "STRONG SELL": "Exit all positions"
            }
            entry_strategy = entry_strategy_map.get(entry_signal, "Wait for clearer signals")

            # Price Targets
            stop_loss_distance = atr * 2
            stop_loss = max(current_price - stop_loss_distance, 0)
            target_1 = current_price + (stop_loss_distance * 1.5)
            target_2 = current_price + (stop_loss_distance * 2.5)
            target_3 = current_price + (stop_loss_distance * 4.0)

            support, resistance = self.calculate_support_resistance(data)

            return {
                'entry_signal': entry_signal,
                'entry_strategy': entry_strategy,
                'stop_loss': stop_loss,
                'targets': {'target_1': target_1, 'target_2': target_2, 'target_3': target_3},
                'support': support or current_price * 0.95,
                'resistance': resistance or current_price * 1.05,
                'holding_period': f"{self.swing_trading_params['min_holding_period']}-{self.swing_trading_params['max_holding_period']} days",
                'trade_management_note': trade_management_note
            }
        except Exception as e:
            logger.error(f"Error generating trading plan: {str(e)}")
            return default_plan

    def analyze_swing_trading_stock(self, symbol, period="6mo"):
        """Comprehensive swing trading analysis for a single stock with full error handling"""
        try:
            if not symbol:
                logger.error("Empty symbol provided")
                return None

            logger.info(f"Starting analysis for {symbol}")

            # Get stock data
            data, info, final_symbol = self.get_indian_stock_data(symbol, period)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return None

            # Extract information
            stock_info = self.get_stock_info_from_db(symbol)
            sector = stock_info.get('sector', 'Unknown')
            company_name = stock_info.get('name', symbol)

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

            # Technical indicators
            rsi = self.calculate_rsi(data['Close'])
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['Close'])
            stoch_k, stoch_d = self.calculate_stochastic(data['High'], data['Low'], data['Close'])
            macd_line, signal_line, histogram = self.calculate_macd(data['Close'])
            support, resistance = self.calculate_support_resistance(data)
            volume_profile, poc_price = self.calculate_volume_profile(data)

            # Sentiment analysis
            sentiment_results = self.analyze_news_sentiment(final_symbol)

            # Risk metrics
            risk_metrics = self.calculate_risk_metrics(data)

            # Swing trading score
            swing_score = self.calculate_swing_trading_score(data, sentiment_results, sector)

            # Trading plan
            trading_plan = self.generate_trading_plan(data, swing_score, risk_metrics)

            # Safe value extraction
            rsi_val = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None
            bb_upper_val = bb_upper.iloc[-1] if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else None
            bb_middle_val = bb_middle.iloc[-1] if not bb_middle.empty and not pd.isna(bb_middle.iloc[-1]) else None
            bb_lower_val = bb_lower.iloc[-1] if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else None
            bb_position = ((current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)) if all(
                x is not None for x in [bb_upper_val, bb_lower_val]) and bb_upper_val != bb_lower_val else None
            stoch_k_val = stoch_k.iloc[-1] if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else None
            stoch_d_val = stoch_d.iloc[-1] if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else None
            macd_line_val = macd_line.iloc[-1] if not macd_line.empty and not pd.isna(macd_line.iloc[-1]) else None
            signal_line_val = signal_line.iloc[-1] if not signal_line.empty and not pd.isna(
                signal_line.iloc[-1]) else None
            histogram_val = histogram.iloc[-1] if not histogram.empty and not pd.isna(histogram.iloc[-1]) else None

            # Compile results
            result = {
                'symbol': final_symbol,
                'company_name': company_name,
                'sector': sector,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'rsi': rsi_val,
                'bollinger_bands': {
                    'upper': bb_upper_val, 'middle': bb_middle_val, 'lower': bb_lower_val, 'position': bb_position
                },
                'stochastic': {'k': stoch_k_val, 'd': stoch_d_val},
                'macd': {'line': macd_line_val, 'signal': signal_line_val, 'histogram': histogram_val},
                'support_resistance': {
                    'support': support, 'resistance': resistance,
                    'distance_to_support': ((current_price - support) / support * 100) if support else None,
                    'distance_to_resistance': (
                            (resistance - current_price) / current_price * 100) if resistance else None
                },
                'volume_profile': {
                    'poc_price': poc_price,
                    'current_vs_poc': ((current_price - poc_price) / poc_price * 100) if poc_price else None
                },
                'sentiment': {
                    'scores': sentiment_results[0], 'articles': sentiment_results[1],
                    'confidence': sentiment_results[2], 'method': sentiment_results[3],
                    'source': sentiment_results[4],
                    'sentiment_summary': {
                        'positive': sentiment_results[0].count('positive'),
                        'negative': sentiment_results[0].count('negative'),
                        'neutral': sentiment_results[0].count('neutral')
                    }
                },
                'risk_metrics': risk_metrics,
                'swing_score': swing_score,
                'trading_plan': trading_plan,
                'model_type': self.model_type,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info(f"Successfully analyzed {symbol} with score {swing_score}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def analyze_multiple_stocks(self, symbols, period="6mo"):
        """Analyze multiple stocks with progress tracking and comprehensive error handling"""
        results = []
        total_stocks = len(symbols) if symbols else 0

        if total_stocks == 0:
            logger.error("No symbols provided for analysis")
            return results

        print(f"Analyzing {total_stocks} stocks...")
        logger.info(f"Starting analysis of {total_stocks} stocks for swing trading.")

        successful_analyses = 0
        failed_analyses = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                # Progress printout
                progress_pct = (i / total_stocks) * 100
                print(f"\rAnalyzing: [{i}/{total_stocks}] {symbol}... ({progress_pct:.0f}%)", end="")

                if not symbol or not isinstance(symbol, str):
                    logger.warning(f"Invalid symbol at position {i}: {symbol}")
                    failed_analyses += 1
                    continue

                analysis = self.analyze_swing_trading_stock(symbol.strip(), period)
                if analysis and analysis.get('swing_score', 0) > 0:
                    results.append(analysis)
                    successful_analyses += 1
                else:
                    failed_analyses += 1
                    logger.warning(f"Analysis failed or returned zero score for {symbol}")

            except KeyboardInterrupt:
                logger.info("Analysis interrupted by user")
                print(f"\nAnalysis interrupted. Processed {i - 1}/{total_stocks} stocks.")
                break
            except Exception as e:
                logger.error(f"Unexpected error analyzing {symbol}: {str(e)}")
                failed_analyses += 1
                continue

        print("\nAnalysis complete.")
        # Sort by swing trading score
        try:
            results.sort(key=lambda x: x.get('swing_score', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error sorting results: {str(e)}")

        logger.info(f"Analysis completed: {successful_analyses} successful, {failed_analyses} failed")

        return results

    def filter_stocks_by_risk_appetite(self, results, risk_appetite):
        """Filter stocks based on user's risk appetite with error handling"""
        try:
            if not results:
                logger.warning("No results to filter")
                return []

            if not risk_appetite:
                logger.warning("No risk appetite specified, using MEDIUM")
                risk_appetite = "MEDIUM"

            risk_thresholds = {
                'LOW': 0.25,  # <=25% volatility
                'MEDIUM': 0.40,  # <=40% volatility
                'HIGH': 1.0  # <=100% volatility (all stocks)
            }

            max_volatility = risk_thresholds.get(risk_appetite.upper(), 0.40)

            filtered_stocks = []
            for stock in results:
                try:
                    if not isinstance(stock, dict):
                        continue

                    risk_metrics = stock.get('risk_metrics', {})
                    trading_plan = stock.get('trading_plan', {})

                    volatility = risk_metrics.get('volatility', 1.0)  # Default high volatility
                    entry_signal = trading_plan.get('entry_signal', 'HOLD/WATCH')

                    if (volatility <= max_volatility and
                            entry_signal in ['BUY', 'STRONG BUY']):
                        filtered_stocks.append(stock)

                except Exception as e:
                    logger.warning(f"Error filtering stock {stock.get('symbol', 'Unknown')}: {str(e)}")
                    continue

            logger.info(
                f"Filtered {len(filtered_stocks)} stocks from {len(results)} based on {risk_appetite} risk appetite")
            return filtered_stocks

        except Exception as e:
            logger.error(f"Error filtering stocks by risk appetite: {str(e)}")
            return []

    def generate_portfolio_allocation(self, results, total_capital, risk_appetite):
        """
        --- COMPLETELY REWRITTEN ---
        Generate portfolio allocation based on a fixed risk percentage per trade to control drawdown.
        This is the primary fix for the high drawdown seen in the backtest.
        """
        try:
            if not results or not isinstance(results, list):
                print("Error: No suitable stocks found for portfolio creation")
                return None

            if total_capital <= 0:
                print("Error: Invalid total capital amount")
                return None

            risk_per_trade_pct = self.swing_trading_params['risk_per_trade']
            capital_at_risk_per_trade = total_capital * risk_per_trade_pct

            print(f"\nPORTFOLIO ALLOCATION (Total Capital: Rs.{total_capital:,.2f})")
            print(
                f"Risk Model: Max {risk_per_trade_pct:.0%} of capital per trade (i.e., Rs.{capital_at_risk_per_trade:,.2f} per trade)")
            print("=" * 100)

            portfolio_data = []
            print(
                f"{'Rank':<4} {'Symbol':<12} {'Company':<25} {'Price':<10} {'Stop Loss':<10} {'Shares':<8} {'Amount':<15} {'Risk':<12}")
            print("-" * 100)

            total_allocated = 0

            for i, result in enumerate(results, 1):
                try:
                    current_price = result.get('current_price', 0)
                    trading_plan = result.get('trading_plan', {})
                    stop_loss = trading_plan.get('stop_loss', 0)

                    if current_price <= 0 or stop_loss <= 0 or current_price <= stop_loss:
                        continue

                    risk_per_share = current_price - stop_loss
                    num_shares = int(capital_at_risk_per_trade / risk_per_share)

                    if num_shares == 0:
                        continue

                    amount_to_invest = num_shares * current_price
                    if total_allocated + amount_to_invest > total_capital:
                        continue

                    total_allocated += amount_to_invest
                    company_name = result.get('company_name', result.get('symbol', 'Unknown'))
                    company_short = company_name[:23] + "..." if len(company_name) > 25 else company_name

                    portfolio_data.append({
                        'symbol': result.get('symbol', 'Unknown'),
                        'company': company_name,
                        'score': result.get('swing_score', 0),
                        'amount': amount_to_invest,
                        'sector': result.get('sector', 'Unknown')
                    })

                    print(
                        f"{i:<4} {result.get('symbol', 'Unk'):<12} {company_short:<25} "
                        f"₹{current_price:<9.2f} ₹{stop_loss:<9.2f} {num_shares:<8} "
                        f"₹{amount_to_invest:<14,.2f} ₹{capital_at_risk_per_trade:<11,.2f}"
                    )
                except Exception as e:
                    logger.error(f"Error processing stock {i} in portfolio allocation: {str(e)}")

            if not portfolio_data:
                print(
                    f"\n{Fore.RED}Could not allocate any positions based on the current risk model and budget.{Style.RESET_ALL}")
                return None

            avg_score = sum(r['score'] for r in portfolio_data) / len(portfolio_data)

            sector_allocation = {}
            for stock in portfolio_data:
                sector = stock.get('sector', 'Unknown')
                sector_allocation[sector] = sector_allocation.get(sector, 0) + stock['amount']

            for sector in sector_allocation:
                sector_allocation[sector] = (sector_allocation[sector] / total_allocated) * 100

            print(f"\nPORTFOLIO SUMMARY")
            print("-" * 50)
            print(f"Total Budget: Rs.{total_capital:,.2f}")
            print(f"Total Allocated: Rs.{total_allocated:,.2f} ({total_allocated / total_capital * 100:.1f}%)")
            print(f"Number of Stocks: {len(portfolio_data)}")
            print(f"Average Score: {avg_score:.1f}/100")
            print(f"Portfolio Risk Level: {risk_appetite}")

            if sector_allocation:
                print(f"\nSECTOR DIVERSIFICATION")
                print("-" * 30)
                for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
                    print(f"{sector}: {allocation:.1f}%")

            return portfolio_data
        except Exception as e:
            logger.error(f"Error generating portfolio allocation: {str(e)}")
            return None

    def get_single_best_recommendation(self, results):
        """Get detailed recommendation for the single best stock with enhanced formatting and error handling."""
        try:
            if not results or not isinstance(results, list):
                logger.warning("No results available for recommendation")
                return None

            best_stock = results[0]
            if not isinstance(best_stock, dict):
                logger.error("Invalid best stock data format")
                return None

            print(f"\n{Fore.YELLOW}⭐ SINGLE BEST STOCK RECOMMENDATION ⭐{Style.RESET_ALL}")
            print("=" * 70)

            # --- Safely extract all data using .get() to prevent errors ---
            company_name = best_stock.get('company_name', 'Unknown')
            symbol = best_stock.get('symbol', 'Unknown')
            sector = best_stock.get('sector', 'Unknown')
            swing_score = best_stock.get('swing_score', 0)
            current_price = best_stock.get('current_price', 0)
            price_change = best_stock.get('price_change', 0)
            price_change_pct = best_stock.get('price_change_pct', 0)
            risk_metrics = best_stock.get('risk_metrics', {})
            risk_level = risk_metrics.get('risk_level', 'Unknown')

            # --- Enhanced Display ---
            price_color = Fore.GREEN if price_change >= 0 else Fore.RED
            score_color = Fore.GREEN if swing_score >= 75 else Fore.YELLOW if swing_score >= 60 else Fore.RED

            print(f"{'Company:':<15} {company_name} ({symbol})")
            print(f"{'Sector:':<15} {sector}")
            print(
                f"{'Price:':<15} {price_color}₹{current_price:,.2f} ({price_change:+.2f}, {price_change_pct:+.2f}%){Style.RESET_ALL}")
            print(f"{'Swing Score:':<15} {score_color}{swing_score:.0f}/100{Style.RESET_ALL}")
            print(f"{'Risk Level:':<15} {risk_level}")

            # --- Trading Recommendation with more color ---
            trading_plan = best_stock.get('trading_plan', {})
            signal = trading_plan.get('entry_signal', 'N/A')
            signal_color = Fore.GREEN if "BUY" in signal else (Fore.RED if "SELL" in signal else Fore.YELLOW)

            print(f"\n{Fore.YELLOW}ACTIONABLE TRADING PLAN{Style.RESET_ALL}")
            print("-" * 30)
            print(f"{'Signal:':<15} {signal_color}{signal}{Style.RESET_ALL}")
            print(f"{'Strategy:':<15} {trading_plan.get('entry_strategy', 'N/A')}")

            targets = trading_plan.get('targets', {})
            print(f"{'Stop Loss:':<15} {Fore.RED}₹{trading_plan.get('stop_loss', 0):.2f}{Style.RESET_ALL}")
            print(f"{'Target 1:':<15} {Fore.GREEN}₹{targets.get('target_1', 0):.2f}{Style.RESET_ALL}")
            print(f"{'Target 2:':<15} {Fore.GREEN}₹{targets.get('target_2', 0):.2f}{Style.RESET_ALL}")

            # Display the trade management advice
            if trading_plan.get('trade_management_note'):
                print(f"{'Pro Tip:':<15} {Fore.CYAN}{trading_plan.get('trade_management_note')}{Style.RESET_ALL}")

            # Key technical levels
            print(f"\n{Fore.YELLOW}KEY LEVELS & DATA{Style.RESET_ALL}")
            print("-" * 30)
            print(f"{'Support:':<15} ₹{trading_plan.get('support', 0):.2f}")
            print(f"{'Resistance:':<15} ₹{trading_plan.get('resistance', 0):.2f}")

            rsi_val = best_stock.get('rsi')
            if rsi_val is not None:
                print(f"{'RSI (14-day):':<15} {rsi_val:.1f}")

            # Sentiment summary
            sentiment = best_stock.get('sentiment', {}).get('sentiment_summary', {})
            print(
                f"{'News Sentiment:':<15} Pos: {sentiment.get('positive', 0)}, Neg: {sentiment.get('negative', 0)}, Neu: {sentiment.get('neutral', 0)}")

            return best_stock

        except Exception as e:
            logger.error(f"Error getting single best recommendation: {str(e)}")
            return None

    def print_analysis_summary(self, all_results, filtered_results, risk_appetite, total_budget):
        """Print comprehensive analysis summary with error handling"""
        try:
            print(f"\nMARKET ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Total Stocks Analyzed: {len(all_results) if all_results else 0}")
            print(f"Risk Appetite: {risk_appetite}")
            print(f"Budget: Rs.{total_budget:,}")
            print(f"Suitable Stocks Found: {len(filtered_results) if filtered_results else 0}")

            if all_results and len(all_results) > 0:
                try:
                    avg_market_score = sum(r.get('swing_score', 0) for r in all_results) / len(all_results)
                    print(f"Average Market Score: {avg_market_score:.1f}/100")
                except:
                    print("Average Market Score: Unable to calculate")

                # Risk distribution
                risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'UNKNOWN': 0}
                for result in all_results:
                    try:
                        risk_level = result.get('risk_metrics', {}).get('risk_level', 'UNKNOWN')
                        if risk_level in risk_distribution:
                            risk_distribution[risk_level] += 1
                        else:
                            risk_distribution['UNKNOWN'] += 1
                    except:
                        risk_distribution['UNKNOWN'] += 1

                print(f"\nMARKET RISK DISTRIBUTION")
                print("-" * 25)
                for risk, count in risk_distribution.items():
                    if count > 0:
                        percentage = (count / len(all_results)) * 100
                        print(f"{risk} Risk: {count} stocks ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Error printing analysis summary: {str(e)}")
            print(f"Error generating analysis summary: {str(e)}")


# ========================= MAIN EXECUTION =========================

def safe_input_int(prompt, default=None, min_val=None, max_val=None):
    """Safe integer input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = int(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


def safe_input_float(prompt, default=None, min_val=None, max_val=None):
    """Safe float input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = float(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


# Unified Trading Platform
class UnifiedTradingPlatform:
    """Unified platform combining position and swing trading systems"""

    def __init__(self):
        # Initialize both trading systems
        self.position_system = EnhancedPositionTradingSystem()
        self.swing_system = EnhancedSwingTradingSystem()

        # User preferences and portfolio
        self.user_preferences = {}
        self.portfolio = {}

        logger.info("UnifiedTradingPlatform initialized successfully")

    def set_user_preferences(self, risk_appetite, investment_horizon, capital):
        """Set user trading preferences"""
        self.user_preferences = {
            'risk_appetite': risk_appetite,  # Low, Medium, High
            'investment_horizon': investment_horizon,  # Short, Medium, Long
            'capital': capital
        }

    def recommend_strategy(self, symbol):
        """Recommend whether to use position or swing trading for a symbol"""
        # Analyze the stock with both systems
        position_analysis = self.position_system.analyze_position_trading_stock(symbol)
        swing_analysis = self.swing_system.analyze_swing_trading_stock(symbol)

        # Get user preferences
        horizon = self.user_preferences.get('investment_horizon', 'Medium')
        risk = self.user_preferences.get('risk_appetite', 'Medium')

        # Determine recommended strategy
        if horizon == 'Long' or (horizon == 'Medium' and risk == 'Low'):
            return 'position', position_analysis
        else:
            return 'swing', swing_analysis

    def create_unified_portfolio(self):
        """Create a portfolio combining both position and swing trading strategies"""
        # Get all available symbols
        all_symbols = list(set(list(self.position_system.indian_stocks.keys()) +
                               list(self.swing_system.swing_stocks.keys())))

        portfolio = {
            'position_trades': [],
            'swing_trades': [],
            'allocations': {}
        }

        capital = self.user_preferences.get('capital', 100000)

        # Analyze each symbol and assign to appropriate strategy
        for symbol in all_symbols:
            strategy, analysis = self.recommend_strategy(symbol)

            if strategy == 'position' and analysis['position_score'] >= 65:
                # Calculate position size for position trading
                position_size = capital * 0.08  # Max 8% per position trade
                portfolio['position_trades'].append({
                    'symbol': symbol,
                    'strategy': 'position',
                    'score': analysis['position_score'],
                    'size': position_size,
                    'analysis': analysis
                })
            elif strategy == 'swing' and analysis['swing_score'] >= 70:
                # Calculate position size for swing trading
                position_size = capital * 0.05  # Max 5% per swing trade
                portfolio['swing_trades'].append({
                    'symbol': symbol,
                    'strategy': 'swing',
                    'score': analysis['swing_score'],
                    'size': position_size,
                    'analysis': analysis
                })

        # Calculate allocations
        total_position = sum(trade['size'] for trade in portfolio['position_trades'])
        total_swing = sum(trade['size'] for trade in portfolio['swing_trades'])

        portfolio['allocations'] = {
            'position_percentage': total_position / capital * 100,
            'swing_percentage': total_swing / capital * 100,
            'cash_percentage': (capital - total_position - total_swing) / capital * 100
        }

        return portfolio

    def monitor_portfolio(self, portfolio):
        """Monitor and rebalance portfolio based on market conditions"""
        # Implementation for portfolio monitoring and rebalancing
        pass

    def generate_performance_report(self, portfolio):
        """Generate performance report for the portfolio"""
        # Implementation for performance reporting
        pass


import sys
import os
import time
from datetime import datetime
import traceback
from colorama import Fore, Style, init


def safe_input_int(prompt, default=None, min_val=None, max_val=None):
    """Safe integer input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = int(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


def safe_input_float(prompt, default=None, min_val=None, max_val=None):
    """Safe float input with validation"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default

            value = float(user_input)

            if min_val is not None and value < min_val:
                print(f"Error: Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"Error: Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None


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


def analyze_stock_interactive(trading_system, symbol):
    """Interactive analysis of a single stock"""
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Analyzing {symbol}...{Style.RESET_ALL}")

    # Show loading animation
    try:
        from halo import Halo
        with Halo(text='Analyzing stock data', spinner='dots') as spinner:
            result = trading_system.analyze_position_trading_stock(symbol)
            spinner.succeed("Analysis complete!")
    except ImportError:
        print("Analyzing stock data...")
        result = trading_system.analyze_position_trading_stock(symbol)
        print("Analysis complete!")

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

        try:
            from halo import Halo
            with Halo(text=f'Analyzing {symbol}', spinner='dots') as spinner:
                result = trading_system.analyze_position_trading_stock(symbol)
                if result:
                    results.append(result)
                    spinner.succeed(f"✓ {symbol} analyzed")
                else:
                    spinner.fail(f"✗ {symbol} failed")
        except ImportError:
            result = trading_system.analyze_position_trading_stock(symbol)
            if result:
                results.append(result)
                print(f"✓ {symbol} analyzed")
            else:
                print(f"✗ {symbol} failed")

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


def run_swing_trading_mode():
    """Run the swing trading system"""
    try:
        # Initialize the enhanced swing trading system
        print("Initializing Enhanced Swing Trading System...")
        swing_trader = EnhancedSwingTradingSystem(
            model_path="D:/Python_files/models/sentiment_pipeline.joblib",
            news_api_key=os.getenv("NEWS_API_KEY")
        )

        print("ENHANCED SWING TRADING SYSTEM")
        print("Advanced Portfolio Creation with Budget & Risk Management")
        print("=" * 70)

        # ===== USER INPUT COLLECTION =====

        # Get user budget
        total_budget = safe_input_float(
            "\nEnter your total investment budget in INR (e.g., 500000): ",
            min_val=1000
        )

        if total_budget is None:
            print("Exiting...")
            return

        # Get user risk appetite
        risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        print(f"\nRisk Appetite Options:")
        print("• LOW: Conservative (<=25% volatility) - Blue chip stocks")
        print("• MEDIUM: Balanced (<=40% volatility) - Mixed portfolio")
        print("• HIGH: Aggressive (<=100% volatility) - All opportunities")

        while True:
            try:
                risk_appetite = input("\nEnter your risk appetite (LOW/MEDIUM/HIGH): ").upper().strip()
                if risk_appetite not in risk_levels:
                    print("Error: Invalid risk level. Please enter LOW, MEDIUM, or HIGH.")
                else:
                    break
            except KeyboardInterrupt:
                print("\nExiting...")
                return

        print(f"\nConfiguration Set:")
        print(f"Budget: Rs.{total_budget:,.0f}")
        print(f"Risk Appetite: {risk_appetite}")

        # ===== COMPREHENSIVE MARKET ANALYSIS =====

        print(f"\nAnalyzing Indian Stock Market...")
        print(f"Scanning {len(swing_trader.get_all_stock_symbols())} stocks across BSE & NSE")
        print("This may take several minutes...")

        # Get all stock symbols from database
        all_symbols = swing_trader.get_all_stock_symbols()

        # Analyze stocks
        start_time = datetime.now()
        all_results = swing_trader.analyze_multiple_stocks(all_symbols)
        analysis_time = datetime.now() - start_time

        print(f"Analysis completed in {analysis_time.total_seconds():.0f} seconds")

        # ===== RISK-BASED FILTERING =====

        print(f"\nFiltering stocks by risk appetite...")
        filtered_results = swing_trader.filter_stocks_by_risk_appetite(all_results, risk_appetite)

        if not filtered_results:
            print(f"\nNo suitable stocks found matching your criteria:")
            print(f"• Risk Appetite: {risk_appetite}")
            print(f"• Minimum Signal: BUY or STRONG BUY")
            print("\nSuggestions:")
            print("• Consider increasing your risk tolerance")
            print("• Try a different time period")
            print("• Check market conditions")
        else:
            # ===== PORTFOLIO CREATION =====

            print(f"\nFound {len(filtered_results)} suitable investment opportunities")

            # Generate portfolio allocation
            portfolio = swing_trader.generate_portfolio_allocation(
                filtered_results,
                int(total_budget),
                risk_appetite
            )

            # ===== SINGLE BEST RECOMMENDATION =====

            best_stock = swing_trader.get_single_best_recommendation(filtered_results)

            # ===== COMPREHENSIVE SUMMARY =====

            swing_trader.print_analysis_summary(all_results, filtered_results, risk_appetite, total_budget)

            # ===== DETAILED RANKINGS =====

            print(f"\nTOP 10 STOCK RANKINGS")
            print("=" * 70)
            print(f"{'Rank':<4} {'Symbol':<12} {'Company':<20} {'Score':<6} {'Signal':<12} {'Risk':<8}")
            print("-" * 70)

            for i, result in enumerate(filtered_results[:10], 1):
                try:
                    company_name = result.get('company_name', 'Unknown')
                    company_short = company_name[:18] + "..." if len(company_name) > 20 else company_name
                    symbol = result.get('symbol', 'Unknown')
                    score = result.get('swing_score', 0)
                    signal = result.get('trading_plan', {}).get('entry_signal', 'Unknown')
                    risk = result.get('risk_metrics', {}).get('risk_level', 'Unknown')

                    print(f"{i:<4} {symbol:<12} {company_short:<20} {score:<6.0f} {signal:<12} {risk:<8}")
                except Exception as e:
                    logger.error(f"Error displaying ranking for position {i}: {str(e)}")

        # ===== INTERACTIVE MODE =====

        print(f"\nINTERACTIVE MODE")
        print("Available commands:")
        print("• Enter stock symbol for detailed analysis")
        print("• Type 'portfolio' for custom portfolio analysis")
        print("• Type 'settings' to change budget/risk settings")
        print("• Type 'quit' to exit")

        while True:
            try:
                user_input = input("\n> Enter command: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thank you for using Enhanced Swing Trading System!")
                    print("Happy Trading!")
                    break

                elif user_input.lower() == 'portfolio':
                    symbols = input("Enter stock symbols (comma-separated) or 'sample' for sample analysis: ").strip()

                    if symbols.lower() == 'sample':
                        symbols_list = swing_trader.get_all_stock_symbols()[:20]  # Sample 20 stocks
                        print(f"Analyzing sample portfolio of {len(symbols_list)} stocks...")
                    else:
                        symbols_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

                    if symbols_list:
                        interactive_results = swing_trader.analyze_multiple_stocks(symbols_list)
                        if interactive_results:
                            filtered_interactive = swing_trader.filter_stocks_by_risk_appetite(
                                interactive_results, risk_appetite)
                            if filtered_interactive:
                                swing_trader.generate_portfolio_allocation(
                                    filtered_interactive, int(total_budget), risk_appetite)
                            else:
                                print("No suitable stocks found in your selection")
                        else:
                            print("No valid analysis results obtained")
                    else:
                        print("No valid symbols provided")

                elif user_input.lower() == 'settings':
                    print("\nCurrent settings:")
                    print(f"Budget: Rs.{total_budget:,.0f}")
                    print(f"Risk Appetite: {risk_appetite}")

                    change = input("Change settings? (y/n): ").lower()
                    if change == 'y':
                        new_budget = safe_input_float(
                            f"Enter new budget (current: Rs.{total_budget:,.0f}): ",
                            default=total_budget, min_val=1000
                        )
                        if new_budget:
                            total_budget = new_budget

                        print("Risk levels: LOW, MEDIUM, HIGH")
                        new_risk = input(f"Enter new risk appetite (current: {risk_appetite}): ").upper().strip()
                        if new_risk in risk_levels:
                            risk_appetite = new_risk

                        print(f"Settings updated - Budget: Rs.{total_budget:,.0f}, Risk: {risk_appetite}")

                elif user_input.upper() in swing_trader.get_all_stock_symbols():
                    # User entered a valid stock symbol
                    symbol = user_input.upper()
                    print(f"\nAnalyzing {symbol}...")

                    # Analyze the single stock
                    analysis = swing_trader.analyze_swing_trading_stock(symbol)

                    if analysis:
                        # Print detailed analysis
                        print(f"\nDETAILED ANALYSIS: {analysis['symbol']} ({analysis['company_name']})")
                        print("=" * 70)
                        print(f"Sector: {analysis['sector']}")
                        print(f"Analysis Date: {analysis['analysis_date']}")
                        print(f"Current Price: Rs.{analysis['current_price']:.2f}")
                        print(f"Price Change: Rs.{analysis['price_change']:.2f} ({analysis['price_change_pct']:.2f}%)")
                        print(f"Swing Score: {analysis['swing_score']:.0f}/100")
                        print(f"Risk Level: {analysis['risk_metrics']['risk_level']}")

                        # Technical Indicators
                        print("\nTECHNICAL INDICATORS")
                        print("-" * 30)
                        if analysis['rsi']:
                            print(f"• RSI: {analysis['rsi']:.1f} (14-day)")

                        if analysis['bollinger_bands']['position']:
                            bb_pos = analysis['bollinger_bands']['position'] * 100
                            print(f"• Bollinger Bands Position: {bb_pos:.1f}% (0% = lower band, 100% = upper band)")

                        if analysis['stochastic']['k'] and analysis['stochastic']['d']:
                            print(
                                f"• Stochastic: K={analysis['stochastic']['k']:.1f}, D={analysis['stochastic']['d']:.1f}")

                        if analysis['macd']['line']:
                            print(
                                f"• MACD: Line={analysis['macd']['line']:.2f}, Signal={analysis['macd']['signal']:.2f}, Histogram={analysis['macd']['histogram']:.2f}")

                        if analysis['support_resistance']['support']:
                            print(f"• Support: Rs.{analysis['support_resistance']['support']:.2f}")

                        if analysis['support_resistance']['resistance']:
                            print(f"• Resistance: Rs.{analysis['support_resistance']['resistance']:.2f}")

                        # Sentiment Summary
                        sentiment = analysis['sentiment']['sentiment_summary']
                        print("\nSENTIMENT OVERVIEW")
                        print("-" * 20)
                        print(
                            f"• Positive: {sentiment['positive']}, Negative: {sentiment['negative']}, Neutral: {sentiment['neutral']}")
                        print(
                            f"• Method: {analysis['sentiment']['method']} | Source: {analysis['sentiment']['source']}")

                        # Sample News Headlines
                        if analysis['sentiment']['articles']:
                            print("\nSAMPLE NEWS HEADLINES")
                            print("-" * 25)
                            for i, article in enumerate(analysis['sentiment']['articles'][:3], 1):
                                print(f"{i}. {article}")

                        # Trading Plan
                        tp = analysis['trading_plan']
                        print("\nTRADING PLAN")
                        print("-" * 20)
                        print(f"• Signal: {tp['entry_signal']}")
                        print(f"• Strategy: {tp['entry_strategy']}")
                        print(f"• Stop Loss: Rs.{tp['stop_loss']:.2f}")
                        print(
                            f"• Targets: Rs.{tp['targets']['target_1']:.2f} | Rs.{tp['targets']['target_2']:.2f} | Rs.{tp['targets']['target_3']:.2f}")
                        print(f"• Holding Period: {tp['holding_period']}")

                        # Risk Metrics
                        rm = analysis['risk_metrics']
                        print("\nRISK METRICS")
                        print("-" * 15)
                        print(f"• Volatility: {rm['volatility'] * 100:.1f}% (annualized)")
                        print(f"• Max Drawdown: {rm['max_drawdown'] * 100:.1f}%")
                        print(f"• Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
                        print(f"• Value at Risk (95%): {rm['var_95'] * 100:.1f}%")

                    else:
                        print(f"Could not analyze {symbol}. Please try another symbol.")

                else:
                    # Try to analyze as custom symbol
                    if len(user_input) > 0:
                        print(f"Attempting to analyze {user_input.upper()}...")
                        analysis = swing_trader.analyze_swing_trading_stock(user_input.upper())

                        if analysis:
                            print(f"Analysis completed for {user_input.upper()}")
                            print(f"Score: {analysis['swing_score']:.0f}/100")
                            print(f"Signal: {analysis['trading_plan']['entry_signal']}")
                            print(f"Risk: {analysis['risk_metrics']['risk_level']}")
                        else:
                            print(f"Could not analyze {user_input.upper()}. Symbol may not be available.")
                    else:
                        print("Invalid command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\nThank you for using Enhanced Swing Trading System!")
                print("Happy Trading!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"Error: {str(e)}")
                print("Please try again or type 'quit' to exit.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
        return
    except Exception as e:
        logger.error(f"Critical error in swing trading execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Critical error occurred: {str(e)}")
        print("Please check the logs for more details.")
        return


def run_position_trading_mode():
    """Run the position trading system"""
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
        print(f"{Fore.RED}Error in position trading execution: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for the unified trading platform"""
    init(autoreset=True)  # Initialize colorama

    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}UNIFIED TRADING PLATFORM{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")

    # Main menu
    while True:
        print(f"\n{Fore.YELLOW}Main Menu:{Style.RESET_ALL}")
        print("1. Swing Trading System")
        print("2. Position Trading System")
        print("3. Unified Platform (Both Systems)")
        print("4. Exit")

        choice = input(f"{Fore.GREEN}Enter your choice (1-4): {Style.RESET_ALL}").strip()

        if choice == "1":
            run_swing_trading_mode()
        elif choice == "2":
            run_position_trading_mode()
        elif choice == "3":
            # Initialize the unified platform
            platform = UnifiedTradingPlatform()

            # Get user preferences
            print(f"\n{Fore.GREEN}Please set your trading preferences:{Style.RESET_ALL}")

            risk_appetite = input("Risk Appetite (Low/Medium/High): ").strip().capitalize()
            investment_horizon = input("Investment Horizon (Short/Medium/Long): ").strip().capitalize()

            try:
                capital = float(input("Investment Capital (INR): ").strip())
            except ValueError:
                print(f"{Fore.RED}Invalid capital amount. Using default 100,000 INR.{Style.RESET_ALL}")
                capital = 100000

            platform.set_user_preferences(risk_appetite, investment_horizon, capital)

            # Unified platform functionality would go here
            print(f"{Fore.GREEN}Unified platform functionality is under development.{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Please use the individual trading systems for now.{Style.RESET_ALL}")

        elif choice == "4":
            print(f"{Fore.CYAN}Thank you for using the Unified Trading Platform!{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()