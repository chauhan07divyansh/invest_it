# enhanced_main.py - Improved version with better transparency
import sys
import logging
from colorama import Fore, Style, Back, init
import pandas as pd
import time
from datetime import datetime, timedelta
import json

# Import your system classes and config
from systems.position_trading import EnhancedPositionTradingSystem
from systems.swing_trading import EnhancedSwingTradingSystem
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractiveTradingPlatform:
    def __init__(self):
        self.user_profile = {}
        self.swing_system = None
        self.position_system = None
        self.initialize_systems()

    def initialize_systems(self):
        """Initialize trading systems with loading animation."""
        print(f"\n{Fore.CYAN}🚀 Initializing Advanced Trading Platform...{Style.RESET_ALL}")

        loading_messages = [
            "Loading market data...",
            "Initializing AI models...",
            "Connecting to news feeds...",
            "Setting up technical indicators...",
            "Calibrating sentiment analysis..."
        ]

        for msg in loading_messages:
            print(f"{Fore.YELLOW}⚡ {msg}{Style.RESET_ALL}")
            time.sleep(0.5)

        try:
            self.swing_system = EnhancedSwingTradingSystem()
            self.position_system = EnhancedPositionTradingSystem()
            print(f"{Fore.GREEN}✅ Systems initialized successfully!{Style.RESET_ALL}")
        except Exception as e:
            logger.critical(f"Could not initialize trading systems. Error: {e}")
            sys.exit(1)

    def create_user_profile(self):
        """Create a personalized user profile for better recommendations."""
        print(f"\n{Back.BLUE}{Fore.WHITE} PERSONALIZATION SETUP {Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}Let's create your personalized trading profile for better recommendations!{Style.RESET_ALL}\n")

        # Basic Information
        self.user_profile['name'] = input("👤 What's your name? ").strip()

        # Experience Level
        print(f"\n{Fore.YELLOW}📊 Trading Experience Level:{Style.RESET_ALL}")
        print("1. 🔰 Beginner (0-1 years)")
        print("2. 📈 Intermediate (1-3 years)")
        print("3. 🎯 Advanced (3+ years)")
        experience = self.get_valid_choice([1, 2, 3], "Select your experience level: ")
        experience_map = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}
        self.user_profile['experience'] = experience_map[experience]

        # Investment Goals
        print(f"\n{Fore.YELLOW}🎯 Primary Investment Goals:{Style.RESET_ALL}")
        print("1. 💰 Wealth Creation")
        print("2. 📈 Regular Income")
        print("3. 🛡️ Capital Preservation")
        print("4. 🚀 Aggressive Growth")
        goal = self.get_valid_choice([1, 2, 3, 4], "Select your primary goal: ")
        goal_map = {1: 'Wealth Creation', 2: 'Regular Income', 3: 'Capital Preservation', 4: 'Aggressive Growth'}
        self.user_profile['goal'] = goal_map[goal]

        # Risk Tolerance
        print(f"\n{Fore.YELLOW}⚖️ Risk Tolerance Assessment:{Style.RESET_ALL}")
        print("1. 🟢 Conservative - Low risk, steady returns")
        print("2. 🟡 Moderate - Balanced risk and return")
        print("3. 🔴 Aggressive - High risk, high return potential")
        risk = self.get_valid_choice([1, 2, 3], "Select your risk tolerance: ")
        risk_map = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
        self.user_profile['risk_tolerance'] = risk_map[risk]

        # Age Group
        print(f"\n{Fore.YELLOW}👥 Age Group:{Style.RESET_ALL}")
        print("1. 18-30 (High growth focus)")
        print("2. 31-45 (Balanced approach)")
        print("3. 46-60 (Conservative growth)")
        print("4. 60+ (Income focused)")
        age = self.get_valid_choice([1, 2, 3, 4], "Select your age group: ")
        age_map = {1: '18-30', 2: '31-45', 3: '46-60', 4: '60+'}
        self.user_profile['age_group'] = age_map[age]

        print(f"\n{Fore.GREEN}✅ Profile created successfully! Welcome, {self.user_profile['name']}!{Style.RESET_ALL}")
        self.display_user_profile()

    def display_user_profile(self):
        """Display the user's profile in a formatted way."""
        print(f"\n{Back.CYAN}{Fore.BLACK} YOUR TRADING PROFILE {Style.RESET_ALL}")
        print(f"👤 Name: {Fore.CYAN}{self.user_profile['name']}{Style.RESET_ALL}")
        print(f"📊 Experience: {Fore.YELLOW}{self.user_profile['experience']}{Style.RESET_ALL}")
        print(f"🎯 Goal: {Fore.GREEN}{self.user_profile['goal']}{Style.RESET_ALL}")
        print(
            f"⚖️ Risk Tolerance: {Fore.RED if self.user_profile['risk_tolerance'] == 'HIGH' else Fore.YELLOW if self.user_profile['risk_tolerance'] == 'MEDIUM' else Fore.GREEN}{self.user_profile['risk_tolerance']}{Style.RESET_ALL}")
        print(f"👥 Age Group: {Fore.MAGENTA}{self.user_profile['age_group']}{Style.RESET_ALL}")

    def get_valid_choice(self, valid_choices, prompt):
        """Get a valid choice from user input."""
        while True:
            try:
                choice = int(input(f"{Fore.GREEN}{prompt}{Style.RESET_ALL}"))
                if choice in valid_choices:
                    return choice
                else:
                    print(f"{Fore.RED}❌ Invalid choice. Please select from {valid_choices}{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}❌ Please enter a valid number.{Style.RESET_ALL}")

    def calculate_target_price(self, result, system_type):
        """Calculate a more realistic target price based on analysis."""
        current_price = result.get('current_price', 0)
        if current_price <= 0:
            return 0.0

        # Get the AI score
        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        ai_score = result.get(score_key, 0)

        # Calculate target based on score and system type
        if system_type == 'Swing':
            # Swing trading: shorter timeframe, lower targets
            if ai_score >= 80:
                multiplier = 1.08  # 8% target
            elif ai_score >= 70:
                multiplier = 1.06  # 6% target
            elif ai_score >= 60:
                multiplier = 1.04  # 4% target
            else:
                multiplier = 1.02  # 2% target
        else:
            # Position trading: longer timeframe, higher targets
            if ai_score >= 80:
                multiplier = 1.25  # 25% target
            elif ai_score >= 70:
                multiplier = 1.18  # 18% target
            elif ai_score >= 60:
                multiplier = 1.12  # 12% target
            else:
                multiplier = 1.06  # 6% target

        # Consider technical indicators for fine-tuning
        rsi = result.get('rsi', 50)
        if rsi < 30:  # Oversold - higher potential
            multiplier += 0.02
        elif rsi > 70:  # Overbought - lower potential
            multiplier -= 0.02

        # Consider fundamental factors for position trading
        if system_type == 'Position':
            fundamentals = result.get('fundamentals', {})
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio and pe_ratio < 15:  # Undervalued
                multiplier += 0.03
            elif pe_ratio and pe_ratio > 25:  # Overvalued
                multiplier -= 0.03

        return current_price * multiplier

    def get_detailed_analysis_reasoning(self, result, system_type):
        """Generate detailed reasoning for the analysis."""
        reasoning = []

        # Score analysis
        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        score = result.get(score_key, 0)

        if score >= 70:
            reasoning.append(f"✅ High AI confidence score ({score:.1f}/100) indicates strong investment potential")
        elif score >= 55:
            reasoning.append(f"⚠️ Moderate AI confidence score ({score:.1f}/100) suggests cautious approach")
        else:
            reasoning.append(f"❌ Low AI confidence score ({score:.1f}/100) indicates high risk")

        # Technical analysis reasoning
        rsi = result.get('rsi')
        if rsi:
            if rsi < 30:
                reasoning.append(f"📈 RSI ({rsi:.1f}) indicates oversold condition - potential buying opportunity")
            elif rsi > 70:
                reasoning.append(f"📉 RSI ({rsi:.1f}) indicates overbought condition - exercise caution")
            else:
                reasoning.append(f"➡️ RSI ({rsi:.1f}) in neutral zone - no strong directional bias")

        # MACD analysis
        macd = result.get('macd', {})
        if macd:
            macd_line = macd.get('line', 0)
            macd_signal = macd.get('signal', 0)
            if macd_line > macd_signal:
                reasoning.append("📈 MACD shows bullish momentum")
            else:
                reasoning.append("📉 MACD shows bearish momentum")

        # Fundamental analysis for position trading
        if system_type == 'Position':
            fundamentals = result.get('fundamentals', {})
            pe_ratio = fundamentals.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:
                    reasoning.append(f"💰 P/E ratio ({pe_ratio:.1f}) suggests undervalued stock")
                elif pe_ratio > 25:
                    reasoning.append(f"💸 P/E ratio ({pe_ratio:.1f}) suggests overvalued stock")
                else:
                    reasoning.append(f"⚖️ P/E ratio ({pe_ratio:.1f}) indicates fair valuation")

        # Sentiment analysis
        sentiment_data = result.get('sentiment', {})
        if sentiment_data:
            sentiment_summary = sentiment_data.get('sentiment_summary', {})
            positive = sentiment_summary.get('positive', 0)
            negative = sentiment_summary.get('negative', 0)

            if positive > negative * 2:
                reasoning.append(f"📰 Strong positive news sentiment ({positive} positive vs {negative} negative)")
            elif negative > positive * 2:
                reasoning.append(f"📰 Negative news sentiment concern ({negative} negative vs {positive} positive)")
            else:
                reasoning.append(f"📰 Balanced news sentiment ({positive} positive, {negative} negative)")

        return reasoning

    def display_transparent_analysis(self, result, system_type):
        """Enhanced transparent analysis display with detailed explanations."""
        if not result:
            print(f"{Fore.RED}❌ Analysis failed or no data returned.{Style.RESET_ALL}")
            return

        symbol = result.get('symbol', 'N/A')
        company_name = result.get('company_name', 'N/A')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate proper target price
        target_price = self.calculate_target_price(result, system_type)

        print(f"\n{Back.BLUE}{Fore.WHITE} COMPREHENSIVE AI ANALYSIS - {current_time} {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📈 STOCK: {symbol} | {company_name} | {system_type} Strategy{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

        # Overall Score and Recommendation
        score_key = 'swing_score' if 'swing_score' in result else 'position_score'
        score = result.get(score_key, 0)
        score_color = Fore.GREEN if score >= 70 else (Fore.YELLOW if score >= 55 else Fore.RED)
        score_emoji = "🟢" if score >= 70 else ("🟡" if score >= 55 else "🔴")

        print(
            f"\n{Back.GREEN if score >= 70 else Back.YELLOW if score >= 55 else Back.RED}{Fore.BLACK} OVERALL AI SCORE: {score:.1f}/100 {score_emoji} {Style.RESET_ALL}")

        # Trading Plan with calculated target
        trading_plan = result.get('trading_plan', {})
        signal = trading_plan.get('entry_signal', 'HOLD')
        signal_color = Fore.GREEN if "BUY" in signal else (Fore.RED if "SELL" in signal else Fore.YELLOW)
        signal_emoji = "🟢" if "BUY" in signal else ("🔴" if "SELL" in signal else "🟡")

        print(f"\n{Back.WHITE}{Fore.BLACK} 🎯 TRADING RECOMMENDATION {Style.RESET_ALL}")
        print(f"📊 Strategy: {Fore.CYAN}{trading_plan.get('entry_strategy', 'N/A')}{Style.RESET_ALL}")
        print(f"🎯 Signal: {signal_color}{signal} {signal_emoji}{Style.RESET_ALL}")
        print(f"🛡️ Stop Loss: {Fore.RED}₹{trading_plan.get('stop_loss', 0):,.2f}{Style.RESET_ALL}")
        print(f"🎯 Target Price: {Fore.GREEN}₹{target_price:,.2f}{Style.RESET_ALL}")
        print(f"💰 Current Price: {Fore.BLUE}₹{result.get('current_price', 0):,.2f}{Style.RESET_ALL}")

        # Calculate potential return
        current_price = result.get('current_price', 0)
        if current_price > 0 and target_price > 0:
            potential_return = ((target_price - current_price) / current_price) * 100
            return_color = Fore.GREEN if potential_return > 0 else Fore.RED
            print(f"📊 Potential Return: {return_color}{potential_return:+.1f}%{Style.RESET_ALL}")

        # AI Reasoning Section
        print(f"\n{Back.MAGENTA}{Fore.WHITE} 🤖 AI ANALYSIS REASONING {Style.RESET_ALL}")
        reasoning = self.get_detailed_analysis_reasoning(result, system_type)
        for i, reason in enumerate(reasoning, 1):
            print(f"{i:2d}. {reason}")

        # Enhanced Technical Analysis
        self.display_technical_indicators(result)

        # Fundamental Analysis (for Position Trading)
        if system_type == "Position" and result.get('fundamentals'):
            self.display_fundamental_indicators(result)

        # Market Sentiment & News
        self.display_sentiment_analysis(result)

        # Risk Assessment
        self.display_risk_assessment(result, system_type)

        # Investment Recommendation Summary
        self.display_investment_summary(result, system_type, target_price)

        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")

    def display_technical_indicators(self, result):
        """Display technical indicators with explanations."""
        print(f"\n{Back.YELLOW}{Fore.BLACK} 📊 TECHNICAL ANALYSIS BREAKDOWN {Style.RESET_ALL}")

        # RSI Analysis
        rsi = result.get('rsi')
        if rsi is not None:
            rsi_signal, rsi_explanation = self.get_rsi_detailed_analysis(rsi)
            rsi_color = Fore.RED if rsi > 70 else (Fore.GREEN if rsi < 30 else Fore.YELLOW)
            print(f"📈 RSI (14): {rsi_color}{rsi:.2f}{Style.RESET_ALL} - {rsi_signal}")
            print(f"   💡 {rsi_explanation}")

        # MACD Analysis
        macd = result.get('macd', {})
        if macd:
            macd_signal, macd_explanation = self.get_macd_detailed_analysis(macd)
            print(f"📊 MACD: {macd_signal}")
            print(f"   💡 {macd_explanation}")

        # Bollinger Bands
        bb = result.get('bollinger_bands', {})
        if bb:
            bb_signal, bb_explanation = self.get_bb_detailed_analysis(bb)
            print(f"📏 Bollinger Bands: {bb_signal}")
            print(f"   💡 {bb_explanation}")

        # Volume Analysis
        volume_data = result.get('volume_analysis', {})
        if volume_data:
            volume_trend = volume_data.get('trend', 'Normal')
            volume_color = Fore.GREEN if 'High' in volume_trend else Fore.YELLOW
            print(f"📊 Volume Trend: {volume_color}{volume_trend}{Style.RESET_ALL}")

    def display_fundamental_indicators(self, result):
        """Display fundamental analysis with detailed explanations."""
        print(f"\n{Back.MAGENTA}{Fore.WHITE} 📋 FUNDAMENTAL ANALYSIS BREAKDOWN {Style.RESET_ALL}")

        fundamentals = result.get('fundamentals', {})

        # P/E Ratio Analysis
        pe_ratio = fundamentals.get('pe_ratio')
        if pe_ratio is not None:
            pe_signal, pe_explanation = self.get_pe_detailed_analysis(pe_ratio)
            pe_color = Fore.GREEN if pe_ratio < 15 else (Fore.RED if pe_ratio > 25 else Fore.YELLOW)
            print(f"📊 P/E Ratio: {pe_color}{pe_ratio:.2f}{Style.RESET_ALL} - {pe_signal}")
            print(f"   💡 {pe_explanation}")

        # ROE Analysis
        roe = fundamentals.get('roe')
        if roe is not None:
            roe_signal, roe_explanation = self.get_roe_detailed_analysis(roe)
            roe_color = Fore.GREEN if roe > 15 else (Fore.RED if roe < 10 else Fore.YELLOW)
            print(f"📈 ROE: {roe_color}{roe:.2f}%{Style.RESET_ALL} - {roe_signal}")
            print(f"   💡 {roe_explanation}")

        # Debt/Equity Analysis
        de_ratio = fundamentals.get('debt_to_equity')
        if de_ratio is not None:
            de_signal, de_explanation = self.get_de_detailed_analysis(de_ratio)
            de_color = Fore.GREEN if de_ratio < 0.5 else (Fore.RED if de_ratio > 1.5 else Fore.YELLOW)
            print(f"⚖️ Debt/Equity: {de_color}{de_ratio:.2f}{Style.RESET_ALL} - {de_signal}")
            print(f"   💡 {de_explanation}")

    def display_sentiment_analysis(self, result):
        """Display news sentiment analysis with insights."""
        sentiment_data = result.get('sentiment', {})
        if sentiment_data.get('articles'):
            print(f"\n{Back.CYAN}{Fore.BLACK} 📰 MARKET SENTIMENT & NEWS ANALYSIS {Style.RESET_ALL}")

            sentiment_summary = sentiment_data.get('sentiment_summary', {})
            positive = sentiment_summary.get('positive', 0)
            negative = sentiment_summary.get('negative', 0)
            neutral = sentiment_summary.get('neutral', 0)
            total_articles = positive + negative + neutral

            if total_articles > 0:
                pos_pct = (positive / total_articles) * 100
                neg_pct = (negative / total_articles) * 100
                neu_pct = (neutral / total_articles) * 100

                print(f"📊 News Sentiment Distribution:")
                print(f"   🟢 Positive: {positive} articles ({pos_pct:.1f}%)")
                print(f"   🔴 Negative: {negative} articles ({neg_pct:.1f}%)")
                print(f"   🟡 Neutral: {neutral} articles ({neu_pct:.1f}%)")

                # Sentiment interpretation
                if pos_pct > 60:
                    sentiment_interpretation = "Strong positive sentiment suggests market confidence"
                elif neg_pct > 60:
                    sentiment_interpretation = "Strong negative sentiment indicates market concerns"
                elif pos_pct > neg_pct * 1.5:
                    sentiment_interpretation = "Moderately positive sentiment supports bullish outlook"
                elif neg_pct > pos_pct * 1.5:
                    sentiment_interpretation = "Moderately negative sentiment suggests caution"
                else:
                    sentiment_interpretation = "Balanced sentiment - market is undecided"

                print(f"💡 Interpretation: {sentiment_interpretation}")

                # Display recent headlines
                print(f"\n📈 Recent News Headlines:")
                for i, article in enumerate(sentiment_data['articles'][:3], 1):
                    headline = article[:70] + "..." if len(article) > 70 else article
                    print(f"   {i}. {headline}")

    def display_risk_assessment(self, result, system_type):
        """Display comprehensive risk assessment."""
        print(f"\n{Back.RED}{Fore.WHITE} ⚠️ COMPREHENSIVE RISK ASSESSMENT {Style.RESET_ALL}")

        # Volatility analysis
        volatility = result.get('volatility', 50)
        vol_level = "Low" if volatility < 30 else ("Medium" if volatility < 60 else "High")
        vol_color = Fore.GREEN if vol_level == "Low" else (Fore.YELLOW if vol_level == "Medium" else Fore.RED)

        print(f"📊 Price Volatility: {vol_color}{vol_level} ({volatility:.1f}%){Style.RESET_ALL}")

        # Risk score calculation
        risk_factors = []
        risk_score = 0

        # Volatility risk
        if volatility > 60:
            risk_factors.append("High price volatility")
            risk_score += 30
        elif volatility > 40:
            risk_score += 20

        # Technical risk factors
        rsi = result.get('rsi', 50)
        if rsi > 75 or rsi < 25:
            risk_factors.append("Extreme RSI levels indicate potential reversal risk")
            risk_score += 20

        # Fundamental risks (for position trading)
        if system_type == 'Position':
            fundamentals = result.get('fundamentals', {})
            pe_ratio = fundamentals.get('pe_ratio', 20)
            de_ratio = fundamentals.get('debt_to_equity', 1.0)

            if pe_ratio > 30:
                risk_factors.append("High P/E ratio suggests overvaluation risk")
                risk_score += 25

            if de_ratio > 2.0:
                risk_factors.append("High debt levels increase financial risk")
                risk_score += 25

        # Market sentiment risk
        sentiment_data = result.get('sentiment', {})
        if sentiment_data:
            sentiment_summary = sentiment_data.get('sentiment_summary', {})
            negative = sentiment_summary.get('negative', 0)
            positive = sentiment_summary.get('positive', 0)

            if negative > positive * 2:
                risk_factors.append("Predominantly negative news sentiment")
                risk_score += 15

        # Risk level determination
        overall_risk = "Low" if risk_score < 30 else ("Medium" if risk_score < 60 else "High")
        risk_color = Fore.GREEN if overall_risk == "Low" else (Fore.YELLOW if overall_risk == "Medium" else Fore.RED)

        print(f"⚠️ Overall Risk Level: {risk_color}{overall_risk}{Style.RESET_ALL}")

        if risk_factors:
            print(f"🔍 Risk Factors Identified:")
            for i, factor in enumerate(risk_factors, 1):
                print(f"   {i}. {factor}")

        # Risk mitigation suggestions
        print(f"\n💡 Risk Management Suggestions:")
        if overall_risk == "High":
            print(f"   • Consider reducing position size")
            print(f"   • Use tight stop-loss orders")
            print(f"   • Monitor closely for exit opportunities")
        elif overall_risk == "Medium":
            print(f"   • Maintain standard position sizing")
            print(f"   • Set appropriate stop-loss levels")
            print(f"   • Regular monitoring recommended")
        else:
            print(f"   • Standard risk management practices apply")
            print(f"   • Suitable for normal position sizing")

    def display_investment_summary(self, result, system_type, target_price):
        """Display final investment summary and recommendations."""
        print(f"\n{Back.GREEN}{Fore.BLACK} 📋 INVESTMENT SUMMARY & FINAL RECOMMENDATION {Style.RESET_ALL}")

        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        score = result.get(score_key, 0)
        current_price = result.get('current_price', 0)

        # Investment grade
        if score >= 80:
            grade = "A+ (Excellent)"
            grade_color = Fore.GREEN
        elif score >= 70:
            grade = "A (Good)"
            grade_color = Fore.GREEN
        elif score >= 60:
            grade = "B (Average)"
            grade_color = Fore.YELLOW
        elif score >= 50:
            grade = "C (Below Average)"
            grade_color = Fore.YELLOW
        else:
            grade = "D (Poor)"
            grade_color = Fore.RED

        print(f"🏆 Investment Grade: {grade_color}{grade}{Style.RESET_ALL}")

        # Time horizon
        time_horizon = "1-4 weeks" if system_type == "Swing" else "6-18 months"
        print(f"⏰ Recommended Time Horizon: {Fore.CYAN}{time_horizon}{Style.RESET_ALL}")

        # Position sizing recommendation
        user_risk = self.user_profile.get('risk_tolerance', 'MEDIUM')
        if score >= 70 and user_risk == 'HIGH':
            position_size = "Standard to Large (3-8% of portfolio)"
        elif score >= 60:
            position_size = "Standard (2-5% of portfolio)"
        elif score >= 50:
            position_size = "Small (1-3% of portfolio)"
        else:
            position_size = "Very Small or Avoid (0-2% of portfolio)"

        print(f"💼 Position Size Recommendation: {Fore.MAGENTA}{position_size}{Style.RESET_ALL}")

        # Entry strategy
        if score >= 70:
            entry_strategy = "Consider immediate entry or dollar-cost averaging"
        elif score >= 60:
            entry_strategy = "Wait for better entry point or small initial position"
        else:
            entry_strategy = "Avoid or wait for significant improvement in fundamentals"

        print(f"📈 Entry Strategy: {Fore.BLUE}{entry_strategy}{Style.RESET_ALL}")

    # Detailed analysis methods
    def get_rsi_detailed_analysis(self, rsi):
        """Get detailed RSI analysis."""
        if rsi > 80:
            return "Severely Overbought", "Stock is in extreme overbought territory. High probability of price correction."
        elif rsi > 70:
            return "Overbought", "Stock is overbought. Consider taking profits or waiting for pullback."
        elif rsi < 20:
            return "Severely Oversold", "Stock is in extreme oversold territory. Potential buying opportunity."
        elif rsi < 30:
            return "Oversold", "Stock is oversold. May be good entry point for value investors."
        elif 40 <= rsi <= 60:
            return "Neutral Zone", "RSI is in neutral territory. No strong momentum signals."
        else:
            return "Trending", "RSI shows directional momentum. Confirm with other indicators."

    def get_macd_detailed_analysis(self, macd):
        """Get detailed MACD analysis."""
        line = macd.get('line', 0)
        signal = macd.get('signal', 0)
        histogram = macd.get('histogram', 0)

        if line > signal and histogram > 0:
            return "Strong Bullish", "MACD line above signal with positive momentum. Strong buy signal."
        elif line > signal:
            return "Bullish", "MACD line above signal line. Positive momentum building."
        elif line < signal and histogram < 0:
            return "Strong Bearish", "MACD line below signal with negative momentum. Strong sell signal."
        else:
            return "Bearish", "MACD line below signal line. Negative momentum building."

    def get_bb_detailed_analysis(self, bb):
        """Get detailed Bollinger Bands analysis."""
        position = bb.get('position', 'middle')
        if position == 'upper':
            return "Near Upper Band", "Price near upper Bollinger Band. Overbought condition possible."
        elif position == 'lower':
            return "Near Lower Band", "Price near lower Bollinger Band. Oversold condition possible."
        else:
            return "Middle Range", "Price in middle of Bollinger Bands. Normal trading range."

    def get_pe_detailed_analysis(self, pe_ratio):
        """Get detailed P/E ratio analysis."""
        if pe_ratio < 10:
            return "Very Undervalued", "Extremely low P/E suggests deep value or potential problems. Investigate further."
        elif pe_ratio < 15:
            return "Undervalued", "Low P/E ratio suggests stock may be undervalued relative to earnings."
        elif pe_ratio <= 25:
            return "Fair Value", "P/E ratio in reasonable range. Stock appears fairly valued."
        elif pe_ratio <= 35:
            return "Overvalued", "High P/E suggests stock may be overvalued. Growth expectations may be high."
        else:
            return "Very Overvalued", "Extremely high P/E. Stock may be in bubble territory or have exceptional growth."

    def get_roe_detailed_analysis(self, roe):
        """Get detailed ROE analysis."""
        if roe > 20:
            return "Excellent", "Outstanding return on equity indicates very efficient use of shareholder capital."
        elif roe > 15:
            return "Good", "Strong return on equity shows good management efficiency and profitability."
        elif roe > 10:
            return "Average", "Moderate return on equity is acceptable but room for improvement exists."
        elif roe > 5:
            return "Below Average", "Low return on equity suggests inefficient use of capital or operational issues."
        else:
            return "Poor", "Very low or negative ROE indicates significant operational or financial problems."

    def get_de_detailed_analysis(self, de_ratio):
        """Get detailed Debt/Equity analysis."""
        if de_ratio < 0.3:
            return "Very Conservative", "Very low debt levels provide financial stability but may limit growth."
        elif de_ratio < 0.6:
            return "Conservative", "Healthy debt levels provide good balance between safety and growth."
        elif de_ratio <= 1.0:
            return "Moderate", "Reasonable debt levels. Monitor cash flow and interest coverage."
        elif de_ratio <= 2.0:
            return "High Risk", "High debt levels increase financial risk, especially in economic downturns."
        else:
            return "Very High Risk", "Excessive debt levels pose significant financial stability concerns."

    def display_portfolio(self, portfolio, capital, system_type):
        """Enhanced portfolio display with more details."""
        if not portfolio:
            print(f"❌ Could not generate a portfolio with the given criteria.")
            return

        total_allocated = sum(data['investment_amount'] for data in portfolio.values())
        remaining = capital - total_allocated

        print(f"\n💼 YOUR PERSONALIZED {system_type.upper()} PORTFOLIO ")
        print("=" * 90)
        print(f"👤 Investor: {self.user_profile.get('name', 'N/A')} | "
              f"💰 Budget: ₹{capital:,.2f} | "
              f"⚖️ Risk: {self.user_profile.get('risk_tolerance', 'N/A')}")
        print("=" * 90)

        print(f"{'Symbol':<10} {'Company':<25} {'Shares':<8} {'Amount':<12} {'Weight':<8} {'Score':<8}")
        print("-" * 90)

        for symbol, data in portfolio.items():
            weight = (data['investment_amount'] / capital) * 100
            score = data.get('score', 0)
            score_color = "HIGH" if score >= 70 else ("MED" if score >= 55 else "LOW")

            print(f"{symbol:<10} "
                  f"{data['company_name'][:23]:<25} "
                  f"{data['num_shares']:<8} "
                  f"₹{data['investment_amount']:>9,.0f} "
                  f"{weight:>6.1f}% "
                  f"{score_color} {score:>6.1f}")

        print("-" * 90)
        print(f"💰 Total Allocated: ₹{total_allocated:,.2f} "
              f"({(total_allocated / capital) * 100:.1f}%)")
        print(f"💵 Remaining Cash: ₹{remaining:,.2f} "
              f"({(remaining / capital) * 100:.1f}%)")

        # Portfolio insights
        print(f"\n📊 PORTFOLIO INSIGHTS")
        diversification = len(portfolio)
        print(f"🎯 Diversification: {diversification} stocks")

        avg_score = sum(data.get('score', 0) for data in portfolio.values()) / len(portfolio)
        print(f"📈 Average AI Score: {avg_score:.1f}/100")

        print("=" * 90)

    def run_enhanced_portfolio_builder(self, system, system_type):
        """Enhanced portfolio builder with more customization options."""
        print(f"\n🎯 PERSONALIZED {system_type.upper()} PORTFOLIO BUILDER")

        # Budget input with validation
        while True:
            try:
                budget = float(input(f"\n💰 Enter your investment budget (INR): ₹"))
                if budget >= 10000:
                    break
                else:
                    print("❌ Minimum budget should be ₹10,000")
            except ValueError:
                print("❌ Please enter a valid amount")

        # Use profile risk tolerance or ask again
        use_profile_risk = input(
            f"\n⚖️ Use your profile risk tolerance ({self.user_profile.get('risk_tolerance', 'N/A')})? (y/n): ").lower() == 'y'

        if use_profile_risk:
            risk_appetite = self.user_profile.get('risk_tolerance', 'MEDIUM')
        else:
            print(f"\n⚖️ Risk Appetite Options:")
            print("1. 🟢 Low Risk - Conservative approach")
            print("2. 🟡 Medium Risk - Balanced strategy")
            print("3. 🔴 High Risk - Aggressive growth")
            risk_choice = self.get_valid_choice([1, 2, 3], "Select risk level: ")
            risk_map = {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH'}
            risk_appetite = risk_map[risk_choice]

        # Additional customization for position trading
        if system_type == "Position":
            print(f"\n⏰ Investment Time Horizon:")
            print("1. 6-12 months")
            print("2. 1-2 years")
            print("3. 2-5 years")
            print("4. 5+ years")
            time_choice = self.get_valid_choice([1, 2, 3, 4], "Select time period: ")
            time_map = {1: 9, 2: 18, 3: 36, 4: 60}
            time_period = time_map[time_choice]

            print(f"\n🔄 Creating your personalized position portfolio...")
            time.sleep(1)

            results = system.create_personalized_portfolio(risk_appetite, time_period, budget)
            portfolio = results.get('portfolio', {})
        else:
            print(f"\n🔄 Analyzing swing trading opportunities...")
            time.sleep(1)

            all_stocks = system.get_all_stock_symbols()
            all_results = system.analyze_multiple_stocks(all_stocks)
            filtered = system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
            portfolio = system.generate_portfolio_allocation(filtered, budget, risk_appetite)

        self.display_portfolio(portfolio, budget, system_type)

        # Save portfolio option
        save_portfolio = input(f"\n💾 Would you like to save this portfolio? (y/n): ").lower() == 'y'
        if save_portfolio:
            filename = f"portfolio_{system_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            portfolio_data = {
                'user_profile': self.user_profile,
                'portfolio': portfolio,
                'budget': budget,
                'risk_appetite': risk_appetite,
                'system_type': system_type,
                'created_at': datetime.now().isoformat()
            }

            try:
                with open(filename, 'w') as f:
                    json.dump(portfolio_data, f, indent=2, default=str)
                print(f"✅ Portfolio saved as {filename}")
            except Exception as e:
                print(f"❌ Could not save portfolio: {e}")

    def run_system_menu(self, system, system_type):
        """Enhanced system menu with more options."""
        while True:
            print(f"\n📈 {system_type.upper()} TRADING SYSTEM")
            print(f"Choose your action:")
            print("1. 📊 Show All Available Stocks")
            print("2. 🔍 Analyze Single Stock (Deep Dive)")
            print("3. 💼 Create Personalized Portfolio")
            print("4. 📈 Market Overview & Top Picks")
            print("5. ⚙️ Update User Profile")
            print("6. 🔙 Back to Main Menu")

            choice = input(f"\nEnter your choice (1-6): ").strip()

            if choice == '1':
                stocks = system.get_all_stock_symbols()
                print(f"\n📊 AVAILABLE STOCKS FOR {system_type.upper()} TRADING")
                print(f"Total stocks available: {len(stocks)}")

                # Display in columns
                for i in range(0, len(stocks), 5):
                    row = stocks[i:i + 5]
                    print(" | ".join(f"{stock:<8}" for stock in row))

            elif choice == '2':
                symbol = input(f"\n🔍 Enter stock symbol to analyze: ").upper()
                print(f"\n🔄 Analyzing {symbol}...")
                time.sleep(1)

                if system_type == 'Swing':
                    result = system.analyze_swing_trading_stock(symbol)
                else:
                    result = system.analyze_position_trading_stock(symbol)

                self.display_transparent_analysis(result, system_type)

            elif choice == '3':
                self.run_enhanced_portfolio_builder(system, system_type)

            elif choice == '4':
                self.show_market_overview(system, system_type)

            elif choice == '5':
                print(f"\n🔍 Updating your profile...")
                self.create_user_profile()

            elif choice == '6':
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def show_market_overview(self, system, system_type):
        """Show market overview and top stock picks."""
        print(f"\n🌍 MARKET OVERVIEW & TOP PICKS - {system_type.upper()}")
        print(f"🔄 Analyzing market conditions...")
        time.sleep(2)

        try:
            # Get all stocks and analyze top performers
            all_stocks = system.get_all_stock_symbols()[:20]  # Limit for performance
            print(f"📊 Scanning {len(all_stocks)} stocks...")

            results = []
            for i, stock in enumerate(all_stocks, 1):
                print(f"\r🔍 Analyzing {stock}... ({i}/{len(all_stocks)})", end="", flush=True)
                try:
                    if system_type == 'Swing':
                        result = system.analyze_swing_trading_stock(stock)
                        score_key = 'swing_score'
                    else:
                        result = system.analyze_position_trading_stock(stock)
                        score_key = 'position_score'

                    if result and result.get(score_key):
                        results.append({
                            'symbol': stock,
                            'score': result.get(score_key, 0),
                            'signal': result.get('trading_plan', {}).get('entry_signal', 'HOLD'),
                            'company': result.get('company_name', 'N/A'),
                            'current_price': result.get('current_price', 0)
                        })
                except Exception:
                    continue

            print(f"\n✅ Analysis complete!")

            # Sort by score and display top picks
            results.sort(key=lambda x: x['score'], reverse=True)
            top_picks = results[:10]

            print(f"\n🏆 TOP 10 PICKS FOR {system_type.upper()} TRADING")
            print(f"{'Rank':<5} {'Symbol':<8} {'Company':<25} {'Score':<8} {'Signal':<8} {'Price':<12}")
            print("-" * 75)

            for i, stock in enumerate(top_picks, 1):
                score_level = "HIGH" if stock['score'] >= 70 else ("MED" if stock['score'] >= 55 else "LOW")
                signal_indicator = "BUY" if "BUY" in stock['signal'] else (
                    "SELL" if "SELL" in stock['signal'] else "HOLD")

                print(f"{i:<5} "
                      f"{stock['symbol']:<8} "
                      f"{stock['company'][:23]:<25} "
                      f"{score_level} {stock['score']:<5.1f} "
                      f"{signal_indicator:<8} "
                      f"₹{stock['current_price']:<10,.2f}")

            # Market sentiment summary
            buy_signals = len([s for s in results if "BUY" in s['signal']])
            sell_signals = len([s for s in results if "SELL" in s['signal']])
            hold_signals = len([s for s in results if "HOLD" in s['signal']])

            print(f"\n📊 MARKET SENTIMENT SUMMARY")
            print(f"🟢 Buy Signals: {buy_signals} | "
                  f"🔴 Sell Signals: {sell_signals} | "
                  f"🟡 Hold Signals: {hold_signals}")

            if buy_signals > sell_signals:
                market_sentiment = "Bullish"
            elif sell_signals > buy_signals:
                market_sentiment = "Bearish"
            else:
                market_sentiment = "Neutral"

            print(f"🔮 Overall Market Sentiment: {market_sentiment}")

        except Exception as e:
            print(f"\n❌ Error generating market overview: {e}")

    def display_welcome_screen(self):
        """Display an enhanced welcome screen."""
        print(f"\n" + "=" * 80)
        print(f"{'🚀 ADVANCED AI-POWERED TRADING PLATFORM 🚀':^80}")
        print("=" * 80)

        features = [
            "🎯 Personalized Investment Strategies",
            "📊 Real-time Technical & Fundamental Analysis",
            "📰 AI-Powered News Sentiment Analysis",
            "💼 Custom Portfolio Generation",
            "⚡ Risk Assessment & Management",
            "🔍 Deep Stock Analysis with Transparency"
        ]

        print(f"\n✨ Platform Features:")
        for feature in features:
            print(f"   {feature}")

        print(f"\n🎓 Suitable for all experience levels - from beginners to experts!")

    def run(self):
        """Main application loop."""
        init(autoreset=True)

        self.display_welcome_screen()

        # Create user profile
        create_profile = input(
            f"\n🎯 Would you like to create a personalized profile for better recommendations? (y/n): ").lower() == 'y'
        if create_profile:
            self.create_user_profile()
        else:
            # Set default profile
            self.user_profile = {
                'name': 'Guest User',
                'experience': 'Intermediate',
                'goal': 'Wealth Creation',
                'risk_tolerance': 'MEDIUM',
                'age_group': '31-45'
            }
            print(f"👋 Welcome, Guest User! Using default profile.")

        # Main menu loop
        while True:
            print(f"\n🎯 MAIN TRADING MENU 🎯")
            print(f"👋 Welcome back, {self.user_profile['name']}!")
            print(f"⚖️ Your Risk Profile: {self.user_profile['risk_tolerance']}")

            print(f"\nChoose your trading style:")
            print("1. 🚀 Swing Trading - Short-term opportunities (1-30 days)")
            print("   └─ High potential returns, active monitoring required")
            print("2. ⚖️ Position Trading - Long-term investments (3-12 months)")
            print("   └─ Steady growth, fundamental analysis focused")
            print("3. 📊 Compare Both Strategies")
            print("4. ⚙️ Settings & Profile Management")
            print("5. 🚪 Exit Platform")

            choice = input(f"\nEnter your choice (1-5): ").strip()

            if choice == "1":
                self.run_system_menu(self.swing_system, "Swing")
            elif choice == "2":
                self.run_system_menu(self.position_system, "Position")
            elif choice == "3":
                self.compare_strategies()
            elif choice == "4":
                self.settings_menu()
            elif choice == "5":
                self.display_goodbye_message()
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def compare_strategies(self):
        """Compare swing vs position trading strategies."""
        print(f"\n⚖️ STRATEGY COMPARISON: SWING VS POSITION TRADING")

        symbol = input(f"\n🔍 Enter stock symbol to compare strategies: ").upper()

        print(f"\n🔄 Running comprehensive analysis on {symbol}...")
        time.sleep(2)

        try:
            # Analyze with both systems
            swing_result = self.swing_system.analyze_swing_trading_stock(symbol)
            position_result = self.position_system.analyze_position_trading_stock(symbol)

            if swing_result and position_result:
                swing_score = swing_result.get('swing_score', 0)
                position_score = position_result.get('position_score', 0)

                print(f"\n📊 STRATEGY COMPARISON RESULTS")
                print(f"{'Strategy':<20} {'Score':<10} {'Signal':<12} {'Time Horizon':<15} {'Risk Level'}")
                print("-" * 75)

                # Swing trading row
                swing_signal = swing_result.get('trading_plan', {}).get('entry_signal', 'N/A')
                swing_level = "HIGH" if swing_score >= 70 else ("MED" if swing_score >= 55 else "LOW")
                print(f"🚀 Swing Trading     {swing_level} {swing_score:<7.1f} "
                      f"{swing_signal:<12} {'1-30 days':<15} High")

                # Position trading row
                position_signal = position_result.get('trading_plan', {}).get('entry_signal', 'N/A')
                position_level = "HIGH" if position_score >= 70 else ("MED" if position_score >= 55 else "LOW")
                print(f"⚖️ Position Trading  {position_level} {position_score:<7.1f} "
                      f"{position_signal:<12} {'3-12 months':<15} Medium")

                # Recommendation based on user profile
                print(f"\n🎯 PERSONALIZED RECOMMENDATION")
                user_risk = self.user_profile.get('risk_tolerance', 'MEDIUM')

                if user_risk == 'HIGH' and swing_score > position_score:
                    recommended = "Swing Trading"
                    reason = "matches your high risk appetite and shows better short-term potential"
                elif user_risk == 'LOW' and position_score > swing_score:
                    recommended = "Position Trading"
                    reason = "aligns with your conservative approach and long-term focus"
                else:
                    if swing_score > position_score:
                        recommended = "Swing Trading"
                        reason = f"shows higher AI score ({swing_score:.1f} vs {position_score:.1f})"
                    else:
                        recommended = "Position Trading"
                        reason = f"shows higher AI score ({position_score:.1f} vs {swing_score:.1f})"

                print(f"🏆 Recommended Strategy: {recommended}")
                print(f"🔍 Reason: {reason}")

            else:
                print(f"❌ Could not analyze {symbol} with both strategies")

        except Exception as e:
            print(f"❌ Error during comparison: {e}")

    def settings_menu(self):
        """Settings and profile management menu."""
        while True:
            print(f"\n⚙️ SETTINGS & PROFILE MANAGEMENT")
            print("1. 🔍 Update Trading Profile")
            print("2. 👤 View Current Profile")
            print("3. 💾 Export Profile")
            print("4. 📊 View Trading History")
            print("5. 🔙 Back to Main Menu")

            choice = input(f"\nEnter your choice (1-5): ").strip()

            if choice == "1":
                self.create_user_profile()
            elif choice == "2":
                self.display_user_profile()
            elif choice == "3":
                self.export_profile()
            elif choice == "4":
                print("📊 Trading history feature coming soon!")
            elif choice == "5":
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def export_profile(self):
        """Export user profile to JSON file."""
        try:
            filename = f"user_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
            print(f"✅ Profile exported as {filename}")
        except Exception as e:
            print(f"❌ Could not export profile: {e}")

    def display_goodbye_message(self):
        """Display goodbye message with trading tips."""
        print(f"\n👋 THANK YOU FOR USING OUR PLATFORM! 👋")
        print("=" * 70)

        tips = [
            "💡 Remember: Never invest more than you can afford to lose",
            "📊 Always do your own research before making investment decisions",
            "⚖️ Diversification is key to managing risk",
            "📈 Stay updated with market news and trends",
            "🎯 Stick to your investment strategy and risk tolerance"
        ]

        print(f"\n💡 Final Trading Tips:")
        for tip in tips:
            print(f"   {tip}")

        print(f"\n🚀 Happy Trading, {self.user_profile['name']}! See you next time!")
        print("=" * 70)


def main():
    """Main entry point."""
    try:
        platform = InteractiveTradingPlatform()
        platform.run()
    except KeyboardInterrupt:
        print(f"\n\n👋 Platform closed by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        logger.error(f"Critical error in main: {e}")


if __name__ == "__main__":
    main()