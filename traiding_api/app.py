from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from functools import wraps
import traceback
import math

# --- Graceful Import of Trading Systems ---
SYSTEMS_AVAILABLE = False
try:
    from systems.position_trading import EnhancedPositionTradingSystem
    from systems.swing_trading import EnhancedSwingTradingSystem

    SYSTEMS_AVAILABLE = True
    logging.info("Successfully imported trading system modules.")
except ImportError as e:
    logging.critical(f"Could not import trading systems: {e}. API will run in a degraded mode.")

app = Flask(__name__)
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Validation Functions ---
def validate_budget(budget):
    try:
        budget = float(budget)
        if not 10000 <= budget <= 10000000:
            raise ValueError("Budget must be between ₹10,000 and ₹10,000,000")
        return budget
    except (TypeError, ValueError):
        raise ValueError("Budget must be a valid number")


def validate_risk_appetite(risk):
    if not isinstance(risk, str) or risk.upper() not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError("Risk appetite must be LOW, MEDIUM, or HIGH")
    return risk.upper()


def validate_symbol(symbol):
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string")
    return symbol.upper().strip()


def validate_time_period(time_period):
    try:
        time_period = int(time_period)
        if time_period not in [9, 18, 36, 60]:
            raise ValueError("Time period must be 9, 18, 36, or 60 months")
        return time_period
    except (TypeError, ValueError):
        raise ValueError("Time period must be a valid integer")


# --- Simple In-Memory Cache ---
simple_cache = {}
CACHE_TIMEOUT = 300  # 5 minutes


def get_from_cache(key):
    if key in simple_cache:
        data, timestamp = simple_cache[key]
        if (datetime.now().timestamp() - timestamp) < CACHE_TIMEOUT:
            return data
        del simple_cache[key]
    return None


def set_cache(key, value):
    simple_cache[key] = (value, datetime.now().timestamp())


class TradingAPI:
    """Handles all trading logic and system interactions."""

    def __init__(self):
        self.swing_system = None
        self.position_system = None
        if SYSTEMS_AVAILABLE:
            self.initialize_systems()
        else:
            logger.warning("Trading systems not imported. API is in a degraded state.")

    def initialize_systems(self):
        """Initializes each trading system individually for better resilience."""
        logger.info("Initializing trading systems...")
        try:
            self.swing_system = EnhancedSwingTradingSystem()
            logger.info("✅ EnhancedSwingTradingSystem initialized successfully.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedSwingTradingSystem", exc_info=True)

        try:
            self.position_system = EnhancedPositionTradingSystem()
            logger.info("✅ EnhancedPositionTradingSystem initialized successfully.")
        except Exception:
            logger.critical("❌ FAILED to initialize EnhancedPositionTradingSystem", exc_info=True)

    def _clean_fundamental_data(self, fundamentals):
        """Sanitizes and corrects fundamental data points known to have issues."""
        if not isinstance(fundamentals, dict):
            return {}

        cleaned_data = fundamentals.copy()


        if 'debt_to_equity' in cleaned_data and isinstance(cleaned_data['debt_to_equity'], (int, float)):
            if cleaned_data['debt_to_equity'] > 3.0:
                cleaned_data['debt_to_equity'] = round(cleaned_data['debt_to_equity'] / 100, 3)


        if 'book_value' in cleaned_data and isinstance(cleaned_data['book_value'], (int, float)):
            if cleaned_data['book_value'] < 10:
                cleaned_data['book_value'] = round(cleaned_data['book_value'] * 100, 2)


        if 'price_to_book' in cleaned_data and isinstance(cleaned_data['price_to_book'], (int, float)):
            if cleaned_data['price_to_book'] > 100:
                cleaned_data['price_to_book'] = round(cleaned_data['price_to_book'] / 50, 2)


        if 'price_to_sales' in cleaned_data and isinstance(cleaned_data['price_to_sales'], (int, float)):
            if cleaned_data['price_to_sales'] > 50:
                cleaned_data['price_to_sales'] = round(cleaned_data['price_to_sales'] / 30, 2)


        if 'pe_ratio' in cleaned_data and isinstance(cleaned_data['pe_ratio'], (int, float)):
            if cleaned_data['pe_ratio'] < 10 or cleaned_data['pe_ratio'] > 100:
                cleaned_data['pe_ratio'] = round(cleaned_data['pe_ratio'] * 2.5, 2)


        if 'dividend_yield' in cleaned_data and isinstance(cleaned_data['dividend_yield'], (int, float)):
            if cleaned_data['dividend_yield'] > 10:
                cleaned_data['dividend_yield'] = round(cleaned_data['dividend_yield'] / 10, 2)


        for key in ['market_cap', 'enterprise_value']:
            if key in cleaned_data and isinstance(cleaned_data[key], (int, float)):
                cleaned_data[key] = f"{cleaned_data[key]:,}"


        for key in ['operating_margin', 'profit_margin', 'revenue_growth', 'earnings_growth']:
            if key in cleaned_data and isinstance(cleaned_data[key], float) and abs(cleaned_data[key]) < 1:
                cleaned_data[key] = f"{cleaned_data[key] * 100:.2f}%"

        return cleaned_data

    def calculate_target_price(self, result, system_type):
        current_price = result.get('current_price', 0)
        if current_price <= 0: return 0.0

        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        ai_score = result.get(score_key, 0)

        multiplier = 1.0

        if system_type == 'Swing':
            if ai_score >= 80:
                multiplier = 1.15
            elif ai_score >= 70:
                multiplier = 1.10
            elif ai_score >= 60:
                multiplier = 1.07
            else:
                multiplier = 1.04

        else:  # Position
            if ai_score >= 80:
                multiplier = 1.50
            elif ai_score >= 70:
                multiplier = 1.35
            elif ai_score >= 60:
                multiplier = 1.25
            else:
                multiplier = 1.20

        rsi = result.get('technical_indicators', {}).get('rsi', 50)
        if rsi < 30:
            multiplier += 0.02
        elif rsi > 70:
            multiplier -= 0.02

        return current_price * multiplier

    def generate_trading_plan(self, result, system_type):
        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        score = result.get(score_key, 0)
        current_price = result.get('current_price', 0)
        target_price = self.calculate_target_price(result, system_type)
        sentiment = result.get('sentiment', {}).get('overall_sentiment', 'Neutral').lower()

        if current_price <= 0:
            return {'signal': 'UNAVAILABLE', 'strategy': 'Price data not available.', 'entry_price': 'N/A',
                    'stop_loss': 'N/A', 'target_price': 'N/A'}

        signal = "AVOID"
        strategy = "The stock does not meet the criteria for a favorable entry. The risk/reward ratio is not optimal at the current price."

        if score >= 80:
            signal = "STRONG BUY"
            strategy = f"A high-conviction BUY signal driven by an excellent AI score. The analysis indicates strong potential for upward movement in the {system_type.lower()} horizon, supported by a '{sentiment}' market sentiment."
        elif score >= 70:
            signal = "BUY"
            strategy = f"A solid BUY signal based on a good AI score. The stock shows positive indicators for a {system_type.lower()} trade, aligning with the current '{sentiment}' sentiment."
        elif score >= 60:
            signal = "HOLD / MONITOR"
            strategy = "The stock shows neutral to slightly positive signs. It's recommended to hold existing positions and monitor for a stronger buy signal before entering. The current setup lacks a clear catalyst."
        elif score >= 50:
            signal = "HOLD / MONITOR"
            strategy = "The technical and fundamental signals are mixed. While not a sell signal, caution is advised. Wait for a clearer trend to emerge before committing new capital."
        else:  # score < 50
            strategy = "The analysis suggests potential weakness or unfavorable conditions for this stock. It is advised to avoid this stock and look for stronger opportunities elsewhere."

        stop_loss_percentage = 0.95 if system_type == 'Swing' else 0.90
        stop_loss = current_price * stop_loss_percentage if "BUY" in signal else "N/A"

        return {
            'signal': signal, 'strategy': strategy, 'entry_price': f"Around {current_price:.2f}",
            'stop_loss': f"{stop_loss:.2f}" if isinstance(stop_loss, float) else stop_loss,
            'target_price': f"{target_price:.2f}",
        }

    def format_analysis_response(self, result, system_type):
        if not result: return None
        score_key = 'swing_score' if system_type == 'Swing' else 'position_score'
        score = result.get(score_key, 0)
        current_price = result.get('current_price', 0)
        target_price = self.calculate_target_price(result, system_type)
        potential_return = ((target_price - current_price) / current_price) * 100 if current_price > 0 else 0

        grade = "D (Poor)"
        if score >= 80:
            grade = "A+ (Excellent)"
        elif score >= 70:
            grade = "A (Good)"
        elif score >= 60:
            grade = "B (Average)"
        elif score >= 50:
            grade = "C (Below Average)"

        trading_plan = self.generate_trading_plan(result, system_type)

        sentiment_data = result.get('sentiment', {})
        if system_type == 'Position' and result.get('mda_analysis'):
            sentiment_data['mda_tone'] = result['mda_analysis'].get('tone')
            sentiment_data['mda_score'] = result['mda_analysis'].get('score')

        cleaned_fundamentals = self._clean_fundamental_data(result.get('fundamentals', {}))

        return {
            'symbol': result.get('symbol', 'N/A'),
            'company_name': result.get('company_name', 'N/A'),
            'analysis_timestamp': datetime.now().isoformat(),
            'system_type': system_type,
            'overall_score': score,
            'investment_grade': grade,
            'current_price': current_price,
            'target_price': target_price,
            'potential_return': potential_return,
            'trading_plan': trading_plan,
            'technical_indicators': result.get('technical_indicators', {}),
            'fundamentals': cleaned_fundamentals,
            'sentiment': sentiment_data,
            'time_horizon': "1-4 weeks" if system_type == "Swing" else "6-18 months"
        }

    def _calculate_shares(self, portfolio_list):
        for item in portfolio_list:
            investment_amount = item.get('investment_amount', 0)
            price = item.get('price', item.get('current_price', 0))
            if price > 0:
                item['number_of_shares'] = math.floor(investment_amount / price)
            else:
                item['number_of_shares'] = 0
        return portfolio_list

    def generate_swing_portfolio(self, budget, risk_appetite):
        if not self.swing_system: raise ConnectionAbortedError('Swing trading system not available')
        all_stocks = self.swing_system.get_all_stock_symbols()
        all_results = self.swing_system.analyze_multiple_stocks(all_stocks)
        filtered = self.swing_system.filter_stocks_by_risk_appetite(all_results, risk_appetite)
        portfolio_list = self.swing_system.generate_portfolio_allocation(filtered, budget, risk_appetite)
        portfolio_with_shares = self._calculate_shares(portfolio_list)
        total_allocated = sum(item.get('investment_amount', 0) for item in portfolio_with_shares)
        avg_score = sum(item.get('score', 0) for item in portfolio_with_shares) / len(
            portfolio_with_shares) if portfolio_with_shares else 0
        return {'portfolio': portfolio_with_shares,
                'summary': {'total_budget': budget, 'total_allocated': total_allocated,
                            'remaining_cash': budget - total_allocated, 'diversification': len(portfolio_with_shares),
                            'average_score': avg_score, }}

    def generate_position_portfolio(self, budget, risk_appetite, time_period):
        if not self.position_system: raise ConnectionAbortedError('Position trading system not available')
        results = self.position_system.create_personalized_portfolio(risk_appetite, time_period, budget)
        portfolio_data = results.get('portfolio', {})
        portfolio_list = []
        if isinstance(portfolio_data, dict):
            for symbol, details in portfolio_data.items():
                details['symbol'] = symbol
                portfolio_list.append(details)
        else:
            portfolio_list = portfolio_data
        portfolio_with_shares = self._calculate_shares(portfolio_list)
        total_allocated = sum(item.get('investment_amount', 0) for item in portfolio_with_shares)
        avg_score = sum(item.get('score', 0) for item in portfolio_with_shares) / len(
            portfolio_with_shares) if portfolio_with_shares else 0
        return {'portfolio': portfolio_with_shares,
                'summary': {'total_budget': budget, 'total_allocated': total_allocated,
                            'remaining_cash': budget - total_allocated, 'diversification': len(portfolio_with_shares),
                            'average_score': avg_score, }}


trading_api = TradingAPI()


# --- API Endpoints ---
@app.route('/api/stocks', methods=['GET'])
def get_all_stocks():
    cache_key = "all_stocks"
    if cached := get_from_cache(cache_key): return jsonify({'success': True, 'data': cached})
    if not trading_api.swing_system: return jsonify({'success': False, 'error': 'Trading system not available'}), 503
    stocks = trading_api.swing_system.get_all_stock_symbols()
    result = {'stocks': stocks, 'total_count': len(stocks)}
    set_cache(cache_key, result)
    return jsonify({'success': True, 'data': result})


def analyze_stock(system_type, symbol):
    try:
        symbol = validate_symbol(symbol)
        cache_key = f"{system_type}_analysis_{symbol}"
        if cached := get_from_cache(cache_key): return jsonify({'success': True, 'data': cached})
        system = getattr(trading_api, f"{system_type}_system")
        if not system: return jsonify(
            {'success': False, 'error': f'{system_type.capitalize()} trading system not available'}), 503
        analysis_func = getattr(system, f"analyze_{system_type}_trading_stock")
        result = analysis_func(symbol)
        if not result: return jsonify({'success': False, 'error': f'Could not analyze stock {symbol}'}), 404
        formatted_result = trading_api.format_analysis_response(result, system_type.capitalize())
        set_cache(cache_key, formatted_result)
        return jsonify({'success': True, 'data': formatted_result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analyze/{system_type}/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/analyze/swing/<symbol>', methods=['GET'])
def analyze_swing_stock_endpoint(symbol): return analyze_stock('swing', symbol)


@app.route('/api/analyze/position/<symbol>', methods=['GET'])
def analyze_position_stock_endpoint(symbol): return analyze_stock('position', symbol)


@app.route('/api/portfolio/swing', methods=['POST'])
def create_swing_portfolio_endpoint():
    try:
        data = request.get_json()
        if not data: return jsonify({'success': False, 'error': 'Request body cannot be empty'}), 400
        budget = validate_budget(data.get('budget'))
        risk = validate_risk_appetite(data.get('risk_appetite'))
        result = trading_api.generate_swing_portfolio(budget, risk)
        return jsonify({'success': True, 'data': result})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in portfolio/swing: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/portfolio/position', methods=['POST'])
def create_position_portfolio_endpoint():
    try:
        data = request.get_json()
        if not data: return jsonify({'success': False, 'error': 'Request body cannot be empty'}), 400
        budget = validate_budget(data.get('budget'))
        risk = validate_risk_appetite(data.get('risk_appetite'))
        time_period = validate_time_period(data.get('time_period'))
        result = trading_api.generate_position_portfolio(budget, risk, time_period)
        return jsonify({'success': True, 'data': result})
    except (ValueError, KeyError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in portfolio/position: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


@app.route('/api/compare/<symbol>', methods=['GET'])
def compare_strategies_endpoint(symbol):
    try:
        symbol = validate_symbol(symbol)
        cache_key = f"compare_{symbol}"
        if cached := get_from_cache(cache_key): return jsonify({'success': True, 'data': cached})
        if not trading_api.swing_system or not trading_api.position_system: return jsonify(
            {'success': False, 'error': 'One or more trading systems are unavailable'}), 503
        swing_result = trading_api.swing_system.analyze_swing_trading_stock(symbol)
        position_result = trading_api.position_system.analyze_position_trading_stock(symbol)
        if not swing_result or not position_result: return jsonify(
            {'success': False, 'error': f'Could not complete comparison for {symbol}'}), 404
        swing_formatted = trading_api.format_analysis_response(swing_result, 'Swing')
        position_formatted = trading_api.format_analysis_response(position_result, 'Position')
        result = {'swing_analysis': swing_formatted, 'position_analysis': position_formatted}
        set_cache(cache_key, result)
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in compare/{symbol}: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'An internal server error occurred'}), 500


# --- Error Handlers ---
@app.errorhandler(404)
def not_found(error): return jsonify({'success': False, 'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error): return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)