import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# --- FIX for joblib loading issue ---
# The joblib/pickle file expects a class named 'SBERTTransformer' to be in the
# __main__ module's namespace. We explicitly import it and add it to sys.modules
# to ensure the model can be loaded correctly.
# This directly resolves the "AttributeError: Can't get attribute 'SBERTTransformer' on <module '__main__'>"
try:
    # This assumes 'position_trading.py' (which is imported by your unified_platform.py)
    # is in the same directory and contains the SBERTTransformer class.
    from position_trading_code import SBERTTransformer

    sys.modules['__main__'].SBERTTransformer = SBERTTransformer
except ImportError:
    # This is a fallback. The application will still run but will use TextBlob
    # for sentiment analysis if the custom SBERTTransformer class can't be found.
    logging.getLogger(__name__).warning(
        "Could not import 'SBERTTransformer' from 'position_trading_code'. "
        "The SBERT model will fail to load, and the system will fall back to TextBlob."
    )
except Exception as e:
    logging.getLogger(__name__).error(f"An unexpected error occurred while importing SBERTTransformer: {e}")

# Import the classes from your existing script
# Make sure 'unified_platform.py' is in the same directory as this 'main.py' file.
from unified_platform import EnhancedSwingTradingSystem, EnhancedPositionTradingSystem

# --- Configuration & Initialization ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="Unified Trading Platform API",
    description="An API for swing and position trading analysis for the Indian stock market.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This is crucial for allowing your frontend application to communicate with this API.
# The "*" allows all origins, which is fine for development.
# For production, you should restrict this to your frontend's domain.
# e.g., origins=["http://localhost:3000", "https://your-frontend-domain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global Instances ---
# Instantiate your trading systems once when the API starts.
# This is efficient as models are loaded only once.
# NOTE: Model paths are hardcoded in your script. Ensure they are correct.
# D:/Python_files/models/sentiment_pipeline.joblib
# D:/best_model_fold_1.pth
try:
    logger.info("Initializing trading systems...")
    swing_system = EnhancedSwingTradingSystem()
    position_system = EnhancedPositionTradingSystem()
    logger.info("Trading systems initialized successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize trading systems. Error: {e}")
    # This will stop the app from starting if models can't be loaded.
    raise


# --- Pydantic Models for Request Bodies ---
# These models define the structure and validation for incoming request data.

class SwingPortfolioRequest(BaseModel):
    total_capital: float = Field(..., gt=1000, description="Total investment budget in INR (e.g., 500000)")
    risk_appetite: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$", description="Risk appetite: LOW, MEDIUM, or HIGH")
    symbols: Optional[List[str]] = Field(None,
                                         description="Optional: A specific list of stock symbols to analyze for the portfolio.")


class PositionPortfolioRequest(BaseModel):
    budget: float = Field(..., gt=10000, description="Total investment budget in INR (e.g., 1000000)")
    risk_appetite: str = Field(..., pattern="^(Low|Medium|High)$", description="Risk appetite: Low, Medium, or High")
    time_period_months: int = Field(..., ge=6, description="Investment time horizon in months (minimum 6)")


# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Welcome endpoint for the API."""
    return {"message": "Welcome to the Unified Trading Platform API. Go to /docs for interactive documentation."}


@app.get("/stocks/swing", tags=["Stocks"])
def get_swing_trading_stocks():
    """Get the list of all stock symbols available for the swing trading system."""
    try:
        return {"symbols": swing_system.get_all_stock_symbols()}
    except Exception as e:
        logger.error(f"Error getting swing stocks: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve swing stock list.")


@app.get("/stocks/position", tags=["Stocks"])
def get_position_trading_stocks():
    """Get the list of all stock symbols available for the position trading system."""
    try:
        return {"symbols": position_system.get_all_stock_symbols()}
    except Exception as e:
        logger.error(f"Error getting position stocks: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve position stock list.")


@app.get("/analyze/swing/{symbol}", tags=["Analysis"])
def analyze_swing_stock(symbol: str):
    """
    Performs a detailed swing trading analysis for a given stock symbol.
    - **symbol**: The stock symbol to analyze (e.g., 'INFY', 'RELIANCE').
    """
    logger.info(f"Received swing analysis request for symbol: {symbol}")
    result = swing_system.analyze_swing_trading_stock(symbol.upper())
    if not result:
        raise HTTPException(status_code=404,
                            detail=f"Could not analyze symbol '{symbol}'. It might be invalid or data is unavailable.")
    return result


@app.get("/analyze/position/{symbol}", tags=["Analysis"])
def analyze_position_stock(symbol: str):
    """
    Performs a detailed position trading analysis for a given stock symbol, including MDA sentiment.
    - **symbol**: The stock symbol to analyze (e.g., 'TCS', 'HDFCBANK').
    """
    logger.info(f"Received position analysis request for symbol: {symbol}")
    result = position_system.analyze_position_trading_stock(symbol.upper())
    if not result:
        raise HTTPException(status_code=404,
                            detail=f"Could not analyze symbol '{symbol}'. It might be invalid or data is unavailable.")
    return result


@app.post("/portfolio/swing", tags=["Portfolio"])
def create_swing_portfolio(request: SwingPortfolioRequest):
    """
    Creates a personalized swing trading portfolio based on budget and risk appetite.
    It analyzes a list of stocks, filters them by risk, and calculates position sizes.
    """
    logger.info(
        f"Received swing portfolio request with capital: {request.total_capital} and risk: {request.risk_appetite}")

    # Determine which symbols to analyze
    if request.symbols:
        symbols_to_analyze = [s.upper() for s in request.symbols]
        logger.info(f"Analyzing a specific list of {len(symbols_to_analyze)} symbols.")
    else:
        symbols_to_analyze = swing_system.get_all_stock_symbols()
        logger.info(f"Analyzing all {len(symbols_to_analyze)} available swing stocks.")

    # Step 1: Analyze stocks
    all_results = swing_system.analyze_multiple_stocks(symbols_to_analyze)
    if not all_results:
        raise HTTPException(status_code=404, detail="No valid analysis results for the given symbols.")

    # Step 2: Filter by risk appetite
    filtered_results = swing_system.filter_stocks_by_risk_appetite(all_results, request.risk_appetite)
    if not filtered_results:
        raise HTTPException(status_code=404,
                            detail=f"No stocks found matching the '{request.risk_appetite}' risk appetite.")

    # Step 3: Generate portfolio allocation (This function prints to console in your script, we'll adapt it to return data)
    # The original function `generate_portfolio_allocation` prints a lot of info.
    # We will call it but also build a JSON response.
    portfolio_data = swing_system.generate_portfolio_allocation(
        filtered_results,
        request.total_capital,
        request.risk_appetite
    )

    if not portfolio_data:
        raise HTTPException(status_code=500, detail="Failed to generate portfolio allocation.")

    # Also get the single best recommendation from the filtered list
    best_recommendation = swing_system.get_single_best_recommendation(filtered_results)

    return {
        "summary": {
            "risk_appetite": request.risk_appetite,
            "total_capital": request.total_capital,
            "suitable_stocks_found": len(filtered_results),
            "portfolio_size": len(portfolio_data)
        },
        "best_recommendation": best_recommendation,
        "portfolio_allocation": portfolio_data
    }


@app.post("/portfolio/position", tags=["Portfolio"])
def create_position_portfolio(request: PositionPortfolioRequest):
    """
    Creates a personalized long-term position trading portfolio based on budget, risk, and time horizon.
    """
    logger.info(
        f"Received position portfolio request with budget: {request.budget} for {request.time_period_months} months.")

    # The function in your class directly handles the analysis and portfolio creation
    portfolio_result = position_system.create_personalized_portfolio(
        risk_appetite=request.risk_appetite,
        time_period_months=request.time_period_months,
        budget=request.budget
    )

    if 'error' in portfolio_result:
        raise HTTPException(status_code=404, detail=portfolio_result['error'])

    return portfolio_result

