from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List
import math
import random

app = FastAPI(title="Stock Market Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERIOD_MAP = {
    "1M": "1mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
}

_sentiment_analyzer = SentimentIntensityAnalyzer()


def safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def format_market_cap(val):
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    yf_period = PERIOD_MAP.get(period.upper(), "1y")
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=yf_period)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker.upper()}")
    hist.index = hist.index.tz_localize(None)
    return hist


@app.get("/api/stock/{ticker}")
async def get_stock_data(ticker: str, period: str = "1Y"):
    try:
        hist = fetch_history(ticker, period)
        data = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_float(row["Open"]),
                "high": safe_float(row["High"]),
                "low": safe_float(row["Low"]),
                "close": safe_float(row["Close"]),
                "volume": int(row["Volume"]) if not math.isnan(float(row["Volume"])) else 0,
            }
            for date, row in hist.iterrows()
        ]
        return {"ticker": ticker.upper(), "period": period, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/indicators")
async def get_indicators(ticker: str, period: str = "1Y"):
    try:
        hist = fetch_history(ticker, period)
        close = hist["Close"]

        hist["SMA20"] = close.rolling(20).mean()
        hist["SMA50"] = close.rolling(50).mean()
        hist["SMA200"] = close.rolling(200).mean()
        hist["RSI"] = calculate_rsi(close)
        hist["Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

        result = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": safe_float(row["Close"]),
                "sma20": safe_float(row["SMA20"]),
                "sma50": safe_float(row["SMA50"]),
                "sma200": safe_float(row["SMA200"]),
                "rsi": safe_float(row["RSI"]),
                "volatility": safe_float(row["Volatility"]),
            }
            for date, row in hist.iterrows()
        ]
        return {"ticker": ticker.upper(), "indicators": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/info")
async def get_stock_info(ticker: str):
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        hist = stock.history(period="5d")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker.upper()} not found")

        def get_val(key, default=None):
            v = info.get(key, default)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return default
            return v

        market_cap = get_val("marketCap")

        return {
            "ticker": ticker.upper(),
            "name": get_val("longName") or get_val("shortName", ticker.upper()),
            "sector": get_val("sector", "N/A"),
            "industry": get_val("industry", "N/A"),
            "country": get_val("country", "N/A"),
            "currency": get_val("currency", "USD"),
            "pe_ratio": safe_float(get_val("trailingPE")),
            "forward_pe": safe_float(get_val("forwardPE")),
            "market_cap": market_cap,
            "market_cap_formatted": format_market_cap(market_cap),
            "dividend_yield": safe_float(get_val("dividendYield")),
            "week_52_high": safe_float(get_val("fiftyTwoWeekHigh")),
            "week_52_low": safe_float(get_val("fiftyTwoWeekLow")),
            "avg_volume": get_val("averageVolume"),
            "beta": safe_float(get_val("beta")),
            "description": (get_val("longBusinessSummary") or "")[:600],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/risk")
async def get_risk_analysis(ticker: str, period: str = "1Y"):
    try:
        hist = fetch_history(ticker, period)
        close = hist["Close"]
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            raise HTTPException(status_code=400, detail="Not enough data for risk analysis. Use 6M or 1Y period.")

        risk_free_daily = 0.04 / 252
        excess = returns - risk_free_daily
        sharpe = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() != 0 else 0.0

        var_95 = float(np.percentile(returns, 5)) * 100
        var_99 = float(np.percentile(returns, 1)) * 100

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) * 100

        annual_return = float(returns.mean() * 252) * 100
        annual_vol = float(returns.std() * np.sqrt(252)) * 100

        current_price = float(close.iloc[-1])
        mu = float(returns.mean())
        sigma = float(returns.std())
        n_sims = 200
        n_days = 30

        np.random.seed(42)
        sim_matrix = np.zeros((n_sims, n_days))
        for i in range(n_sims):
            daily = np.random.normal(mu, sigma, n_days)
            sim_matrix[i] = current_price * np.cumprod(1 + daily)

        sample_indices = random.sample(range(n_sims), 15)

        return {
            "ticker": ticker.upper(),
            "sharpe_ratio": round(sharpe, 3),
            "var_95": round(var_95, 3),
            "var_99": round(var_99, 3),
            "max_drawdown": round(max_drawdown, 3),
            "annual_return": round(annual_return, 3),
            "annual_volatility": round(annual_vol, 3),
            "current_price": round(current_price, 2),
            "monte_carlo": {
                "n_simulations": n_sims,
                "n_days": n_days,
                "percentiles": {
                    "p5": np.percentile(sim_matrix, 5, axis=0).round(2).tolist(),
                    "p25": np.percentile(sim_matrix, 25, axis=0).round(2).tolist(),
                    "p50": np.percentile(sim_matrix, 50, axis=0).round(2).tolist(),
                    "p75": np.percentile(sim_matrix, 75, axis=0).round(2).tolist(),
                    "p95": np.percentile(sim_matrix, 95, axis=0).round(2).tolist(),
                },
                "sample_paths": sim_matrix[sample_indices].round(2).tolist(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/anomalies")
async def get_anomalies(ticker: str, period: str = "1Y"):
    try:
        hist = fetch_history(ticker, period)

        if len(hist) < 30:
            raise HTTPException(status_code=400, detail="Not enough data for anomaly detection. Use 6M or 1Y period.")

        close = hist["Close"]
        returns = close.pct_change().fillna(0)

        features = pd.DataFrame({
            "returns": returns,
            "volatility": returns.rolling(5).std().fillna(0),
            "volume_change": hist["Volume"].pct_change().fillna(0).clip(-5, 5),
            "price_range": ((hist["High"] - hist["Low"]) / close.replace(0, np.nan)).fillna(0),
        })

        clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        predictions = clf.fit_predict(features.values)
        scores = clf.decision_function(features.values)

        data = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "is_anomaly": bool(predictions[i] == -1),
                "anomaly_score": round(float(scores[i]), 4),
            }
            for i, (date, row) in enumerate(hist.iterrows())
        ]

        return {
            "ticker": ticker.upper(),
            "data": data,
            "anomaly_count": int((predictions == -1).sum()),
            "total_points": len(data),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stock/{ticker}/sentiment")
async def get_sentiment(ticker: str):
    try:
        stock = yf.Ticker(ticker.upper())
        try:
            news_items = stock.news or []
        except Exception:
            news_items = []

        articles = []
        scores = []

        for item in news_items[:20]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            if not title:
                continue

            sentiment = _sentiment_analyzer.polarity_scores(title)
            compound = sentiment["compound"]
            scores.append(compound)

            label = "Bullish" if compound >= 0.05 else ("Bearish" if compound <= -0.05 else "Neutral")

            publisher = (
                content.get("provider", {}).get("displayName")
                or item.get("publisher", "")
            )
            link = (
                content.get("canonicalUrl", {}).get("url")
                or item.get("link", "")
            )
            pub_date = content.get("pubDate") or str(item.get("providerPublishTime", ""))

            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": pub_date,
                "sentiment_score": round(compound, 3),
                "sentiment_label": label,
            })

        if not scores:
            return {
                "ticker": ticker.upper(),
                "articles": [],
                "overall_sentiment": 0.0,
                "overall_label": "Neutral",
                "message": "No recent news available",
            }

        overall = round(sum(scores) / len(scores), 3)
        overall_label = "Bullish" if overall >= 0.05 else ("Bearish" if overall <= -0.05 else "Neutral")

        return {
            "ticker": ticker.upper(),
            "articles": articles[:10],
            "overall_sentiment": overall,
            "overall_label": overall_label,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class CorrelationRequest(BaseModel):
    tickers: List[str]
    period: str = "1Y"


@app.post("/api/correlation")
async def get_correlation(request: CorrelationRequest):
    if len(request.tickers) < 2:
        raise HTTPException(status_code=400, detail="At least 2 tickers required")
    if len(request.tickers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers allowed")

    yf_period = PERIOD_MAP.get(request.period.upper(), "1y")

    try:
        price_data = {}
        for ticker in request.tickers:
            stock = yf.Ticker(ticker.upper())
            hist = stock.history(period=yf_period)
            if not hist.empty:
                hist.index = hist.index.tz_localize(None)
                price_data[ticker.upper()] = hist["Close"]

        if len(price_data) < 2:
            raise HTTPException(status_code=400, detail="Could not fetch data for enough tickers")

        df = pd.DataFrame(price_data)
        corr = df.pct_change().dropna().corr()
        tickers = list(corr.columns)

        return {
            "tickers": tickers,
            "matrix": [[round(float(corr.loc[t1, t2]), 4) for t2 in tickers] for t1 in tickers],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
