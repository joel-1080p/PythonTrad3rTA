# PythonTrad3rTA

# Technical Analysis Using Python

## Quick Start

PythonTrad3rTA works with any timeframe (Except VWAP which only works in 5min or lower).<br>
All of the methods take historical data as a pandas dataframe.<br>
For this, it takes 'High', 'Low', 'Open', 'Close', 'Volume' with the first letter capitalized.<br>
<br>
Keep in mind that some methods take the historical data as is, while some take only the close, and 1 takes just the current candle.

<b> INDICATORS </b><br>
RSI <br>
EMA <br>
VWAP <br>
MACD <br>
Super Trend <br>
Stochastic <br>
Bollinger Bands <br>
Volume SMA <br>
SMA (Simple Moving Average) <br>

<b> CANDLE PATTERNS </b><br>
Bullish/Bearish Engulfing <br>
Bullish/Bearish Pin Bars <br>
Tweezer Top/Bottom <br>

<b> CHART PATTERNS </b><br>
Hidden Divergence <br>
EMA Golden/Death Cross <br>

<br>

### Example 1 - How to apply the script.

```python
import PythonTrad3rTA as TA
import yfinance as yf
from datetime import date

# Yahoo Finance API parameters. 
ticker = "AAPL"
start = date(2019, 1, 1)
end = date.today()

# Historical data of the ticker.
hist = yf.download(ticker, start, end)

# Sends only the close and gets back the current RSI value.
RSI = TA.Indicators.RSI(hist['Close'])

# Sends only the close and EMA span. Returns the current 200 EMA value.
EMA_200 = TA.Indicators.EMA(hist['Close'], 200)

# Sends historical data and gets back both stochastic K and stochastic D.
stoch_K, stoch_D = TA.Indicators.stochastic(hist)

# Gets the most current candle.
currentCandle = hist.iloc[0]

# Sends the most current candle and gets back if it is bearish/bullish pin or returns 0 if not.
isPinbar = TA.CandlePatterns.PinBar(currentCandle)

# Sends the most current candle and gets back if it is bearish/bullish engulfing or returns 0 if not.
isEngulfing = TA.CandlePatterns.Engulfing(hist)

# Sends the most current candle and gets back if it is golden/death cross or returns 0 if not.
golden_death_cross = TA.ChartPatterns.EMAGoldenDeathCross(hist)
```

### Requirements

-   [Python](https://www.python.org) \>= 2.7, 3.4+
-   [Pandas](https://github.com/pydata/pandas) \>= 1.3.0
-   [Numpy](http://www.numpy.org) \>= 1.16.5

### P.S.

Any feedback is always welcomed.

**PythonTrad3r**
