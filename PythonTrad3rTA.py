import string
import pandas as pd
import numpy as np
import os
import time
import requests
import datetime
import operator


################################
################################

#### RESOURCES ####
## Gets S&P 500 tickers

#### INDICATORS ####
## RSI 
## EMA
## VWAP
## MACD
## Super Trend
## Stochastic
## Bollinger Bands
## Volume SMA
## SMA (Simple Moving Average)

#### CANDLE PATTERNS ####
## Bullish/Bearish Engulfing
## Bullish/Bearish Pin Bars
## Tweezer Top/Bottom

#### CHART PATTERNS ####
## Hidden Divergence
## EMA Golden/Death Cross

################################
################################
class Indicators:
    def __init__(self):
        pass
    
    ################################
    ################################
    ## Returns S&P 500 tickers from Tickers.txt along with their exchange.
    def getTickers() -> list and list:

        # Holds all the tickers.
        tickers = []

        # Holds all the exchanges.
        exchange = []

        # Path to Tickers.txt.
        tickersPath = os.path.expanduser('~/Documents/PythonTrad3rFiles/Tickers.txt')

        # Gets the list from S&P 500 Wikipedia page [DOES NOT INCLUDE EXCHANGE].
        response = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

        # Checks to see if the local list is up to date from the web list.
        # If not, it will clear out the list, and recreate it with new list.
        # If so, it will read from the local list.
        if Indicators.checkForIndexChange(response[1], tickersPath):

            print('Rebuilding ticker list.')

            # Removes everything from ticker file.
            open(tickersPath, "w").close()

            # Gets all the tickers and appends them to the list.
            for i in response[0]['Symbol']:
                
                # Holds the response from TDA.
                instrumentResponse = None

                # Sends request to TDA.
                while instrumentResponse is None:
                    try:
                        instrumentResponse = requests.get('https://api.tdameritrade.com/v1/instruments?apikey=FEGS4YNPGMCNOOHJLUT6FECMQ2JUGRBV&symbol='+i+'&projection=symbol-search')
                    except:
                        pass

                # Casts the response to JSON.
                tickerInfo = instrumentResponse.json()

                # Appends the exchange to the list.
                exchange.append(tickerInfo[i]['exchange'])

                # Appends the ticker to the list.
                tickers.append(i)

                # Sleeps for .5 seconds to stay within the TDA's 120 requests per minute.
                time.sleep(0.5)

            # Opens the Tickers.txt file and writes the updated S&P 500 list.
            # Format - TICKER1:EXCHANGE1,TICKER2:EXCHANGE3,....
            with open(tickersPath, "w") as f:
                for i in range(len(tickers)):
                    tempSTR =  exchange[i] + ':' +  tickers[i] + ','
                    f.write(tempSTR)
                f.write('index')

        # If the list has not been changed, it will read from the local one.
        else:

            # Opens the file and uses f as the file variable.
            with open(tickersPath) as f:
                
                # holds the individual characters from the text file.
                inputChar = ''
                
                # Holds the ticker or exchange.
                inputStr = ''
                
                # Parses the text file.
                while True:
                    
                    # Gets individual character from file.
                    inputChar = f.read(1)
                
                    # If ':' then it will append full string to exchange array and resets the string.
                    if inputChar == ':':
                        exchange.append(inputStr)
                        inputStr = ''
                    
                    # If ',' then it will append fill string to the tickers array and reset the string.
                    elif inputChar ==',':
                        tickers.append(inputStr)
                        inputStr = ''
                        
                    # If it reaches the end of the file.
                    elif not inputChar:
                        break
                    
                    # Combines the characters into one string if no other parameter has been met.
                    else:
                        inputStr+=inputChar

        # Returns the ticker and exchange lists.
        return tickers, exchange


    ################################
    ################################
    # Checks to see if there has been any changes to the S&P 500 Wikipedia page.
    def checkForIndexChange(changesInSNP: list, tickersPath: string) -> int:

        # Checks to see if path to ticker file exsist, if it does not, it will create the directory path.
        if not os.path.exists(os.path.expanduser('~/Documents/PythonTrad3rFiles')):
            os.makedirs(os.path.expanduser('~/Documents/PythonTrad3rFiles'))

        # Checks to see if the word "index" is in the file and if the file has anything in it.
        # If the file does not exsist, it will create it and append to it.
        # If the file does exist, it will check to see if the word "index" is in it.
        # If index is not in it and it has content, it is a custom file.
        # If index is in it, it will continue.
        with open(tickersPath,  "a+") as f:
            if "index" not in f.read() and os.stat(tickersPath).st_size > 0:
                print('Custom List')
                # Returns 0 as it is a custom list that is empty.
                return 0

        # Holds weather or not there has been a change to the list.
        listChanged = True

        # Extracts the added tickers from the response.
        tickerChanges = changesInSNP['Added'].head()

        # Chooses the latest ticker added.
        latestTickerAdded = tickerChanges['Ticker'][0]

        # Opens file and searches the current ticker list for this new addition.
        with open(tickersPath, "r") as f:
            for line in f:
            
                # If the newest ticker is in the list.
                # This indicates the list has not been changed.
                if latestTickerAdded in line:
                    listChanged = False
                    break 

        # If the list has not been changed.
        if not listChanged:
            return 0

        # If the list has been changed.
        return 1
    

    ################################
    ################################
    # Calculates RSI 14 using historical data as a pandas data frame.
    # Returns RSI as a float.
    def RSI(historicalData: pd) -> float:

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()

        # Calculations
        historicalDataCopy['delta'] = delta = historicalDataCopy['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down
        historicalDataCopy['RSI'] = 100-(100/(1+rs))
        
        # Gets the latest RSI value.
        rsi = float(historicalDataCopy['RSI'].tail(1))

        # Rounds RSI to 2 decimal places.
        rsi = round(rsi, 2)

        # Returns RSI.
        return rsi


    ################################
    ################################
    # Calculates EMA based on the days sent.
    # It also uses historical closing data as a pandas data frame.
    # Returns EMA as a float.
    def EMA(historicalCloses: pd, span: int) -> float:
        # algo credit -https://python.plainenglish.io/how-to-calculate-the-ema-of-a-stock-with-python-5d6fb3a29f5

        # Holds the smoothing value.
        smoothing = 2
        
        # Calculations.
        ema = [sum(historicalCloses[:span])/span]
        for historicalCloses in historicalCloses[span:]:
            ema.append(round((historicalCloses*(smoothing/(1+span)))+ema[-1]*(1-(smoothing/(1+span))),2))
        
        # Rounds EMA to 2 decimal places.
        emaFinal = round(ema[-1], 2)

        # Return the latest value of the EMA.
        return emaFinal


    ################################
    ################################
    # Calculates VWAP using historical data as a pandas data frame and returns value as a float.
    # Also takes in the time frame (in minutes) so it can apply it to the trading session.
    # Algo has to calculate based on how many candles have elapsed in this time frame since VWAP is session based.
    def VWAP(historicalData: pd, timeFrameInMinutes: int) -> float:

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()

        # Gets the current date and time.
        now = datetime.datetime.now()

        # Gets the time market opened. 
        marketOpen = now.replace(hour=9, minute=30)

        # How much time has passed since market open.
        timeElapsed = now - marketOpen

        # Casts the time elapsed to minutes.
        timeElapsedInMinutes = int(timeElapsed.total_seconds() / 60)

        # Based on the time frame, it calculates how many of those candles have elapsed since market open.
        numberOfCandlesToUse = int(round(timeElapsedInMinutes/timeFrameInMinutes, 0) + 1)

        # Extracts only the data needed for the VWAP.
        extractedData = historicalDataCopy[-numberOfCandlesToUse:]

        # Calculations.
        v = extractedData['Volume'].values
        tp = (extractedData['Low'] + extractedData['Close'] + extractedData['High']).div(3).values
        data = extractedData.assign(vwap=(tp * v).cumsum() / v.cumsum())

        # Gets the last VWAP data and rounds.
        vwap = round(data['vwap'][-1], 2)

        # Returns VWAP.
        return vwap

    
    ################################
    ################################
    # Calculates the MACD, MACD Histogram, and MACD Signal.
    # Takes in historical data as pandas data frame and returns MACD, Histogram, and Signal as floats.
    def MACD(historicalData: pd) -> float and float and float:

        # Algo credit - https://medium.com/codex/algorithmic-trading-with-macd-in-python-1c2769a6ad1b

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()

        # VARIABLES.
        slow = 26
        fast =  12
        smooth = 9

        # Calculations.
        exp1 = historicalDataCopy['Close'].ewm(span = fast, adjust = False).mean()
        exp2 = historicalDataCopy['Close'].ewm(span = slow, adjust = False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
        signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
        hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
        frames =  [macd, signal, hist]
        historicalDataCopy = pd.concat(frames, join = 'inner', axis = 1)

        # Gets the last data and rounds.
        final_signal = round(historicalDataCopy['signal'][-1], 2)
        final_hist = round(historicalDataCopy['hist'][-1], 2)
        final_macd = round(historicalDataCopy['macd'][-1], 2)

        # Returns the histogram, MACD, and signal.
        return final_hist, final_macd, final_signal


    ################################
    ################################
    # Calculates SuperTrend using historical data as a pandas data frame.
    # Returns the signal as a string (Buy or Sell) and the value as a float.
    def SuperTrend(historicalData: pd)-> string and float:
        # Algo credit - https://medium.datadriveninvestor.com/the-supertrend-implementing-screening-backtesting-in-python-70e8f88f383d

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()

        # Holds the signal calculated by the algo.
        signal = ''

        # Holds the value calculated by the algo.
        ST_value = 0.0

        # ATR period.
        atr_period = 10
        
        # ATR Multiplier.
        multiplier = 3.0
        
        # Historic highs.
        high = historicalDataCopy['High']
        
        # Historic lows.
        low = historicalDataCopy['Low']
        
        # Historic Closes.
        close = historicalDataCopy['Close']
        
        # calculates ATR.
        price_diffs = [high - low, 
                    high - close.shift(), 
                    close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        
        # default ATR calculation in super trend indicator
        atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
        
        # HL2 is simply the average of high and low prices
        hl2 = (high + low) / 2
        
        # Upper band and lower band calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (multiplier * atr)
        final_lowerband = lowerband = hl2 - (multiplier * atr)
        
        # initialize Super trend column to True
        supertrend = [True] * len(historicalDataCopy)
        
        for i in range(1, len(historicalDataCopy.index)):
            curr, prev = i, i-1
            
            # if current close price crosses above upper band
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lower band
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]
                
                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan
        
        
        supertrendFinal = pd.DataFrame({
            'Supertrend': supertrend,
            'Final Lowerband': final_lowerband,
            'Final Upperband': final_upperband
        }, index=historicalData.index)
        
        # joins the values to the pandas data frame
        historicalDataCopy = historicalDataCopy.join(supertrendFinal)
        
        # Checks if The upper/lower band has NaN as the value.
        # If not, it will assign the direction
        if not historicalDataCopy['Final Lowerband'].tail(1).isnull().values.any():
            signal = 'Buy'
            ST_value = round(historicalDataCopy['Final Lowerband'].tail(1)[-1], 2)
        elif not historicalDataCopy['Final Upperband'].tail(1).isnull().values.any():
            signal = 'Sell'
            ST_value = round(historicalDataCopy['Final Upperband'].tail(1)[-1], 2)
        
        return signal, ST_value


    ################################
    ################################
    # Calculates Stochastic using historical data as a pandas data frame.
    # Returns Stochastic K and Stochastic D as floats.
    def stochastic(historicalData: pd) -> float and float:
        # Algo credit - https://www.alpharithms.com/stochastic-oscillator-in-python-483214/

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()
    
        # %K period
        k_period = 14
        
        # %D Period.
        d_period = 3
        
        # Adds a "n_high" column with max value of previous 14 periods.
        historicalDataCopy['n_high'] = historicalDataCopy['High'].rolling(k_period).max()
        
        # Adds an "n_low" column with min value of previous 14 periods.
        historicalDataCopy['n_low'] = historicalDataCopy['Low'].rolling(k_period).min()
        
        # Uses the min/max values to calculate the %k (as a percentage).
        historicalDataCopy['%K'] = (historicalDataCopy['Close'] - historicalDataCopy['n_low']) * 100 / (historicalDataCopy['n_high'] - historicalDataCopy['n_low'])
        
        # Uses the %k to calculates a SMA over the past 3 values of %k.
        historicalDataCopy['%K'] = historicalDataCopy['%K'].rolling(d_period).mean()
        
        # K Smoothing .
        historicalDataCopy['%D'] = historicalDataCopy['%K'].rolling(d_period).mean()
        
        # Rounds stochastic D to 2 decimal places.
        stochasticD = round(historicalDataCopy['%D'].tail(1).values[0],2)

        # Rounds stochastic K to 2 decimal places.
        stochasticK = round(historicalDataCopy['%K'].tail(1).values[0], 2)

        # Returns D and K values.
        return stochasticK, stochasticD


    ################################
    ################################
    # Calculates Bollinger Bands and returns upper and lower bounds as floats.
    # Takes in historical data as pandas data frame and the span as an int.
    def bollingerBands(historicalData: pd, span: int) -> float and float:

        # Algo credit - https://medium.com/codex/how-to-calculate-bollinger-bands-of-a-stock-with-python-f9f7d1184fc3

        # Calculates simple moving average.
        sma = historicalData['Close'].rolling(span).mean()

        # Calculations.
        std = historicalData['Close'].rolling(span).std()
        bollinger_up = sma + std * 2 # Calculate top band
        bollinger_down = sma - std * 2 # Calculate bottom band

        # Gets the last value in the list and rounds.
        upperBB = round(bollinger_up[-1], 2)
        lowerBB = round(bollinger_down[-1], 2)

        # Returns upper and lower bands.
        return upperBB, lowerBB


    ################################
    ################################
    # Calculates average volume based on the days sent.
    # It also uses historical data as a pandas data frame.
    # Returns volume average as a float.
    def volumeSMA(historicalData: pd, span: int) -> float:

        # Keeps tracking of running total. Starts with today's volume for the calculations.
        runningTotal = historicalData['Volume'][-1]

        # Array that holds volume from JSON as an array.
        volumeArray = []
        
        # Assigns array with all the volume data for the last year.    
        volumeArray = []
        for vol in historicalData['Volume']:
            volumeArray.append(vol)

        # Reveres array to have the most current as 0.
        volumeArray.reverse()

        # Creates a sum of volume for the last 9 days. Skips today since it is already accounted for.
        i = 1
        while i < span:
            
            # Running total
            runningTotal += volumeArray[i]
            
            # Increments index
            i += 1
        
        # Calculates 9 day average.
        averageVolume = round(runningTotal/span, 2)
            
        return averageVolume


    ################################
    ################################
    # Calculates simple moving average.
    # Takes in Historical Data pandas data frame and span by integer and returns MA as float.
    def SMA(historicalData: pd, span: int) -> float:

        long_rolling = historicalData['Close'].rolling(window=span).mean()

        sma = round(long_rolling[-1], 2)
        return sma


################################
################################
################################
################################
################################
################################
class CandlePatterns:
    def __init__(self) -> None:
        pass

    ################################
    ################################
    # Calculates if the latest candle is a bullish or bearish pin bar using current candle data as a pandas data frame.
    # Returns string indicating 'Bullish' or 'Bearish'.
    # Returns 0 if it is neither.
    # Input example - hist.iloc[-1]
    def PinBar(currentCandle: pd) -> string:

        # Holds the high, low, open, and close for the latest candle.
        currentCandle_H = currentCandle['High']
        currentCandle_L = currentCandle['Low']
        currentCandle_O = currentCandle['Open']
        currentCandle_C = currentCandle['Close']
        
        # Looks for a green candle.
        if currentCandle_C > currentCandle_O:

            # Gets top and bottom wick sizes.
            topWick = currentCandle_H - currentCandle_C
            bottomWick = currentCandle_O - currentCandle_L
            
            # True if the bottom wick is more than 50% of the candle size.
            #if bottomWick > (candleSize * .75):
            if bottomWick > (topWick * 2):
                return "Bullish"
                
        # Looks for a red candle.
        if currentCandle_C < currentCandle_O:

            # Gets top and bottom wick sizes.
            topWick = currentCandle_H - currentCandle_O
            bottomWick = currentCandle_C - currentCandle_L
            
            # Checks to see if the top wick is double the size of the bottom wick.
            #if topWick > (candleSize * .75):
            if topWick > (bottomWick * 2):
                return "Bearish"
                
        # Neither bullish or bearish pin bars.
        return 0


    ################################
    ################################
    # Calculates if the latest candle is a bullish or bearish engulfing using historical data as a pandas data frame.
    # Returns string indicating 'Bullish' or 'Bearish'.
    # Returns 0 if it is neither.
    def Engulfing(historicalData: pd) -> string:

        # Holds the high, low, open, and close for the previous candle.
        previousCandle_H = historicalData['High'][-2]
        previousCandle_L = historicalData['Low'][-2]

        # Holds the high, low, open, and close for the current candle.
        currentCandle_O = historicalData['Open'][-1]
        currentCandle_C = historicalData['Close'][-1]

        # Checks for bearish engulfing.
        if currentCandle_C > previousCandle_H and currentCandle_O < previousCandle_L:
            return "Bullish"
        
        # Checks for bullish engulfing.
        if currentCandle_O > previousCandle_H and currentCandle_C < previousCandle_L:
            return "Bearish"
        
        # It is neither bearish or bullish engulfing.
        return 0


    ################################
    ################################
    # Checks to see if the latest two candles makes a tweezer top or bottom.
    # Checks for two pin bars and determine if it is a tweezer top or bottom.
    # Takes in Historical data as pandas and returns direction as a string.
    # Returns 0 if it's nothing.
    def Tweezer(historicalData: pd) -> string:

        # Checks to see if current candle is a bullish or bearish pin.
        # Returns 0 if neither.
        currentCandleDirection = CandlePatterns.PinBar(historicalData.iloc[-1])

        # Holds the open, close, high, and low of the current candle.
        currentCandleData = historicalData.iloc[-1]

        # Holds the high, low, open, close.
        previousCandle_H = historicalData['High'][-2]
        previousCandle_L = historicalData['Low'][-2]
        previousCandle_O = historicalData['Open'][-2]
        previousCandle_C = historicalData['Close'][-2]

        # If the second candle is green and the first candle is red.
        if currentCandleDirection == 'Bearish' and previousCandle_O < previousCandle_C:

            # If the current candle stayed bellow the high of the previous candle.
            # This prevents gap ups and gaps not closed.
            if currentCandleData['Close'] < previousCandle_H:

                # Gets the top and bottom wicks.
                topWick = previousCandle_H - previousCandle_C
                bottomWick = previousCandle_O - previousCandle_L
            
                # Looks for a bearish green pin.
                if topWick > (bottomWick * 2):
                    return 'Bearish'

        # If the second candle is red and the first candle is green.
        if currentCandleDirection == 'Bullish' and previousCandle_O > previousCandle_C:

            # If the current candle stayed above the low of the previous candle.
            # This prevents gap downs and gaps not closed.
            if currentCandleData['Close'] > previousCandle_L:

                # Gets top and bottom wick sizes.
                topWick = previousCandle_H - previousCandle_O
                bottomWick = previousCandle_C - previousCandle_L

                # Looks for a bullish red pin.
                if bottomWick > (bottomWick * 2):
                    return 'Bullish'

        return 0


################################
################################
################################
################################
################################
################################
class ChartPatterns:
    def __init__(self) -> None:

        # Declare private functions.
        self.__calculate_RSI_history()
        self.__traverseRSI()
        self.__calculateDiv()
        
        pass

    ##################################
    ##################################
    # Calculates hidden divergence.
    # Uses RSI data and compares it with historical data.
    # Once divergence is detected it, it also checks stochastic cross. 
    def HiddenDivergence(historicalData: pd) -> string:

        # Calculates historical RSI data and returns RSI history.
        RSIData = ChartPatterns.__calculate_RSI_history(historicalData)

        # If the latest RSI value is less than 35.
        if float(RSIData[0]) < 35:

            # Looks for RSI higher or lower than the current RSI.
            # Returns index to check if price action agrees.
            # Sends operator less than to indicate direction
            indexRSI = ChartPatterns.__traverseRSI(RSIData, operator.lt)

            if ChartPatterns.__calculateDiv(indexRSI, historicalData['Low'], operator.gt):

                stochK, stochD = Indicators.stochastic(historicalData)

                if stochD < stochK:
                    return 'Bullish'

        elif float(RSIData[0]) > 65:

            # Looks for RSI higher or lower than the current RSI.
            # Returns index to check if price action agrees.
            # Sends operator greater than to indicate direction
            indexRSI = ChartPatterns.__traverseRSI(RSIData, operator.gt)

            if ChartPatterns.__calculateDiv(indexRSI, historicalData['High'], operator.lt):

                stochK, stochD = Indicators.stochastic(historicalData)
                
                if stochD > stochK:
                    return 'Bearish'
        return 0


    ##################################
    ##################################
    # Calculates the RSI using historical data.
    # Returns array with the data.
    def __calculate_RSI_history(historicalData: pd):
        
        # Algo credit https://www.learnpythonwithrune.org/pandas-calculate-the-relative-strength-index-rsi-on-a-stock/

        # Creates copy of historical data to be used and modified for this function.
        historicalDataCopy = historicalData.copy()
        
        # Holds the RSI data.
        rsiData = []

        # Calculations
        historicalDataCopy['delta'] = delta = historicalDataCopy['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up/ema_down
        historicalDataCopy['RSI'] = 100-(100/(1+rs))

        # Casts the data from pandas to a list.
        for val in historicalDataCopy['RSI']:
            rsiData.append(val)

        # Reverses the list.
        rsiData.reverse()

        # Returns RSI data as a list.
        return rsiData


    ##################################
    ##################################
    # Looks for RSI higher or lower than the current RSI.
    # Returns index to check if price action agrees.
    def __traverseRSI(RSIData, conditionalOperator) -> int:

        # Holds the current RSI value.
        currentRSI = RSIData[0]

        # This will skip n amounts of days for the RSI.
        RSIbuffer = 5

        # Traverses the RSI value looking for a value higher/lower than the current value.
        for i, val in enumerate(RSIData):

            # Skips buffer values.
            if i > RSIbuffer:

                # If the direction is bullish and value is lower than the current RSI value.
                # If the direction is not bullish and value is higher than the current RSI value.
                if conditionalOperator(val, currentRSI):
                    return i

            # Before the buffer days.
            # Looks to see if the current RSI value is higher/lower than the next 4.
            else:
                if conditionalOperator(val, currentRSI):
                    return 0

        return 0


    ##################################
    ##################################
    # Now that it has the correct 'i'th position in the RSI,
    # It will locate that position in the candle and compair it to RSI data.
    def __calculateDiv(indexRSI, historicalData, conditionalOperator) -> bool:
        if indexRSI:
                historicalCloses = []

                # Casts the data from pandas to a list.
                for val in historicalData:
                    historicalCloses.append(val)
                
                historicalCloses.reverse()

                if conditionalOperator(historicalCloses[indexRSI], historicalCloses[0]):
                    return 1
        return 0


    ##################################
    ##################################
    # Calculates golden/death cross of the 50 and 200 EMA.
    # Returns string if it is Bullish or Bearish.
    # If it is neither, it returns 0.
    def EMAGoldenDeathCross(hist: pd) -> string:

        # Current day 20 and 200 EMA.
        today_ema_50 = Indicators.EMA(hist['Close'], 50)
        today_ema_200 = Indicators.EMA(hist['Close'], 200)

        # Makes a copy of the history
        # It is missing the latest day.
        histOneless = hist[:-1]

        # Holds yesterday's EMA.
        yesterday_ema_50 = Indicators.EMA(histOneless['Close'], 50)
        yesterday_ema_200 = Indicators.EMA(histOneless['Close'], 200)

        # Checks for Golden and Death Cross and returns them as a string.
        if yesterday_ema_50 > yesterday_ema_200 and today_ema_50 < today_ema_200:
            return "Death Cross"
        elif yesterday_ema_50 < yesterday_ema_200 and today_ema_50 > today_ema_200:
            return "Golden Cross"

        # Returns 0 if it could not find any crosses.
        return 0
