import ccxt
import pandas as pd
import time

api_key = ''
api_secret = ''

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})


symbol = 'BTC/USDT'  
timeframe = '1h'
ma_period = 20 


def get_data(symbol, timeframe, limit=100):
    """Récupère les données de marché."""
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_ma(data, period):
    """Calcule la moyenne mobile sur une période donnée."""
    return data['close'].rolling(window=period).mean()


def place_order(symbol, side, amount):
    """Place un ordre sur Binance."""
    order = binance.create_order(symbol, 'market', side, amount)
    print(f"Ordre {side} passé : {order}")
    return order


def run_bot():
    position = None 

    while True:
        data = get_data(symbol, timeframe)

        data['ma'] = calculate_ma(data, ma_period)

        last_close = data['close'].iloc[-1]  
        last_ma = data['ma'].iloc[-1]  

        if last_close > last_ma and position != 'buy':
            print(f"Achat signalé - Prix actuel : {last_close}, MA : {last_ma}")
            place_order(symbol, 'buy', 0.001) 
            position = 'buy'

        elif last_close < last_ma and position == 'buy':
            # Vendre si le prix tombe en dessous de la moyenne mobile
            print(f"Vente signalée - Prix actuel : {last_close}, MA : {last_ma}")
            place_order(symbol, 'sell', 0.001)
            position = 'sell'

        time.sleep(30)


if __name__ == '__main__':
    run_bot()
