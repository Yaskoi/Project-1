import backtrader as bt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# Télécharger les données financières pour backtest

asset = ['SHIB-USD']

data_1 = yf.download(asset, interval='1m', start="2024-10-12", end="2024-10-16")
data_2 = yf.download(asset, interval='1m', start="2024-10-17", end="2024-10-24")
data_3 = yf.download(asset, interval='1m', start="2024-10-25", end="2024-11-01")
data_4 = yf.download(asset, interval='1m', start="2024-11-02", end="2024-11-09")

data = pd.concat([data_1, data_2, data_3, data_4])


data['Returns'] = data['Adj Close'].pct_change()
data.dropna(inplace=True)

# Ajouter le signal Mackey-Glass
def mackey_glass(tau=25):
    x = [0.5]
    for t in range(1, len(data)):
        x_t = x[-1] + 0.2 * x[-tau] / (1 + x[-tau]**15) - 0.15 * x[-1] if t >= tau else 0.5
        x.append(x_t)
    return np.array(x)

data['Mackey_Glass'] = mackey_glass()

# Créer des lags pour le modèle
data['Lag1_Returns'] = data['Returns'].shift(1)
data['Lag2_Returns'] = data['Returns'].shift(2)
data.dropna(inplace=True)

# Diviser en X et y pour l'entraînement
X = data[['Mackey_Glass', 'Lag1_Returns', 'Lag2_Returns']]
y = np.where(data['Returns'] > 0, 1, 0)

# Entraîner le modèle
train_size = int(0.8 * len(data))
X_train, y_train = X[:train_size], y[:train_size]
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Classe de stratégie pour Backtrader
class MLTradingStrategy(bt.Strategy):
    def __init__(self):
        self.model = model
        self.signals = self.model.predict(X)

    def next(self):
        # Utiliser l'index actuel de Backtrader pour éviter les dépassements
        idx = len(self)
        if idx >= len(self.signals):  # Vérifier que l'index est dans la portée
            return  # Sortir si on dépasse la longueur des signaux
        
        taille_position = capital_initial / data['Adj Close'].iloc[0]

        if self.signals[idx] == 1 and not self.position:
            self.buy(size = taille_position)
        elif self.signals[idx] == 0 and self.position:
            self.sell(size = self.position.size)

# Initialiser Cerebro et ajouter les données et la stratégie
cerebro = bt.Cerebro()
cerebro.addstrategy(MLTradingStrategy)

# Convertir les données pour Backtrader
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Ajouter l'analyseur SharpeRatio
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.045)

# Définir le capital de départ
capital_initial = 100
cerebro.broker.setcash(capital_initial)

# Exécuter le backtest
results = cerebro.run()
cerebro.plot()

# Obtenir la valeur finale du portefeuille et le rendement
valeur_finale = cerebro.broker.getvalue()
rendement = (valeur_finale - capital_initial) / capital_initial * 100

# Afficher la valeur finale et le rendement
print(f"Valeur finale du portefeuille : {valeur_finale:.2f} USD")
print(f"Rendement total : {rendement:.2f} %")

# Extraire et afficher le Sharpe Ratio
sharpe_ratio = results[0].analyzers.sharpe.get_analysis()['sharperatio']
print(f"Sharpe Ratio : {sharpe_ratio:.2f}")
