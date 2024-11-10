import backtrader as bt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# Fonction pour simuler le système de Lorenz
def lorenz_system(state, t, sigma=18, rho=28, beta=3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Simuler le système de Lorenz
def simulate_lorenz(initial_state, t):
    return odeint(lorenz_system, initial_state, t)

# Paramètres pour la simulation du Lorenz
initial_state = [0.0, 1.0, 1.05]
t = np.linspace(0, 40, 10000)

# Simuler le système de Lorenz
lorenz_data = simulate_lorenz(initial_state, t)
x, y, z = lorenz_data.T

asset = ['SHIB-USD']

# Télécharger les données financières pour backtest
data_1 = yf.download(asset, interval='1m', start="2024-10-11", end="2024-10-16")
data_2 = yf.download(asset, interval='1m', start="2024-10-17", end="2024-10-24")
data_3 = yf.download(asset, interval='1m', start="2024-10-25", end="2024-11-01")
data_4 = yf.download(asset, interval='1m', start="2024-11-02", end="2024-11-09")

data = pd.concat([data_1, data_2, data_3, data_4])

# Calcul des rendements
data['Returns'] = data['Adj Close'].pct_change()

# Intégrer les signaux chaotiques (Lorenz) aux données
data['Lorenz_x'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), x)
data['Lorenz_y'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), y)
data['Lorenz_z'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), z)

# Création des features pour le modèle de machine learning
data['Lag1_Returns'] = data['Returns'].shift(1)
data['Lag2_Returns'] = data['Returns'].shift(2)
data.dropna(inplace=True)

# Diviser en X et y pour l'entraînement
X = data[['Lorenz_x', 'Lorenz_y', 'Lorenz_z', 'Lag1_Returns', 'Lag2_Returns']]
y = np.where(data['Returns'] > 0, 1, 0)

# Entraîner le modèle
train_size = int(0.8 * len(data))
X_train, y_train = X[:train_size], y[:train_size]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Classe de stratégie pour Backtrader
class MLTradingStrategy(bt.Strategy):
    def __init__(self):
        self.data_close = self.datas[0].close
        self.model = model
        self.signals = self.model.predict(X)  # Prédire sur tout le jeu de données
        self.order = None

    def next(self):
        # Index actuel pour accéder aux signaux prédits
        idx = len(self)
        if idx >= len(self.signals):
            return  # S'assurer qu'on ne dépasse pas la taille des signaux prédits

        taille_position = capital_initial / data['Close'].iloc[0]
        
        # Générer des ordres en fonction du signal
        if self.signals[idx] == 1 and not self.position:
            self.order = self.buy(size = taille_position)
        elif self.signals[idx] == 0 and self.position:
            self.order = self.sell(size = self.position.size)

# Initialiser Cerebro et ajouter les données et la stratégie
cerebro = bt.Cerebro()
cerebro.addstrategy(MLTradingStrategy)

# Convertir les données pour Backtrader
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Ajouter un analyseur SharpeRatio
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate = 0.03) # Taux actuel du Livret Jeune

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
sharpe_ratio = results[0].analyzers.sharpe.get_analysis().get('sharperatio')
print(f"Sharpe Ratio : {sharpe_ratio:.2f}")
