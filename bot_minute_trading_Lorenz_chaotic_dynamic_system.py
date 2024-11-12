import backtrader as bt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
data_1 = yf.download(asset, interval='1m', start="2024-10-14", end="2024-10-19")
data_2 = yf.download(asset, interval='1m', start="2024-10-20", end="2024-10-27")
data_3 = yf.download(asset, interval='1m', start="2024-10-28", end="2024-11-04")
data_4 = yf.download(asset, interval='1m', start="2024-11-05", end="2024-11-12")

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
            self.order = self.buy(size = taille_position)  # Acheter avec tout le capital disponible
        elif self.signals[idx] == 0 and self.position:
            self.order = self.sell(size= self.position.size)  # Vendre toute la position

# Initialiser Cerebro et ajouter les données et la stratégie
cerebro = bt.Cerebro()
cerebro.addstrategy(MLTradingStrategy)

# Convertir les données pour Backtrader
data_feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_feed)

# Ajouter un analyseur SharpeRatio
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.03)

# Maximum Drawdown pour la stratégie ML
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
drawdown = results[0].analyzers.drawdown.get_analysis()['max']['drawdown']

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

# Maximum Drawdown pour la stratégie ML

print(f"Maximum Drawdown pour la stratégie ML : {drawdown:.2f} %")

# Valeur final du portefeuille pour la stratégie Buy & Hold
rendement_bh = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) / data['Adj Close'].iloc[0] * 100
valeur_finale_bh = capital_initial * (1 + rendement_bh / 100)

print(f"Valeur finale du portefeuille en Buy & Hold: {valeur_finale_bh:.2f} USD")
print(f"Rendement Buy & Hold : {rendement_bh:.2f} %")

# Sharpe Ratio pour le modèle Buy & Hold
rendement_bh_daily = data['Adj Close'].pct_change()[1:]  # Rendement quotidien

sharpe_ratio_bh = (rendement_bh_daily.mean() / rendement_bh_daily.std()) * np.sqrt(252)  # 252 jours de trading dans une année

print(f"Sharpe Ratio Buy & Hold : {sharpe_ratio_bh:.2f}")

# Maximum Drawdown pour la stratégie Buy & Hold
drawdown_bh = (data['Adj Close'].cummax() - data['Adj Close']) / data['Adj Close'].cummax() * 100
max_drawdown_bh = drawdown_bh.max()

print(f"Maximum Drawdown pour la stratégie Buy & Hold : {max_drawdown_bh:.2f} %")

# Comparaison des rendements
if rendement > rendement_bh:
    print("La stratégie de trading basée sur le modèle de machine learning a surperformé la stratégie Buy & Hold.")
elif rendement < rendement_bh:
    print("La stratégie Buy & Hold a surperformé la stratégie de trading basée sur le modèle de machine learning.")
else:
    print("Les deux stratégies ont produit le même rendement.")
