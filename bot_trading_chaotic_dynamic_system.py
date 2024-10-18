import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Fonction pour simuler le système de Lorenz
def lorenz_system(state, t, sigma=20, rho=30, beta=3.5):
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

# Liste des actifs pour le portefeuille
assets = ['BTC-USD', 'ETH-USD', 'SHIB-USD']

# Télécharger les données des actifs sur Yahoo Finance
data_1 = yf.download(assets, interval='1m', start="2024-09-25", end="2024-10-02")['Close']
data_2 = yf.download(assets, interval='1m', start="2024-10-02", end="2024-10-09")['Close']
data_3 = yf.download(assets,interval='1m', start="2024-10-10", end="2024-10-17")['Close']

# Concaténer les données des différentes périodes
data = pd.concat([data_1, data_2, data_3])

# Calculer les rendements quotidiens pour chaque actif
returns = data.pct_change().dropna()

# Ajouter les signaux chaotiques (Lorenz) aux données des actifs
returns['Lorenz_x'] = np.interp(np.arange(len(returns)), np.arange(len(lorenz_data)), x)
returns['Lorenz_y'] = np.interp(np.arange(len(returns)), np.arange(len(lorenz_data)), y)
returns['Lorenz_z'] = np.interp(np.arange(len(returns)), np.arange(len(lorenz_data)), z)

# Création des features pour le modèle de machine learning
for asset in assets:
    returns[f'Lag1_{asset}'] = returns[asset].shift(1)
    returns[f'Lag2_{asset}'] = returns[asset].shift(2)

# Retirer les données manquantes
returns.dropna(inplace=True)

# Pondérations ajustables pour chaque actif (par exemple, pondérations personnalisées)
weights = np.array([0.5, 0.3, 0.2])  # Somme = 1

# Définir les features et la variable cible pour le modèle
# Utilisation des signaux chaotiques et des rendements décalés comme features
X = returns[['Lorenz_x', 'Lorenz_y', 'Lorenz_z'] + [f'Lag1_{asset}' for asset in assets] + [f'Lag2_{asset}' for asset in assets]]

# Utilisation du rendement moyen pondéré pour générer la variable cible
y = np.where(np.dot(returns[assets], weights) > 0, 1, 0)  # 1 = achat (rendement moyen pondéré positif), 0 = vente (rendement moyen pondéré négatif)

# Diviser en ensembles d'entraînement et de test
train_size = int(0.8 * len(returns))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Modèle de machine learning : Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions et signaux de trading
returns['ML_Signal'] = model.predict(X)

# Calcul de la précision du modèle
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Générer des positions de trading basées sur les prédictions du modèle
returns['Position'] = returns['ML_Signal'].shift(1)

# Calculer les rendements pondérés du portefeuille uniquement sur les actifs financiers
returns['Portfolio_Returns'] = np.dot(returns[assets], weights)

# Calcul des rendements stratégiques en fonction des positions de trading
returns['Strategy_Returns'] = returns['Portfolio_Returns'] * returns['Position']

# Calcul du rendement cumulé du portefeuille pour la stratégie ML
returns['Cumulative_Strategy_Returns'] = (1 + returns['Strategy_Returns']).cumprod()

# Calcul du rendement cumulé pour la stratégie Buy & Hold
returns['Cumulative_Buy_Hold_Returns'] = (1 + returns['Portfolio_Returns']).cumprod()

# Calcul du Sharpe Ratio pour la stratégie ML
sharpe_ratio = returns['Strategy_Returns'].mean() / returns['Strategy_Returns'].std() * np.sqrt(252)

# Affichage des résultats
plt.figure(figsize=(12, 8))
plt.plot(returns['Cumulative_Strategy_Returns'], label="Stratégie ML + Lorenz")
plt.plot((1 + returns['Portfolio_Returns']).cumprod(), label="Buy & Hold Portfolio")
plt.title(f"Stratégie de Trading ML + Lorenz vs Buy & Hold Portfolio (Sharpe Ratio: {sharpe_ratio:.2f})")
plt.legend()
plt.show()

# Afficher les rendements et le Sharpe Ratio
total_return_strategy = returns['Cumulative_Strategy_Returns'].iloc[-1] - 1
total_return_buy_hold = returns['Cumulative_Buy_Hold_Returns'].iloc[-1] - 1
print(f"Valeur finale du portefeuille ML : {(1 + total_return_strategy) * 1000000:.2F}")
print(f"Rendement total de la stratégie ML : {total_return_strategy * 100:.2f}%")
print(f"Sharpe Ratio : {sharpe_ratio:.2f}")

# Calcul du Sharpe Ratio pour Buy & Hold Portfolio
risk_free_rate = 0.045 / 252  # Taux sans risque approximé sur une base quotidienne
mean_return = returns['Portfolio_Returns'].mean()
volatility = returns['Portfolio_Returns'].std()
sharpe_ratio_buy_hold = (mean_return - risk_free_rate) / volatility * np.sqrt(252)

# Affichage du résultat du Sharpe Ratio pour Buy & Hold Portfolio
print(f"Valeur finale du portefeuille ML : {(1 + total_return_buy_hold) * 1000000:.2F}")
print(f"Rendement total du portefeuille Buy & Hold : {total_return_buy_hold * 100:.2f}%")
print(f"Sharpe Ratio pour le portefeuille Buy & Hold : {sharpe_ratio_buy_hold:.2f}")
