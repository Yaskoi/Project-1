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

# Étape 1 : Obtenir les données historiques depuis Yahoo Finance
data_1 = yf.download("SHIB-USD", interval='1m', start="2024-09-25", end="2024-09-29")
data_2 = yf.download("SHIB-USD", interval='1m', start="2024-09-30", end="2024-10-07")
data_3 = yf.download("SHIB-USD", interval='1m', start="2024-10-08", end="2024-10-15")
data_4 = yf.download("SHIB-USD", interval='1m', start="2024-10-16", end="2024-10-23")

data = pd.concat([data_1, data_2, data_3, data_4])

data['Returns'] = data['Adj Close'].pct_change()

# Intégrer les signaux chaotiques (Lorenz) aux données du BTC
data['Lorenz_x'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), x)
data['Lorenz_y'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), y)
data['Lorenz_z'] = np.interp(np.arange(len(data)), np.arange(len(lorenz_data)), z)

# Création des features pour le modèle de machine learning
data['Lag1_Returns'] = data['Returns'].shift(1)
data['Lag2_Returns'] = data['Returns'].shift(2)

# Retirer les données manquantes
data.dropna(inplace=True)

# Définir les features et la variable cible pour le modèle
X = data[['Lorenz_x', 'Lorenz_y', 'Lorenz_z', 'Lag1_Returns', 'Lag2_Returns']]
y = np.where(data['Returns'] > 0, 1, 0)  # 1 = achat (rendements positifs), 0 = vente (rendements négatifs)

# Diviser en ensembles d'entraînement et de test
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Modèle de machine learning : Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions et signaux de trading
data['ML_Signal'] = model.predict(X)

# Calcul de la précision du modèle
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Générer des positions de trading basées sur les prédictions du modèle
data['Position'] = data['ML_Signal'].shift(1)
data['Strategy_Returns'] = data['Position'] * data['Returns']

# Calcul du rendement cumulé et du Sharpe Ratio
data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()

# Calcul du Sharpe Ratio
sharpe_ratio = data['Strategy_Returns'].mean() / data['Strategy_Returns'].std() * np.sqrt(252)

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.plot(data['Cumulative_Strategy_Returns'], label="Stratégie ML + Lorenz")
plt.plot((1 + data['Returns']).cumprod(), label="Buy & Hold BTC")
plt.title(f"Stratégie de Trading ML + Lorenz vs Buy & Hold (Sharpe Ratio: {sharpe_ratio:.2f})")
plt.legend()
plt.show()

# Afficher les rendements et le Sharpe Ratio
total_return = data['Cumulative_Strategy_Returns'].iloc[-1] - 1
print(f"Valeur finale du portefeuille : {(1 + total_return) * 1000:.2F}")
print(f"Rendement total de la stratégie : {total_return * 100:.2f}%")
print(f"Sharpe Ratio : {sharpe_ratio:.2f}")

# Paramètres pour le calcul du Sharpe Ratio
risk_free_rate = 0.01 / 252  # Taux sans risque approximé sur une base quotidienne (1% par an)

# Calcul du rendement moyen et de l'écart-type
mean_return = data['Returns'].mean()
volatility = data['Returns'].std()

return_buy_hold = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) / data['Adj Close'].iloc[0]
print(f"Valeur finale du portefeuille en Buy & Hold : {(1 + return_buy_hold) * 1000:.2F}")
print(f"Rendement total pour la stratégie Buy & Hold : {return_buy_hold * 100:.2f}%")

# Calcul du Sharpe Ratio
sharpe_ratio_buy_hold = (mean_return - risk_free_rate) / volatility * np.sqrt(252)

# Affichage du résultat
print(f"Sharpe Ratio pour la stratégie Buy & Hold : {sharpe_ratio_buy_hold:.2f}")
