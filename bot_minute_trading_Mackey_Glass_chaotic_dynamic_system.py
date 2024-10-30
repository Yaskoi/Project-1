import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Fonction pour simuler le système de Mackey-Glass
def mackey_glass(x, t, beta=0.2, gamma=0.1, n=10, tau=25):
    if t - tau < 0:
        xtau = 0.5  # Condition initiale pour le délai
    else:
        xtau = x[int(t - tau)]
    dxdt = beta * xtau / (1 + xtau**n) - gamma * x[t]
    return dxdt

# Simuler le système de Mackey-Glass
def simulate_mackey_glass(initial_value, t_points, tau=25):
    x = np.zeros(len(t_points))
    x[0] = initial_value
    for t in range(1, len(t_points)):
        x[t] = x[t - 1] + mackey_glass(x, t, tau=tau)
    return x

# Paramètres pour la simulation de Mackey-Glass
initial_value = 0.5
t_points = np.arange(0, 1000, 1)  # Points de temps pour la simulation

# Simuler le système de Mackey-Glass
mackey_glass_data = simulate_mackey_glass(initial_value, t_points)

# Étape 1 : Obtenir les données historiques depuis Yahoo Finance
data_1 = yf.download("SHIB-USD", interval='1m', start="2024-10-01", end="2024-10-05")
data_2 = yf.download("SHIB-USD", interval='1m', start="2024-10-06", end="2024-10-13")
data_3 = yf.download("SHIB-USD", interval='1m', start="2024-10-14", end="2024-10-21")
data_4 = yf.download("SHIB-USD", interval='1m', start="2024-10-22", end="2024-10-29")

data = pd.concat([data_1, data_2, data_3, data_4])

# Calcul des rendements
data['Returns'] = data['Adj Close'].pct_change()

# Intégrer les signaux chaotiques (Mackey-Glass) aux données du BTC
data['Mackey_Glass'] = np.interp(np.arange(len(data)), np.arange(len(mackey_glass_data)), mackey_glass_data)

# Appliquer une perturbation aléatoire sur les prix (ajouter du bruit gaussien)
noise = np.random.normal(0, 0.01, len(data))
data['Returns'] = data['Returns'] * (1 + noise)

# Création des features pour le modèle de machine learning
data['Lag1_Returns'] = data['Returns'].shift(1)
data['Lag2_Returns'] = data['Returns'].shift(2)

# Retirer les données manquantes
data.dropna(inplace=True)

# Définir les features et la variable cible pour le modèle
X = data[['Mackey_Glass', 'Lag1_Returns', 'Lag2_Returns']]
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
plt.plot(data['Cumulative_Strategy_Returns'], label="Stratégie ML + Mackey-Glass + Bruit")
plt.plot((1 + data['Returns']).cumprod(), label="Buy & Hold BTC")
plt.title(f"Stratégie de Trading ML + Mackey-Glass + Bruit vs Buy & Hold (Sharpe Ratio: {sharpe_ratio:.2f})")
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

# Calcul du Sharpe Ratio pour Buy & Hold
sharpe_ratio_buy_hold = (mean_return - risk_free_rate) / volatility * np.sqrt(252)
print(f"Sharpe Ratio pour la stratégie Buy & Hold : {sharpe_ratio_buy_hold:.2f}")
