import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import backtrader as bt

data = yf.download('NVDA', start='2023-01-01', end='2024-01-01')

# Étape 2 : Calculer des indicateurs techniques
def add_indicators(data):
    # Moyenne Mobile
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    # Indicateur RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    # Supprimer les valeurs manquantes
    data.dropna(inplace=True)
    return data

data = add_indicators(data)



# Étape 3 : Préparer les données pour l'apprentissage supervisé
# Créer la variable cible : 1 si le prix de clôture augmente le lendemain, 0 sinon
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Sélectionner les caractéristiques (features)
features = ['SMA_10', 'SMA_50', 'RSI']
X = data[features]
y = data['Target']

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Étape 4 : Construire et entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Étape 5 : Utiliser le modèle dans une stratégie de trading Backtrader
class MLBasedStrategy(bt.Strategy):
    def __init__(self):
        # Définir les indicateurs de Backtrader
        self.sma_10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=10)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.datas[0], period=50)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.datas[0], period=14)

    def next(self):
        # Créer une observation pour la journée actuelle
        obs = [
            self.sma_10[0],
            self.sma_50[0],
            self.rsi[0]
        ]

        # Vérifier que toutes les valeurs sont disponibles (pour éviter les NaN au début)
        if None in obs:
            return

        # Créer un DataFrame pour l'observation avec les mêmes noms de colonnes que pour l'entraînement
        obs_df = pd.DataFrame([obs], columns=['SMA_10', 'SMA_50', 'RSI'])

        # Prédire l'action à entreprendre
        pred = model.predict(obs_df)[0]

        # Si le modèle prédit une hausse, acheter
        if not self.position and pred == 1:
            self.buy(size=5000)
        # Si le modèle prédit une baisse, vendre si on a une position
        elif self.position and pred == 0:
            self.sell(size=5000)

# Étape 6 : Backtester la stratégie

# Appliquer une perturbation aléatoire sur les prix (ajouter du bruit gaussien)
noise = np.random.normal(0, 0.02, len(data))
data['Close'] = data['Close'] * (1 + noise)

# Charger les données de Backtrader
data_feed = bt.feeds.PandasData(dataname=data)

# Créer une instance de Cerebro
cerebro = bt.Cerebro()
cerebro.adddata(data_feed)

# Ajouter la stratégie
cerebro.addstrategy(MLBasedStrategy)

# Définir le capital de départ
capital_initial = 250000
cerebro.broker.setcommission(commission=0.001)
cerebro.broker.setcash(capital_initial)

# Lancer le backtest
cerebro.run()

# Obtenir la valeur finale du portefeuille et le rendement
valeur_finale = cerebro.broker.getvalue()
rendement = (valeur_finale - capital_initial) / capital_initial * 100

# Afficher la valeur finale et le rendement
print(f"Valeur finale du portefeuille : {valeur_finale:.2f} USD")
print(f"Rendement total : {rendement:.2f} %")
