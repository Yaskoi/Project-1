import mysql.connector
import argparse
import pymongo
import requests
import pandas as pd
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


connexion = mysql.connector.connect(
    host="144.24.194.65",
    port=3999,
    user="mag1_student",
    password="Gogo1gogo2",
    database="mag1_project"
)

curseur = connexion.cursor()

nom_table = "project"

curseur.execute(f"SELECT * FROM {nom_table}")
donnees = curseur.fetchall()

mysqltable = PrettyTable(["Company ID", "Expenses", "R&D Spend"])

mysqltable.add_rows(donnees)

donnees_mysql_liste = [dict(zip(mysqltable.field_names, ligne)) for ligne in mysqltable.rows]

curseur.close()
connexion.close()


adresse_connexion_mongodb = "208.87.130.253"
port = 27017
nom_utilisateur = "mag1_student"
mot_de_passe = "Gogo1gogo2"
nom_base_de_donnees = "mag1_project"
nom_collection = "project"

client = pymongo.MongoClient(
    f"mongodb://mag1_student:Gogo1gogo2@208.87.130.253/mag1_project"
)

db = client[nom_base_de_donnees]
collection = db[nom_collection]

donnees = collection.find()

colonnes_a_afficher = ["Company ID", "Revenue", "Employee Count", "Credit Rating", "Risk"]

donnees_triees = sorted(donnees, key=lambda x: x.get("Company ID", ""))

mongoDBtable = PrettyTable(colonnes_a_afficher)

for ligne in donnees_triees:
    valeurs = [ligne.get(colonne, "") for colonne in colonnes_a_afficher]
    mongoDBtable.add_row(valeurs)

donnees_mongo_liste = [dict(zip(mongoDBtable.field_names, ligne)) for ligne in mongoDBtable.rows]


url = "https://h.chifu.eu/data.html"

session = requests.Session()

response = session.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')
table = soup.find('table')

colonnes_a_afficher = ["Company ID", "Profit", "Debt-to-Equity Ratio", "Price-to-Earnings", "Market Capitalization"]

htmltable = PrettyTable(colonnes_a_afficher)

donnees = []
for ligne in table.find_all('tr'):
    valeurs = [colonne.text.strip() for colonne in ligne.find_all('td')]
    if valeurs:  # Ignorer les lignes sans données
        donnees.append(valeurs)

donnees_triees = sorted(donnees[1:], key=lambda x: int(x[0]))

for valeurs in donnees_triees:
    htmltable.add_row(valeurs)

donnees_html_liste = [dict(zip(htmltable.field_names, ligne)) for ligne in htmltable.rows]


df1 = pd.DataFrame(donnees_mysql_liste)
df2 = pd.DataFrame(donnees_mongo_liste)
df3 = pd.DataFrame(donnees_html_liste)

df1["Company ID"] = df1["Company ID"].astype(int)
df2["Company ID"] = df2["Company ID"].astype(int)
df3["Company ID"] = df3["Company ID"].astype(int)

df_merged = pd.merge(df3, df2, on="Company ID", how="outer")
df_merged = pd.merge(df_merged, df1, on="Company ID", how="outer")
df_merged = df_merged.dropna()
tableau = tabulate(df_merged, headers='keys', tablefmt='pretty')

df_merged['Credit Rating'] = df_merged['Credit Rating'].map({'AAA' : 9, 'AA' : 8, 'A' : 7, 'BBB' : 6, 'BB' : 5, 'B' : 4, 'CCC' : 3, 'CC' : 2, 'C': 1})



X = df_merged.drop(columns=['Risk', 'Credit Rating'], axis=1)
y = df_merged['Risk']

model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, random_state=76)

model.fit(X, y)

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
conf_matrix = confusion_matrix(y, predictions)
class_report = classification_report(y, predictions)

print(f"Accuracy: {accuracy}")
print("Confusion matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


table3 = PrettyTable(['Company ID', 'Risks'])

for valeur1, valeur2 in zip(df_merged['Company ID'], predictions):
    table3.add_row([valeur1, valeur2])

companyID = df_merged['Company ID']

Risks = predictions

min_length = min(len(companyID), len(Risks))

df = pd.DataFrame(index=range(min_length))

df['Company ID'] = companyID
df['Risks'] = Risks

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="predicts")

args = parser.parse_args()

df.to_csv(args.output, index=False, encoding='utf-8')
