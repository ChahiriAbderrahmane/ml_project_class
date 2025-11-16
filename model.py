import joblib
import pandas as pd
import numpy as np

# Noms des fichiers que vous avez sauvegardés
MODEL_FILEPATH = '/home/chahiri/repos/ml_pr_class/model/lightgbm_revenue_model.joblib'
ENCODERS_FILEPATH = '/home/chahiri/repos/ml_pr_class/model/label_encoders.joblib'

# Charger le modèle et les encodeurs
try:
    loaded_model = joblib.load(MODEL_FILEPATH)
    loaded_encoders = joblib.load(ENCODERS_FILEPATH)
    print("✅ Modèle et encodeurs chargés avec succès.")
except FileNotFoundError:
    print("❌ Erreur: Assurez-vous que 'lightgbm_revenue_model.joblib' et 'label_encoders.joblib' existent.")

# Données brutes reçues en temps réel (votre exemple)
data_brute = {
    'vendorid': [2],
    'pickup_hour': [19],
    'pickup_day_of_week_num': [5],  # Vendredi
    'pickup_month': [11],
    'pulocationid': [142],
    'pickup_zone': ['Upper East Side'], # ⬅️ Cette valeur pose problème
    'dolocationid': [238],
    'dropoff_zone': ['Times Square'],
    'trip_distance': [3.5],
    'fare_per_mile': [2.8],
    'congestion_surcharge': [2.5],
    'airport_fee': [0],
    'rush_hour_flag': ['Off-Peak'],
    'passenger_count_that_day': [1]
}
df_new = pd.DataFrame(data_brute)

# Colonnes catégorielles qui nécessitent l'objet LabelEncoder
categorical_cols_to_encode = ['pickup_zone', 'dropoff_zone', 'rush_hour_flag']
features_order = loaded_model.feature_name_ # Ordre des features attendu

print("\n🔧 Encodage des nouvelles données...")

# Appliquer l'encodage chargé SANS FIT (Uniquement transform)
for col in categorical_cols_to_encode:
    le = loaded_encoders[col]
    
    # CONVERSION CLÉ 1: Convertir la nouvelle valeur en string, puis en array 
    # pour que le.transform() puisse l'accepter
    new_value = df_new[col].astype(str).values 
    
    try:
        # Tenter la transformation normale
        df_new[col] = le.transform(new_value)
        print(f"  - Encoded '{col}' (Ex: '{new_value[0]}' -> {df_new[col].iloc[0]})")
    
    except ValueError as e:
        # CONVERSION CLÉ 2: Gestion des étiquettes jamais vues (Unseen Labels)
        # Si la nouvelle catégorie n'est pas dans le train set, nous utilisons le mode "inconnu"
        print(f"  ❌ Erreur critique dans '{col}' : {e}")
        print("  💡 La valeur sera remplacée par 0 (ou la plus fréquente).")
        df_new[col] = 0 # Remplacer par 0 (index de la classe la plus fréquente, ou une valeur inconnue)

# 4. S'assurer que tous les IDs et nombres sont du bon type (int/float)
for col in features_order:
    # Assurez-vous que les colonnes numériques/ID sont des types numériques
    if col in ['vendorid', 'pulocationid', 'dolocationid', 'pickup_month', 'pickup_hour']:
        df_new[col] = df_new[col].astype(int)
    elif col in ['trip_distance', 'fare_per_mile', 'congestion_surcharge', 'airport_fee', 'passenger_count_that_day']:
        df_new[col] = df_new[col].astype(float)


# Trier les colonnes dans l'ordre exact attendu par le modèle
X_new = df_new[features_order]

# 5. Faire la prédiction
prediction = loaded_model.predict(X_new)
predicted_revenue = prediction[0]

print("\n--- Résultat de la Prédiction ---")
print(f"💰 Le revenu attendu pour ce trajet est de : ${predicted_revenue:.2f}")