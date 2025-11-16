import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================================
# 1. CONFIGURATION & CHARGEMENT DU MODÈLE
# ============================================================================

# Noms des fichiers (à adapter à votre environnement de production !)
MODEL_FILEPATH = '/model/lightgbm_revenue_model.joblib'
ENCODERS_FILEPATH = '/model/label_encoders.joblib'

# Mise en cache du chargement pour accélérer l'application Streamlit
@st.cache_resource
def load_assets():
    """Charge le modèle et les encodeurs une seule fois."""
    try:
        loaded_model = joblib.load(MODEL_FILEPATH)
        loaded_encoders = joblib.load(ENCODERS_FILEPATH)
        return loaded_model, loaded_encoders
    except FileNotFoundError:
        st.error("❌ Erreur: Fichiers de modèle ou d'encodeurs non trouvés.")
        st.stop() # Arrête l'exécution si les fichiers manquent
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des actifs: {e}")
        st.stop()

# Charger les actifs au début
loaded_model, loaded_encoders = load_assets()
features_order = loaded_model.feature_name_
categorical_cols_to_encode = ['pickup_zone', 'dropoff_zone', 'rush_hour_flag']

# ============================================================================
# 2. FONCTION DE PRÉDICTION
# ============================================================================

def predict_revenue(input_data_dict):
    """Prépare les données et fait la prédiction."""
    
    # Transformer le dictionnaire d'entrée en DataFrame (une seule ligne)
    df_new = pd.DataFrame([input_data_dict])
    
    # Appliquer l'encodage chargé sur les colonnes catégorielles
    for col in categorical_cols_to_encode:
        le = loaded_encoders[col]
        new_value = df_new[col].astype(str).values 
        
        try:
            # Transformation normale
            df_new[col] = le.transform(new_value)
        except ValueError:
            # Gestion des étiquettes jamais vues (Unseen Labels)
            # Remplacer par 0 (index de la classe inconnue / par défaut)
            df_new[col] = 0 

    # S'assurer que tous les IDs et nombres sont du bon type (pour LightGBM)
    for col in features_order:
        try:
            # Cas des IDs et des dénombrements (entiers)
            if col in ['vendorid', 'pulocationid', 'dolocationid', 'pickup_month', 'pickup_hour', 'pickup_day_of_week_num']:
                df_new[col] = df_new[col].astype(int)
            # Cas des montants et distances (flottants)
            elif col in ['trip_distance', 'fare_per_mile', 'congestion_surcharge', 'airport_fee', 'passenger_count_that_day']:
                df_new[col] = df_new[col].astype(float)
        except Exception:
            # Fallback si la conversion échoue (devrait être rare)
            df_new[col] = 0 

    # Trier les colonnes dans l'ordre exact attendu par le modèle
    X_new = df_new[features_order]

    # Faire la prédiction
    prediction = loaded_model.predict(X_new)
    return prediction[0]

# ============================================================================
# 3. INTERFACE STREAMLIT
# ============================================================================

st.set_page_config(page_title="Prédiction de Revenu Taxi", layout="centered")

st.title("🚕 Prédiction de Revenu de Trajet (Modèle LightGBM)")
st.markdown("Entrez les détails du trajet pour estimer le revenu du chauffeur.")

with st.form("revenue_form"):
    
    st.header("1. Détails du Chauffeur & Temps")
    
    # Colonnes 1 (Temps)
    col1, col2, col3, col4 = st.columns(4)
    vendorid = col1.number_input("ID Fournisseur (Vendor)", value=2, min_value=1, format="%d")
    pickup_hour = col2.slider("Heure de Départ (24h)", min_value=0, max_value=23, value=19)
    pickup_day_of_week_num = col3.selectbox("Jour (1=Lun, 5=Ven)", options=list(range(1, 8)), index=4)
    pickup_month = col4.slider("Mois", min_value=1, max_value=12, value=11)
    
    st.header("2. Localisation et Zones")
    
    # Suggestions de catégories (basées sur les classes du LabelEncoder)
    try:
        zones = loaded_encoders['pickup_zone'].classes_
        rush_flags = loaded_encoders['rush_hour_flag'].classes_
    except:
        zones = ['Midtown', 'Upper East Side', 'Times Square']
        rush_flags = ['Peak', 'Off-Peak']

    # Colonnes 2 (Zones)
    col5, col6 = st.columns(2)
    pulocationid = col5.number_input("ID Zone Départ (PU)", value=142, min_value=1)
    pickup_zone = col5.selectbox("Nom Zone Départ", options=zones, index=zones.tolist().index('Upper East Side') if 'Upper East Side' in zones else 0)
    
    dolocationid = col6.number_input("ID Zone Arrivée (DO)", value=238, min_value=1)
    dropoff_zone = col6.selectbox("Nom Zone Arrivée", options=zones, index=zones.tolist().index('Times Square') if 'Times Square' in zones else 0)

    st.header("3. Détails du Trajet & Frais")
    
    # Colonnes 3 (Trajet)
    col7, col8, col9 = st.columns(3)
    trip_distance = col7.number_input("Distance (miles)", value=3.5, min_value=0.1, format="%.2f")
    fare_per_mile = col8.number_input("Tarif/Mile", value=2.8, min_value=0.1, format="%.2f")
    passenger_count_that_day = col9.number_input("Nb. Passagers", value=1, min_value=1, format="%d")

    # Colonnes 4 (Frais)
    col10, col11, col12 = st.columns(3)
    congestion_surcharge = col10.number_input("Surcharge Congestion ($)", value=2.5, min_value=0.0, format="%.2f")
    airport_fee = col11.number_input("Frais Aéroport ($)", value=0.0, min_value=0.0, format="%.2f")
    rush_hour_flag = col12.selectbox("Heure de Pointe", options=rush_flags, index=1)
    
    submitted = st.form_submit_button("Estimer le Revenu 💰")

if submitted:
    
    # Assembler les données brutes
    input_data_dict = {
        'vendorid': vendorid,
        'pickup_hour': pickup_hour,
        'pickup_day_of_week_num': pickup_day_of_week_num,
        'pickup_month': pickup_month,
        'pulocationid': pulocationid,
        'pickup_zone': pickup_zone,
        'dolocationid': dolocationid,
        'dropoff_zone': dropoff_zone,
        'trip_distance': trip_distance,
        'fare_per_mile': fare_per_mile,
        'congestion_surcharge': congestion_surcharge,
        'airport_fee': airport_fee,
        'rush_hour_flag': rush_hour_flag,
        'passenger_count_that_day': passenger_count_that_day
    }
    
    with st.spinner('Calcul en cours...'):
        predicted_revenue = predict_revenue(input_data_dict)
    
    st.success(f"### 🎯 Revenu Prédit pour ce Trajet : ${predicted_revenue:.2f}")
    st.balloons()
