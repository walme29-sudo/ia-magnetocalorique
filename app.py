import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# 1. Configuration de la page
st.set_page_config(page_title="Comparaison de Régression", layout="wide", page_icon="📈")

# 2. Mise en cache pour les performances
@st.cache_data
def generate_data():
    """Génère un jeu de données non linéaire (sinusoïdal)."""
    np.random.seed(42)
    X = np.sort(np.random.rand(200, 1) * 10, axis=0)
    y = np.sin(X).ravel() + np.random.randn(200) * 0.2
    return X, y

def main():
    st.title("📈 Comparaison : Régression Linéaire vs Réseau de Neurones")
    st.markdown("Analysez et comparez les performances des modèles sur des données non linéaires.")

    # 3. Barre latérale - Source des données
    st.sidebar.header("1. Source des données")
    data_source = st.sidebar.radio("Choisissez vos données :", ("Données générées", "Importer un fichier CSV"))

    if data_source == "Données générées":
        X, y = generate_data()
    else:
        uploaded_file = st.sidebar.file_uploader("Uploadez un CSV (Col 1: X, Col 2: y)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 1].values
        else:
            st.info("Veuillez uploader un fichier CSV pour continuer.")
            return

    # 4. Barre latérale - Paramètres du modèle
    st.sidebar.header("2. Paramètres du modèle")
    model_choice = st.sidebar.selectbox("Choisissez le modèle :", ["Régression Linéaire", "MLP Regressor"])

    # Prétraitement
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement
    if model_choice == "Régression Linéaire":
        model = LinearRegression()
    else:
        hidden_nodes = st.sidebar.slider("Nombre de nœuds cachés (MLP)", min_value=10, max_value=500, value=100, step=10)
        max_iter = st.sidebar.number_input("Itérations maximales", min_value=500, max_value=5000, value=2000, step=500)
        model = MLPRegressor(hidden_layer_sizes=(hidden_nodes, hidden_nodes), max_iter=max_iter, random_state=42)

    # Ajustement et prédictions
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    # 5. Affichage des Métriques
    st.subheader("📊 Performances du Modèle")
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Modèle Utilisé", model_choice)
    col2.metric("Erreur Quadratique Moyenne (MSE)", f"{mse:.4f}")
    col3.metric("Score R²", f"{r2:.4f}")
    
    if r2 < 0.5 and model_choice == "Régression Linéaire":
        st.warning("Le score R² est faible. La régression linéaire n'est probablement pas adaptée à ce type de données courbes.")

    # 6. Visualisation
    st.subheader("📉 Visualisation des Prédictions")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y, color='gray', alpha=0.5, label='Données réelles (y)')
    
    # Tri des données pour tracer une ligne continue propre
    sort_idx = np.argsort(X.flatten())
    ax.plot(X[sort_idx], y_pred[sort_idx], color='red', linewidth=2, label=f'Prédictions ({model_choice})')
    
    ax.set_xlabel("X (Feature)")
    ax.set_ylabel("y (Target)")
    ax.legend()
    st.pyplot(fig)

    # 7. Bouton de téléchargement
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Télécharger le graphique (PNG)",
        data=buf,
        file_name=f"graphique_{model_choice.replace(' ', '_')}.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()
