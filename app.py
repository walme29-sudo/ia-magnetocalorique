import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from io import BytesIO

# CONFIGURATION
st.set_page_config(page_title="IA Magnétocalorique - ISSAT", layout="wide")

# EXPORT FUNCTIONS
def to_excel_full(df, s_max, rcp, t_c):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Données et Prédictions')
        df_stats = pd.DataFrame({
            "Paramètre": ["Delta S Max (J/kg.K)", 
                          "RCP (Relative Cooling Power)", 
                          "Température de Curie approx (K)"],
            "Valeur": [s_max, rcp, t_c]
        })
        df_stats.to_excel(writer, sheet_name='Résumé Physique', index=False)
    return output.getvalue()

def plot_to_pdf(fig):
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches='tight')
    return output.getvalue()

# INTERFACE
st.title("🧲 Analyse et Prédiction IA - Effet Magnétocalorique")

file = st.file_uploader("Charger fichier CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    cols_needed = ["T", "M_1T", "M_2T", "M_3T"]

    if all(c in data.columns for c in cols_needed):

        T = data["T"].values
        M_matrix = data[["M_1T", "M_2T", "M_3T"]].values

        H_mesure = np.array([1, 2, 3]).reshape(-1, 1)
        H_pred = np.array([[5]])

        # ===== IA POLYNOMIALE =====
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )

        M_predicted_5T = []

        for i in range(len(T)):
            model.fit(H_mesure, M_matrix[i])
            M_predicted_5T.append(model.predict(H_pred)[0])

        M_predicted_5T = np.array(M_predicted_5T)

        # ===== CALCUL ΔS (1 → 5T) =====
        dM_dT_1 = np.gradient(M_matrix[:, 0], T)
        dM_dT_2 = np.gradient(M_matrix[:, 1], T)
        dM_dT_3 = np.gradient(M_matrix[:, 2], T)
        dM_dT_5 = np.gradient(M_predicted_5T, T)

        deltaS = np.trapezoid(
            [dM_dT_1, dM_dT_2, dM_dT_3, dM_dT_5],
            x=[1, 2, 3, 5],
            axis=0
        )

        Smax = np.max(np.abs(deltaS))
        t_curie = T[np.argmax(np.abs(deltaS))]

        # ===== RCP amélioré =====
        indices_fwhm = np.where(np.abs(deltaS) >= Smax / 2)[0]

        if len(indices_fwhm) > 1:
            rcp = np.trapezoid(
                np.abs(deltaS[indices_fwhm]),
                T[indices_fwhm]
            )
        else:
            rcp = 0

        # ===== AFFICHAGE =====
        st.subheader("Résultats")

        col1, col2, col3 = st.columns(3)
        col1.metric("ΔS Max (J/kg·K)", f"{Smax:.4f}")
        col2.metric("RCP", f"{rcp:.2f}")
        col3.metric("Température Curie (K)", f"{t_curie:.1f}")

        st.subheader("Graphiques")

        df_m = pd.DataFrame({
            "1T": M_matrix[:, 0],
            "2T": M_matrix[:, 1],
            "3T": M_matrix[:, 2],
            "5T (IA)": M_predicted_5T
        }, index=T)

        st.line_chart(df_m, height=300)

        df_ds = pd.DataFrame({"ΔS": deltaS}, index=T)
        st.line_chart(df_ds, height=300)

    else:
        st.error("Colonnes requises : T, M_1T, M_2T, M_3T")
else:
    st.info("Veuillez charger un fichier CSV.")
