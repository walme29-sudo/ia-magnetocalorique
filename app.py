import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="IA Magnétocalorique - ISSAT", layout="wide")

# --- FONCTIONS D'EXPORT ---
def to_excel_full(df, s_max, rcp, t_c):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Données et Prédictions')
        df_stats = pd.DataFrame({
            "Paramètre": ["Delta S Max (J/kg.K)", "RCP (Relative Cooling Power)", "Température de Curie approx (K)"],
            "Valeur": [s_max, rcp, t_c]
        })
        df_stats.to_excel(writer, sheet_name='Résumé Physique', index=False)
    return output.getvalue()

def plot_to_pdf(fig):
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches='tight')
    return output.getvalue()

# ====== INTERFACE UTILISATEUR ======
col_logo, col_titre = st.columns([1, 4])
with col_logo:
    try:
        st.image("logo.png", width=120) 
    except:
        st.info("📍 ISSAT Kasserine")

with col_titre:
    st.title("🧲 Analyse et Prédiction IA (Effet Magnétocalorique)")
    st.markdown("**Développeur :** DALHOUMI WALID | **Encadrant :** Projet ISSAT")

st.divider()

# --- CHARGEMENT DU FICHIER ---
file = st.file_uploader("Charger votre fichier CSV (Colonnes requises : T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    try:
        data = pd.read_csv(file, sep=None, engine='python').dropna()
        cols_needed = ["T", "M_1T", "M_2T", "M_3T"]
        
        if all(c in data.columns for c in cols_needed):
            T = data["T"].values
            M_matrix = data[["M_1T", "M_2T", "M_3T"]].values
            H_mesure = np.array([1, 2, 3]).reshape(-1, 1)
            H_pred = np.array([[5]])

            # --- IA : PRÉDICTION À 5 TESLA ---
            model = LinearRegression()
            M_predicted_5T = []
            for i in range(len(T)):
                model.fit(H_mesure, M_matrix[i])
                M_predicted_5T.append(model.predict(H_pred)[0])
            M_predicted_5T = np.array(M_predicted_5T)

            # --- PHYSIQUE : MAXWELL & THERMODYNAMIQUE ---
            dM_dT_1 = np.gradient(M_matrix[:, 0], T)
            dM_dT_2 = np.gradient(M_matrix[:, 1], T)
            dM_dT_3 = np.gradient(M_matrix[:, 2], T)
            
            deltaS_mesure = np.trapezoid([dM_dT_1, dM_dT_2, dM_dT_3], x=[1, 2, 3], axis=0)
            
            # Paramètres clés
            Smax = np.max(np.abs(deltaS_mesure))
            t_curie = T[np.argmax(np.abs(deltaS_mesure))]
            
            # Calcul du RCP
            indices_fwhm = np.where(np.abs(deltaS_mesure) >= Smax/2)[0]
            rcp = Smax * (T[indices_fwhm[-1]] - T[indices_fwhm[0]]) if len(indices_fwhm) > 1 else 0

            # --- AFFICHAGE ---
            tab1, tab2, tab3 = st.tabs(["📈 Graphiques", "📊 Résultats & Métriques", "📥 Exportation"])

           # --- Remplacez la section tab1 par celle-ci ---
            with tab1:
                st.subheader("Visualisation des Données Magnétiques")
    
                # Création de deux colonnes pour mettre les graphiques côte à côte
                col_graph1, col_graph2 = st.columns(2)
    
                    with col_graph1:
                        st.markdown("**Magnétisation (Mesures vs IA)**")
                        df_m_complet = pd.DataFrame({
                            "M (1T)": M_matrix[:, 0],
                            "M (2T)": M_matrix[:, 1],
                            "M (3T)": M_matrix[:, 2],
                            "M (5T) - IA": M_predicted_5T
                        }, index=T)
                        # On réduit la hauteur avec le paramètre height
                        st.line_chart(df_m_complet, height=300)
    
                    with col_graph2:
                        st.markdown("**Variation d'Entropie ΔS**")
                        df_ds = pd.DataFrame({"ΔS (J/kg·K)": deltaS_mesure}, index=T)
                        # On réduit la hauteur avec le paramètre height
                        st.line_chart(df_ds, height=300)

                    st.caption("💡 Les graphiques sont affichés côte à côte pour une meilleure lisibilité sur écran large.")


            with tab2:
                c1, c2, c3 = st.columns(3)
                c1.metric("ΔS Max (J/kg·K)", f"{Smax:.4f}")
                c2.metric("RCP", f"{rcp:.2f}")
                c3.metric("Temp. Curie (K)", f"{t_curie:.1f}")
                st.dataframe(data, use_container_width=True)

            with tab3:
                st.subheader("Télécharger les rapports")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(T, deltaS_mesure, label="ΔS Mesuré (1-3T)", color='blue', linewidth=2)
                ax.set_xlabel("Température (K)")
                ax.set_ylabel("ΔS (J/kg·K)")
                ax.set_title(f"Variation d'Entropie - Tc approx = {t_curie:.1f}K")
                ax.legend()
                
                col_ex, col_pdf = st.columns(2)
                df_export = pd.DataFrame({
                    "T (K)": T, "M_1T": M_matrix[:, 0], "M_5T_IA": M_predicted_5T, "DeltaS": deltaS_mesure
                }).set_index("T (K)")

                col_ex.download_button(
                    "📥 Télécharger Data Excel",
                    data=to_excel_full(df_export, Smax, rcp, t_curie),
                    file_name="Resultats_ISSAT_Walid.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                col_pdf.download_button(
                    "📥 Télécharger Courbe PDF",
                    data=plot_to_pdf(fig),
                    file_name="Rapport_Courbe_DS.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Colonnes manquantes (T, M_1T, M_2T, M_3T).")
    except Exception as e:
        st.error(f"Erreur : {e}")
else:
    st.info("Veuillez charger un fichier CSV.")




