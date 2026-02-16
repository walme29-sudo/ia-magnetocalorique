import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIGURATION =================
st.set_page_config(page_title="IA Magnétocalorique - Master Curve Edition", layout="wide")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Analyse_Expert')
    return output.getvalue()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("🔬 Paramètres IA")
    file = st.file_uploader("Fichier CSV (T, M_1T, M_2T...)", type=["csv"])
    nodes = st.slider("Neurones", 64, 512, 256)
    h_target = st.number_input("Champ Cible (Tesla)", 0.1, 10.0, 5.0)

st.title("🧲 Analyse Expert : Master Curve & Scaling")
st.divider()

if file:
    df = pd.read_csv(file).dropna()
    cols = df.columns.tolist()
    t_col = cols[0]
    m_cols = [c for c in cols if 'M_' in c]
    
    if len(m_cols) >= 2:
        T = df[t_col].values
        M_matrix = df[m_cols].values
        H_vals = np.array([float(''.join(c for c in col if c.isdigit() or c=='.')) for col in m_cols])

        # --- ENTRAÎNEMENT IA ---
        T_g, H_g = np.meshgrid(T, H_vals)
        X = np.column_stack([T_g.ravel(), H_g.ravel()])
        y = M_matrix.T.ravel()
        
        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_s = sc_X.fit_transform(X)
        y_s = sc_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        model = MLPRegressor(hidden_layer_sizes=(nodes, nodes), max_iter=3000, random_state=42)
        model.fit(X_s, y_s)

        # --- GÉNÉRATION DE DONNÉES POUR PLUSIEURS CHAMPS (Pour la Master Curve) ---
        # On prédit pour 1T, 2T, 3T et le champ cible pour voir si elles se superposent
        fields_to_test = sorted(list(set([1.0, 2.0, 3.0, h_target])))
        results_master = []

        for h in fields_to_test:
            X_p = sc_X.transform(np.column_stack([T, np.full_like(T, h)]))
            M_p = sc_y.inverse_transform(model.predict(X_p).reshape(-1, 1)).ravel()
            
            # Calcul Delta S
            dM_dT = np.gradient(M_p, T)
            # Intégration simplifiée pour la démo (trapeze sur H)
            ds = np.abs(dM_dT * h) # Approximation DeltaS ~ (dM/dT)*H
            
            s_max = np.max(ds)
            tc = T[np.argmax(ds)]
            
            # Calcul des températures de référence (T_r1 et T_r2) pour la Master Curve
            # T_r est souvent choisi à DeltaS_max / 2
            idx_r1 = np.where((T < tc) & (ds >= s_max/2))[0]
            idx_r2 = np.where((T > tc) & (ds >= s_max/2))[0]
            
            if len(idx_r1) > 0 and len(idx_r2) > 0:
                tr1, tr2 = T[idx_r1[0]], T[idx_r2[-1]]
                # Température réduite theta
                theta = np.where(T <= tc, -(T - tc) / (tr1 - tc), (T - tc) / (tr2 - tc))
                results_master.append({'h': h, 'ds': ds, 's_max': s_max, 'theta': theta, 'tc': tc})

        # --- AFFICHAGE ---
        tab1, tab2 = st.tabs(["📈 Courbes Normalisées", "🧬 Master Curve (Scaling)"])

        with tab1:
            st.subheader("Variation d'Entropie Magnétique $\Delta S_m(T)$")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            for res in results_master:
                ax1.plot(T, res['ds'], label=f"H = {res['h']}T")
            ax1.set_xlabel("Température (K)")
            ax1.set_ylabel("|\Delta S_m| (J/kg.K)")
            ax1.legend()
            st.pyplot(fig1)

        with tab2:
            st.subheader("Courbe Universelle (Master Curve)")
            st.info("Si les courbes se superposent sur cet axe θ, le matériau suit une transition de phase du 2ème ordre.")
            
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for res in results_master:
                # Normalisation : DeltaS / DeltaS_max en fonction de theta
                ax2.plot(res['theta'], res['ds']/res['s_max'], label=f"{res['h']}T")
            
            ax2.set_xlabel("Température réduite (θ)")
            ax2.set_ylabel("$\Delta S_m / \Delta S_{max}$")
            ax2.set_xlim(-2, 2) # On zoom sur la zone critique
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        # Export
        final_df = pd.DataFrame({"T": T, "Theta": results_master[-1]['theta'], "DS_Norm": results_master[-1]['ds']/results_master[-1]['s_max']})
        st.download_button("📥 Télécharger les points de la Master Curve", to_excel(final_df), "MasterCurve_Data.xlsx")

    else:
        st.error("Format CSV incorrect. Colonnes attendues : T, M_1T, M_2T...")
