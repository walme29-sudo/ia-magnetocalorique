import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIGURATION PAGE =================
st.set_page_config(page_title="IA Magnétocalorique - Analyse Totale", layout="wide")

def to_excel(df_list, sheet_names):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()

# ================= HEADER =================
st.title("🧲 Plateforme d'Analyse Magnétocalorique Intégrale")
st.markdown("**Développeur : DALHOUMI WALID** | Système d'Analyse par Intelligence Artificielle")
st.divider()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration")
    file = st.file_uploader("Charger fichier CSV (T, M_1T, M_2T...)", type=["csv"])
    nodes = st.slider("Puissance IA (Neurones)", 64, 512, 256)
    h_target = st.number_input("Champ Cible pour Analyse (T)", 0.1, 10.0, 5.0)
    cp_const = st.number_input("Chaleur spécifique Cp (J/kg.K)", 100, 1000, 450)
    st.divider()
    st.info("Le fichier doit contenir une colonne 'T' et des colonnes 'M_xT'.")

if file:
    df = pd.read_csv(file).dropna()
    cols = df.columns.tolist()
    t_col = next((c for c in cols if c.lower() in ['t', 'temp']), cols[0])
    m_cols = [c for c in cols if 'M_' in c]
    
    if len(m_cols) >= 2:
        T = df[t_col].values
        M_matrix = df[m_cols].values
        H_vals = np.array([float(''.join(c for c in col if c.isdigit() or c=='.')) for col in m_cols])

        # --- ENTRAÎNEMENT IA ---
        with st.spinner("L'IA apprend la physique du matériau..."):
            T_g, H_g = np.meshgrid(T, H_vals)
            X = np.column_stack([T_g.ravel(), H_g.ravel()])
            y = M_matrix.T.ravel()
            
            sc_X, sc_y = StandardScaler(), StandardScaler()
            X_s = sc_X.fit_transform(X)
            y_s = sc_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            model = MLPRegressor(hidden_layer_sizes=(nodes, nodes), max_iter=3000, random_state=42)
            model.fit(X_s, y_s)

        # --- PRÉDICTIONS MULTI-CHAMPS POUR ANALYSE ---
        fields = sorted(list(set(list(H_vals) + [h_target])))
        results = {}

        for h in fields:
            X_p = sc_X.transform(np.column_stack([T, np.full_like(T, h)]))
            M_p = sc_y.inverse_transform(model.predict(X_p).reshape(-1, 1)).ravel()
            dM_dT = np.gradient(M_p, T)
            # Delta S par intégration de Maxwell
            ds = np.abs(np.trapezoid([np.gradient(scaler_y.inverse_transform(model.predict(scaler_X.transform(np.column_stack([T, np.full_like(T, h_i)]))).reshape(-1,1)).ravel(), T) for h_i in np.linspace(0, h, 10)], x=np.linspace(0, h, 10), axis=0))
            
            s_max = np.max(ds)
            tc = T[np.argmax(ds)]
            # Delta Tad = -(T/Cp) * DeltaS
            dt_ad = (T * ds) / cp_const
            
            # Master Curve préparation
            idx_r = np.where(ds >= s_max/2)[0]
            tr1, tr2 = (T[idx_r[0]], T[idx_r[-1]]) if len(idx_r) > 1 else (T[0], T[-1])
            theta = np.where(T <= tc, -(T - tc) / (tr1 - tc + 1e-5), (T - tc) / (tr2 - tc + 1e-5))
            
            results[h] = {'M': M_p, 'dM': dM_dT, 'ds': ds, 'dt_ad': dt_ad, 's_max': s_max, 'tc': tc, 'theta': theta}

        # ================= AFFICHAGE DES ONGLETS =================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Magnétisme (M, dM/dT)", 
            "❄️ Thermodynamique (ΔS, ΔTad)", 
            "🧬 Critères (Arrott, n)", 
            "🌐 Master Curve", 
            "📊 Données & Export"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            fig_m, ax_m = plt.subplots()
            for h in fields: ax_m.plot(T, results[h]['M'], label=f"{h}T")
            ax_m.set_title("Magnétisation M(T)"); ax_m.set_xlabel("T (K)"); ax_m.legend(); c1.pyplot(fig_m)
            
            fig_dm, ax_dm = plt.subplots()
            ax_dm.plot(T, results[h_target]['dM'], color='green')
            ax_dm.set_title(f"Dérivée dM/dT à {h_target}T"); ax_dm.set_xlabel("T (K)"); c2.pyplot(fig_dm)

        with tab2:
            c3, c4 = st.columns(2)
            fig_ds, ax_ds = plt.subplots()
            ax_ds.fill_between(T, 0, results[h_target]['ds'], color='blue', alpha=0.3)
            ax_ds.plot(T, results[h_target]['ds'], color='blue')
            ax_ds.set_title(f"Entropie |ΔSm| à {h_target}T"); ax_ds.set_ylabel("J/kg.K"); c3.pyplot(fig_ds)
            
            fig_dt, ax_dt = plt.subplots()
            ax_dt.fill_between(T, 0, results[h_target]['dt_ad'], color='red', alpha=0.3)
            ax_dt.plot(T, results[h_target]['dt_ad'], color='red')
            ax_dt.set_title(f"ΔT Adiabatique à {h_target}T"); ax_dt.set_ylabel("K"); c4.pyplot(fig_dt)

        with tab3:
            c5, c6 = st.columns(2)
            # Arrott Plot
            fig_ar, ax_ar = plt.subplots()
            ax_ar.plot(results[h_target]['M']**2, h_target/(results[h_target]['M']+1e-9))
            ax_ar.set_title("Arrott Plot (M² vs H/M)"); c5.pyplot(fig_ar)
            
            # Exposant n
            n_T = np.gradient(np.log(results[h_target]['ds']+1e-9), np.log(h_target))
            fig_n, ax_n = plt.subplots()
            ax_n.plot(T, n_T, color='orange')
            ax_n.axhline(0.66, ls='--', color='black', label='Champ Moyen')
            ax_n.set_title("Exposant Local n(T)"); ax_n.legend(); c6.pyplot(fig_n)

        with tab4:
            st.subheader("Analyse de Scaling Universel")
            fig_mst, ax_mst = plt.subplots(figsize=(10,5))
            for h in fields:
                ax_mst.plot(results[h]['theta'], results[h]['ds']/results[h]['s_max'], label=f"{h}T")
            ax_mst.set_xlim(-2, 2); ax_mst.set_xlabel("θ (Température réduite)"); ax_mst.set_ylabel("ΔS / ΔSmax")
            ax_mst.legend(); ax_mst.grid(True); st.pyplot(fig_mst)

        with tab5:
            res_df = pd.DataFrame({"T": T, "M_IA": results[h_target]['M'], "DeltaS": results[h_target]['ds'], "DeltaTad": results[h_target]['dt_ad']})
            st.dataframe(res_df)
            st.download_button("📥 Télécharger Rapport Complet Excel", to_excel([res_df], ["Resultats"]), "Analyse_Expert_Complet.xlsx")

else:
    st.info("👋 Charge ton CSV pour voir toutes les courbes s'afficher !")

