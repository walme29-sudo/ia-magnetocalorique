import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIGURATION PAGE =================
st.set_page_config(page_title="IA Magnétocalorique Pro", layout="wide", page_icon="🧲")

# ================= FONCTIONS TECHNIQUES =================
def to_excel_full(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data_Predictions', index=False)
        df_stats.to_excel(writer, sheet_name='Thermo_Params', index=False)
    return output.getvalue()

# ================= ENTÊTE =================
col1, col2 = st.columns([1,5])
with col1:
    st.markdown("## 🏛️ ISSAT") # Logo ou texte
with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Analyse Globale & Expert")
    st.markdown("**Développeur : DALHOUMI WALID** | *Analyse de Transition de Phase & Scaling*")

st.divider()

# ================= BARRE LATÉRALE =================
with st.sidebar:
    st.header("⚙️ Configuration IA")
    nodes_m1 = st.slider("Neurones Modèle A (Principal)", 32, 512, 256, step=32)
    nodes_m2 = st.slider("Neurones Modèle B (Comparaison)", 32, 256, 64, step=32)
    st.divider()
    st.subheader("Paramètres Physiques")
    deltaT_tec = st.slider("Fenêtre TEC (ΔT en K)", 1, 20, 5)
    st.info("Le TEC évalue la performance du matériau sur une plage de température réelle d'utilisation.")

# ================= CHARGEMENT DES DONNÉES =================
file = st.file_uploader("Charger le fichier CSV (Colonnes requises: T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    T = data["T"].values
    # Détection automatique des colonnes de magnétisation
    m_cols = [c for c in data.columns if 'M_' in c]
    M_matrix = data[m_cols].values
    H_known = np.array([float(c.split('_')[1].replace('T','')) for c in m_cols])

    # ================= ENTRAÎNEMENT IA =================
    with st.spinner('L\'IA analyse les propriétés magnétiques...'):
        T_grid, H_grid = np.meshgrid(T, H_known)
        X = np.column_stack([T_grid.ravel(), H_grid.ravel()])
        y = M_matrix.T.ravel()

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(nodes_m1, nodes_m1), max_iter=5000, random_state=42)
        model.fit(X_scaled, y_scaled)

    # ================= PRÉDICTION & THERMO =================
    H_user = st.number_input("Prédire pour un champ cible (Tesla)", 0.1, 15.0, 5.0, 0.5)
    X_user = scaler_X.transform(np.column_stack([T, np.full_like(T, H_user)]))
    M_user = scaler_y.inverse_transform(model.predict(X_user).reshape(-1,1)).ravel()

    # Calcul Delta S (Maxwell)
    # On calcule dM/dT pour chaque champ connu + le prédit
    dM_dT_list = [np.gradient(M_matrix[:, i], T) for i in range(len(H_known))]
    dM_dT_list.append(np.gradient(M_user, T))
    H_all = np.append(H_known, H_user)
    
    # Intégration par rapport à H
    deltaS = np.trapz(dM_dT_list, x=H_all, axis=0)
    abs_ds = np.abs(deltaS)
    Smax = np.max(abs_ds)
    Tc = T[np.argmax(abs_ds)]

    # RCP & TEC
    idx_half = np.where(abs_ds >= Smax/2)[0]
    fwhm = (T[idx_half[-1]] - T[idx_half[0]]) if len(idx_half) > 1 else 0
    rcp = Smax * fwhm
    
    # Calcul glissant du TEC
    tec_curve = []
    for t in T:
        mask = (T >= t - deltaT_tec/2) & (T <= t + deltaT_tec/2)
        tec_curve.append(np.mean(abs_ds[mask]) if any(mask) else 0)
    tec_max = np.max(tec_curve)

    # Exposant n(T) : DeltaS ~ H^n
    # On fait une régression linéaire sur log|DeltaS| vs log|H| pour chaque T
    n_T = []
    for j in range(len(T)):
        # On estime DeltaS pour chaque H_known pour calculer n localement
        ds_local = [np.trapz(dM_dT_list[:i+1], x=H_all[:i+1], axis=0)[j] for i in range(len(H_all))]
        log_h = np.log(H_all)
        log_ds = np.log(np.abs(ds_local) + 1e-9)
        n_T.append(np.polyfit(log_h, log_ds, 1)[0])

    # ================= AFFICHAGE MÉTRIQUES =================
    cols_m = st.columns(5)
    cols_m[0].metric("ΔS Max", f"{Smax:.4f} J/kgK")
    cols_m[1].metric("RCP", f"{rcp:.2f}")
    cols_m[2].metric(f"TEC ({deltaT_tec}K)", f"{tec_max:.4f}")
    cols_m[3].metric("Tc (Curie)", f"{Tc:.1f} K")
    cols_m[4].metric("n (à Tc)", f"{n_T[np.argmax(abs_ds)]:.3f}")

    # ================= ONGLETS =================
    t1, t2, t3, t4 = st.tabs(["📉 Aimantation", "❄️ Thermodynamique", "🧬 Transition & Master", "🌐 Surface 3D"])

    with t1:
        fig_m = go.Figure()
        for i, h in enumerate(H_known):
            fig_m.add_scatter(x=T, y=M_matrix[:,i], name=f"{h}T Exp", mode='markers')
        fig_m.add_scatter(x=T, y=M_user, name=f"{H_user}T IA", line=dict(color='red', width=3))
        fig_m.update_layout(title="Cycles de Magnétisation M(T)", xaxis_title="T (K)", yaxis_title="M (emu/g)")
        st.plotly_chart(fig_m, use_container_width=True)

    with t2:
        
        c_a, c_b = st.columns(2)
        with c_a:
            fig_ds, ax_ds = plt.subplots()
            ax_ds.fill_between(T, 0, abs_ds, alpha=0.2, color='blue')
            ax_ds.plot(T, abs_ds, label="|ΔS|", color='blue')
            ax_ds.plot(T, tec_curve, label=f"TEC ({deltaT_tec}K)", color='orange', linestyle='--')
            ax_ds.set_ylabel("|ΔS| (J/kgK)"); ax_ds.legend(); st.pyplot(fig_ds)
        with c_b:
            fig_n, ax_n = plt.subplots()
            ax_n.plot(T, n_T, color='green', lw=2)
            ax_n.axvline(Tc, color='red', ls='--', label=f"Tc={Tc}K")
            ax_n.set_ylabel("Exposant n"); ax_n.set_xlabel("T (K)"); ax_n.legend(); st.pyplot(fig_n)

    with t3:
        
        c1_t3, c2_t3 = st.columns(2)
        with c1_t3:
            st.markdown("**Critère de Banerjee (Arrott Plot)**")
            fig_arr, ax_arr = plt.subplots()
            ax_arr.plot(M_user**2, H_user/(M_user + 1e-9), 'purple')
            ax_arr.set_xlabel("$M^2$"); ax_arr.set_ylabel("$H/M$"); st.pyplot(fig_arr)
        with c2_t3:
            st.markdown("**Master Curve (Universal Scaling)**")
            if fwhm > 0:
                theta = np.where(T <= Tc, -(T-Tc)/(T[idx_half[0]]-Tc), (T-Tc)/(T[idx_half[-1]]-Tc))
                fig_mst, ax_mst = plt.subplots()
                ax_mst.plot(theta, abs_ds/Smax, color='black', lw=2)
                ax_mst.set_xlabel(r"$\theta$"); ax_mst.set_ylabel(r"$\Delta S / \Delta S_{max}$"); st.pyplot(fig_mst)

    with t4:
        st.subheader("Comparaison Topologique des Modèles")
        H_range = np.linspace(0.1, H_user, 20)
        T_mesh, H_mesh = np.meshgrid(T, H_range)
        X_surf = scaler_X.transform(np.column_stack([T_mesh.ravel(), H_mesh.ravel()]))
        
        # Modèle B pour comparaison
        model_B = MLPRegressor(hidden_layer_sizes=(nodes_m2, nodes_m2), max_iter=2000, random_state=1).fit(X_scaled, y_scaled)
        
        Z_A = scaler_y.inverse_transform(model.predict(X_surf).reshape(-1,1)).reshape(len(H_range), len(T))
        Z_B = scaler_y.inverse_transform(model_B.predict(X_surf).reshape(-1,1)).reshape(len(H_range), len(T))
        
        fig3d = go.Figure(data=[
            go.Surface(z=Z_A, x=T, y=H_range, colorscale='Viridis', name='Modèle A'),
            go.Surface(z=Z_B, x=T, y=H_range, colorscale='Reds', opacity=0.4, showscale=False, name='Modèle B')
        ])
        fig3d.update_layout(scene=dict(xaxis_title='T (K)', yaxis_title='H (T)', zaxis_title='M (emu/g)'))
        st.plotly_chart(fig3d, use_container_width=True)

    # ================= EXPORTS =================
    st.divider()
    df_exp = pd.DataFrame({"T":T, "M_IA":M_user, "DeltaS":deltaS, "n_local":n_T, "TEC":tec_curve})
    df_stats = pd.DataFrame({"Paramètre":["Smax", "RCP", "TEC_max", "Tc", "n_at_Tc"], "Valeur":[Smax, rcp, tec_max, Tc, n_T[np.argmax(abs_ds)]]})
    st.download_button("📥 Télécharger l'Analyse Complète (Excel)", data=to_excel_full(df_exp, df_stats), file_name="Rapport_Magneto_Expert.xlsx")

else:
    st.info("👋 Bienvenue ! Veuillez charger votre fichier CSV pour commencer l'analyse magnétocalorique experte.")
