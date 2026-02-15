import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIGURATION PAGE =================
st.set_page_config(page_title="IA Magnétocalorique Pro", layout="wide")

# ================= FONCTIONS D'EXPORT =================
def to_excel_full(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data_Predictions', index=False)
        df_stats.to_excel(writer, sheet_name='Thermo_Params', index=False)
    return output.getvalue()

def plot_to_pdf(fig):
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches='tight')
    return output.getvalue()

# ================= BARRE LATÉRALE (SIDEBAR) =================
with st.sidebar:
    st.header("⚙️ Configuration IA")
    st.write("Ajustez les architectures des réseaux de neurones.")

    st.subheader("Modèle A (Principal)")
    nodes_m1 = st.slider("Neurones (Couches 1 & 2)", 32, 256, 128, step=32)

    st.divider()

    st.subheader("Modèle B (Comparaison)")
    nodes_m2 = st.slider("Neurones (Comparaison 3D)", 32, 256, 64, step=32)

    st.info("Le modèle A est utilisé pour tous les calculs (ΔS, RCP, RC, n).")

# ================= ENTÊTE =================
col1, col2 = st.columns([1,5])
with col1:
    try:
        st.image("logo.png", width=110)
    except:
        st.markdown("### ISSAT")
with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Analyse Globale")
    st.markdown("**Élaboré par : DALHOUMI WALID**")

st.divider()

# ================= CHARGEMENT DES DONNÉES =================
file = st.file_uploader("Charger le fichier CSV (Colonnes: T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    required = ["T","M_1T","M_2T","M_3T"]

    if not all(col in data.columns for col in required):
        st.error(f"Le CSV doit contenir les colonnes : {', '.join(required)}")
        st.stop()

    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # ================= ENTRAÎNEMENT MODÈLE A =================
    with st.spinner('Entraînement de l\'IA en cours...'):
        H_values = np.array([1, 2, 3])
        T_grid_in, H_grid_in = np.meshgrid(T, H_values)

        X = np.column_stack([T_grid_in.ravel(), H_grid_in.ravel()])
        y = M_matrix.T.ravel()

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        # Modèle A basé sur le slider nodes_m1
        model = MLPRegressor(hidden_layer_sizes=(nodes_m1, nodes_m1),
                             activation='relu', solver='adam', max_iter=5000, random_state=42)
        model.fit(X_scaled, y_scaled)

    # ================= PRÉDICTION UTILISATEUR =================
    st.subheader("🔮 Prédiction à Champ Personnalisé")
    H_user = st.number_input("Champ magnétique cible (Tesla)", 0.1, 10.0, 5.0, 0.5)

    X_user = np.column_stack([T, np.full_like(T, H_user)])
    X_user_scaled = scaler_X.transform(X_user)
    M_user = scaler_y.inverse_transform(model.predict(X_user_scaled).reshape(-1,1)).ravel()

    # ================= CALCULS THERMODYNAMIQUES (Modèle A) =================
    # Delta S
    dM1, dM2, dM3, dM_user = [np.gradient(m, T) for m in [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_user]]
    deltaS = np.trapezoid([dM1, dM2, dM3, dM_user], x=[1, 2, 3, H_user], axis=0)

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # RCP & RC
    indices = np.where(np.abs(deltaS) >= Smax/2)[0]
    RCP = Smax * (T[indices[-1]] - T[indices[0]]) if len(indices) > 1 else 0
    RC = np.trapezoid(np.abs(deltaS), T)

    # Exposant n(T)
    H_list = [1, 2, 3, H_user]
    n_T = []
    for i in range(len(T)):
        y_vals = np.array([np.abs(dM1[i]), np.abs(dM2[i]), np.abs(dM3[i]), np.abs(dM_user[i])])
        if np.all(y_vals > 0):
            n_T.append(np.polyfit(np.log(H_list), np.log(y_vals), 1)[0])
        else: n_T.append(np.nan)
    n_exponent = n_T[np.argmin(np.abs(T-Tc))]

    # ================= AFFICHAGE MÉTRIQUES =================
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ΔS Max", f"{Smax:.4f}")
    m2.metric("RCP", f"{RCP:.2f}")
    m3.metric("RC", f"{RC:.2f}")
    m4.metric("n (at Tc)", f"{n_exponent:.3f}")
    m5.metric("Tc (K)", f"{Tc:.1f}")

    # ================= ONGLETS (TABS) =================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Aimantation", "❄️ Thermodynamique", "🧲 Arrott & Master", "🧬 Comparaison 3D"])

        with tab1:
            # ================= ONGLETS (TABS) =================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Aimantation", "❄️ Thermodynamique", "🧲 Arrott & Master", "🧬 Comparaison 3D"])

    with tab1:
        df_m = pd.DataFrame({
            "1T": M_matrix[:,0],
            "2T": M_matrix[:,1],
            "3T": M_matrix[:,2],
            f"{H_user:.1f}T (IA)": M_user
        }, index=T)
        st.line_chart(df_m)

    with tab2:
        c_a, c_b = st.columns(2)
        with c_a:
            fig_ds, ax_ds = plt.subplots(figsize=(5,3.5))
            ax_ds.plot(T, deltaS, color='blue', lw=2)
            ax_ds.set_title("Variation d'Entropie ΔS")
            ax_ds.set_xlabel("T (K)")
            ax_ds.set_ylabel("ΔS")
            st.pyplot(fig_ds)
        with c_b:
            fig_n, ax_n = plt.subplots(figsize=(5,3.5))
            ax_n.plot(T, n_T, color='green')
            ax_n.axvline(Tc, color='red', ls='--')
            ax_n.set_title("Exposant n(T)")
            ax_n.set_xlabel("T (K)")
            ax_n.set_ylabel("n")
            st.pyplot(fig_n)

    with tab3:
        st.subheader("Analyse de Transition de Phase")
        col_tab3_1, col_tab3_2 = st.columns(2)
        col_t3_1, col_t3_2 = st.columns(2)
        H_plot_list = [1, 2, 3, H_user]
        M_plot_list = [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_user]

        # -------- Arrott Plot --------
        with col_tab3_1:
        with col_t3_1:
            st.markdown("**Arrott Plot (H/M vs M²)**")
            fig_arrott, ax_arrott = plt.subplots(figsize=(5, 4))
            H_plot_list = [1, 2, 3, H_user]
            M_plot_list = [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_user]
            for h_val, m_val in zip(H_plot_list, M_plot_list):
                mask = (m_val != 0)
                ax_arrott.plot(m_val[mask]**2, h_val / m_val[mask], label=f"{h_val:.1f}T")
            ax_arrott.set_xlabel("$M^2$")
            ax_arrott.set_ylabel("$H/M$")
            ax_arrott.legend()
            st.pyplot(fig_arrott)

        # -------- Master Curve --------
        with col_tab3_2:
            fig_arr, ax_arr = plt.subplots(figsize=(5, 4))
            for h_v, m_v in zip(H_plot_list, M_plot_list):
                mask = (m_v != 0)
                ax_arr.plot(m_v[mask]**2, h_v / m_v[mask], label=f"{h_v:.1f}T")
            ax_arr.set_xlabel("$M^2$")
            ax_arr.set_ylabel("$H/M$")
            ax_arr.legend()
            st.pyplot(fig_arr)

        with col_t3_2:
            st.markdown("**Master Curve (Scaling)**")
            fig_master, ax_master = plt.subplots(figsize=(5, 4))
            idx_half = np.where(np.abs(deltaS) >= Smax/2)[0]
            if len(idx_half) > 1:
                t_r1, t_r2 = T[idx_half[0]], T[idx_half[-1]]
                for h_val, m_val in zip(H_plot_list, M_plot_list):
                    ds_l = np.abs(np.gradient(m_val, T))
            fig_mst, ax_mst = plt.subplots(figsize=(5, 4))
            idx_h = np.where(np.abs(deltaS) >= Smax/2)[0]
            if len(idx_h) > 1:
                t_r1, t_r2 = T[idx_h[0]], T[idx_h[-1]]
                for h_v, m_v in zip(H_plot_list, M_plot_list):
                    ds_l = np.abs(np.gradient(m_v, T))
                    ds_m_l = np.max(ds_l) if np.max(ds_l) != 0 else 1
                    theta = np.where(T <= Tc, -(T - Tc) / (t_r1 - Tc + 1e-6), (T - Tc) / (t_r2 - Tc + 1e-6))
                    ax_master.plot(theta, ds_l/ds_m_l, label=f"{h_val:.1f}T")
                ax_master.set_xlabel("$\\theta$")
                ax_master.set_ylabel("$\\Delta S / \\Delta S_{max}$")
                ax_master.legend()
                st.pyplot(fig_master)
                    ax_mst.plot(theta, ds_l/ds_m_l, label=f"{h_v:.1f}T")
                ax_mst.set_xlabel("$\\theta$")
                ax_mst.set_ylabel("$\\Delta S / \\Delta S_{max}$")
                ax_mst.legend()
                st.pyplot(fig_mst)
            else:
                st.warning("Données insuffisantes pour Master Curve")
                st.warning("Données insuffisantes")

    with tab4:
        st.subheader("🧬 Comparaison des Surfaces 3D (Modèle A vs B)")
        # Entraînement Modèle B
        model_B = MLPRegressor(hidden_layer_sizes=(nodes_m2, nodes_m2), 
                             activation='relu', solver='adam', max_iter=5000, random_state=1)
        model_B.fit(X_scaled, y_scaled)

        # Grille de surface
        H_surf_range = np.linspace(0.1, H_user, 35)
        T_surf_grid, H_surf_grid = np.meshgrid(T, H_surf_range)
        X_surf_flat = np.column_stack([T_surf_grid.ravel(), H_surf_grid.ravel()])
        X_surf_flat_scaled = scaler_X.transform(X_surf_flat)

        # Prédictions
        Z_A = scaler_y.inverse_transform(model.predict(X_surf_flat_scaled).reshape(-1,1)).reshape(len(H_surf_range), len(T))
        Z_B = scaler_y.inverse_transform(model_B.predict(X_surf_flat_scaled).reshape(-1,1)).reshape(len(H_surf_range), len(T))

        # Plotly
        H_sr = np.linspace(0.1, H_user, 35)
        Ts_g, Hs_g = np.meshgrid(T, H_sr)
        X_sf = np.column_stack([Ts_g.ravel(), Hs_g.ravel()])
        X_sf_s = scaler_X.transform(X_sf)
        ZA = scaler_y.inverse_transform(model.predict(X_sf_s).reshape(-1,1)).reshape(len(H_sr), len(T))
        ZB = scaler_y.inverse_transform(model_B.predict(X_sf_s).reshape(-1,1)).reshape(len(H_sr), len(T))
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Surface(z=Z_A, x=T, y=H_surf_range, colorscale='Viridis', name='Modèle A', opacity=0.9))
        fig_3d.add_trace(go.Surface(z=Z_B, x=T, y=H_surf_range, colorscale='Reds', name='Modèle B', opacity=0.5, showscale=False))
        fig_3d.update_layout(scene=dict(xaxis_title='T (K)', yaxis_title='H (T)', zaxis_title='M'), 
                             height=700, margin=dict(l=0, r=0, b=0, t=40))
        fig_3d.add_trace(go.Surface(z=ZA, x=T, y=H_sr, colorscale='Viridis', name='Modèle A', opacity=0.9))
        fig_3d.add_trace(go.Surface(z=ZB, x=T, y=H_sr, colorscale='Reds', name='Modèle B', opacity=0.5, showscale=False))
        fig_3d.update_layout(scene=dict(xaxis_title='T (K)', yaxis_title='H (T)', zaxis_title='M'), height=700)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ================= EXPORTS FINAUX =================
    st.divider()
    st.subheader("📥 Exportation des Résultats")
    df_export = pd.DataFrame({"T":T, "M_1T":M_matrix[:,0], f"M_{H_user}T":M_user, "DeltaS":deltaS, "n_T":n_T})
    df_stats = pd.DataFrame({"Paramètre":["Smax", "RCP", "RC", "n_Tc", "Tc"], "Valeur":[Smax, RCP, RC, n_exponent, Tc]})
    st.download_button("Excel Complet", data=to_excel_full(df_export, df_stats), file_name="Resultats_IA.xlsx")

else:
    st.info("Veuillez charger un fichier CSV pour démarrer l'analyse.")
