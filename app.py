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

# ================= FONCTIONS TECHNIQUES =================
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
    nodes_m1 = st.slider("Neurones Modèle A (Principal)", 32, 256, 128, step=32)
    nodes_m2 = st.slider("Neurones Modèle B (Comparaison)", 32, 256, 64, step=32)
    st.divider()
    st.subheader("Paramètres Physiques")
    deltaT_tec = st.slider("Plage TEC (ΔT en K)", 1, 10, 3)

# ================= ENTÊTE =================
col1, col2 = st.columns([1,5])
with col1:
    try:
        st.image("logo.png", width=110)
    except:
        st.markdown("### ISSAT")
with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Analyse Globale & Expert")
    st.markdown("**Élaboré par : DALHOUMI WALID**")

st.divider()

# ================= CHARGEMENT DES DONNÉES =================
file = st.file_uploader("Charger le fichier CSV (Colonnes: T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # ================= ENTRAÎNEMENT MODÈLE A =================
    with st.spinner('Entraînement de l\'IA en cours...'):
        H_values = np.array([1, 2, 3])
        T_grid_in, H_grid_in = np.meshgrid(T, H_values)
        X = np.column_stack([T_grid_in.ravel(), H_grid_in.ravel()])
        y = M_matrix.T.ravel()

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(nodes_m1, nodes_m1), max_iter=5000, random_state=42)
        model.fit(X_scaled, y_scaled)

    # ================= PRÉDICTION UTILISATEUR =================
    st.subheader("🔮 Prédiction à Champ Personnalisé")
    H_user = st.number_input("Champ magnétique cible (Tesla)", 0.1, 10.0, 5.0, 0.5)

    X_user = scaler_X.transform(np.column_stack([T, np.full_like(T, H_user)]))
    M_user = scaler_y.inverse_transform(model.predict(X_user).reshape(-1,1)).ravel()

    # ================= CALCULS THERMODYNAMIQUES =================
    # Delta S
    dM_dT = [np.gradient(m, T) for m in [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_user]]
    deltaS = np.trapezoid(dM_dT, x=[1, 2, 3, H_user], axis=0)
    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # RCP & RC
    idx_half = np.where(np.abs(deltaS) >= Smax/2)[0]
    FWHM = (T[idx_half[-1]] - T[idx_half[0]]) if len(idx_half) > 1 else 0
    RCP = Smax * FWHM
    RC = np.trapezoid(np.abs(deltaS), T)

    # TEC & NRC
    TEC = [np.mean(np.abs(deltaS)[(T >= t - deltaT_tec/2) & (T <= t + deltaT_tec/2)]) for t in T]
    TEC_max = np.max(TEC)
    NRC = RCP / H_user if H_user != 0 else 0

    # n(T)
    H_list = [1, 2, 3, H_user]
    n_T = []
    for i in range(len(T)):
        y_vals = np.array([np.abs(d[i]) for d in dM_dT])
        n_T.append(np.polyfit(np.log(H_list), np.log(y_vals + 1e-9), 1)[0])
    n_exponent = n_T[np.argmin(np.abs(T-Tc))]

    # ================= AFFICHAGE MÉTRIQUES =================
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ΔS Max", f"{Smax:.4f}")
    m2.metric("RCP", f"{RCP:.2f}")
    m3.metric(f"TEC ({deltaT_tec}K)", f"{TEC_max:.4f}")
    m4.metric("n (at Tc)", f"{n_exponent:.3f}")
    m5.metric("Tc (K)", f"{Tc:.1f}")

    # ================= ONGLETS (TABS) =================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Aimantation", "❄️ Thermodynamique", "🧲 Arrott & Master", "🧬 Comparaison 3D"])

    with tab1:
        df_m = pd.DataFrame({"1T": M_matrix[:,0], "3T": M_matrix[:,2], f"{H_user}T (IA)": M_user}, index=T)
        st.line_chart(df_m)

    with tab2:
        c_a, c_b = st.columns(2)
        with c_a:
            fig_ds, ax_ds = plt.subplots(figsize=(5,3.5))
            ax_ds.plot(T, np.abs(deltaS), label="|ΔS|", color='blue')
            ax_ds.plot(T, TEC, label=f"TEC({deltaT_tec}K)", color='orange', ls='--')
            ax_ds.set_title("Entropie & TEC")
            ax_ds.legend(); st.pyplot(fig_ds)
        with c_b:
            fig_n, ax_n = plt.subplots(figsize=(5,3.5))
            ax_n.plot(T, n_T, color='green')
            ax_n.axvline(Tc, color='red', ls='--')
            ax_n.set_title("Exposant n(T)"); st.pyplot(fig_n)

    with tab3:
        col_t3_1, col_t3_2 = st.columns(2)
        with col_t3_1:
            st.markdown("**Banerjee Criterion (Arrott Plot)**")
            fig_arr, ax_arr = plt.subplots(figsize=(5, 4))
            ax_arr.plot(M_user**2, H_user/(M_user + 1e-9))
            ax_arr.set_xlabel("$M^2$"); ax_arr.set_ylabel("$H/M$"); st.pyplot(fig_arr)
            pente = np.polyfit(M_user**2, H_user/(M_user+1e-9), 1)[0]
            st.write("Ordre suggéré :", "**2ème**" if pente > 0 else "**1er**")
        with col_t3_2:
            st.markdown("**Master Curve (Scaling)**")
            if len(idx_half) > 1:
                t_r1, t_r2 = T[idx_half[0]], T[idx_half[-1]]
                fig_mst, ax_mst = plt.subplots(figsize=(5, 4))
                theta = np.where(T <= Tc, -(T-Tc)/(t_r1-Tc+1e-6), (T-Tc)/(t_r2-Tc+1e-6))
                ax_mst.plot(theta, np.abs(deltaS)/Smax, label=f"{H_user}T")
                ax_mst.set_xlabel("$\\theta$"); ax_mst.set_ylabel("$\\Delta S / \\Delta S_{max}$"); st.pyplot(fig_mst)

    with tab4:
        st.subheader("🧬 Comparaison Surfaces 3D")
        model_B = MLPRegressor(hidden_layer_sizes=(nodes_m2, nodes_m2), max_iter=3000, random_state=1).fit(X_scaled, y_scaled)
        H_sr = np.linspace(0.1, H_user, 30)
        Ts_g, Hs_g = np.meshgrid(T, H_sr)
        X_sf = scaler_X.transform(np.column_stack([Ts_g.ravel(), Hs_g.ravel()]))
        ZA = scaler_y.inverse_transform(model.predict(X_sf).reshape(-1,1)).reshape(len(H_sr), len(T))
        ZB = scaler_y.inverse_transform(model_B.predict(X_sf).reshape(-1,1)).reshape(len(H_sr), len(T))
        fig3d = go.Figure(data=[go.Surface(z=ZA, x=T, y=H_sr, colorscale='Viridis', name='Modèle A'),
                                go.Surface(z=ZB, x=T, y=H_sr, colorscale='Reds', opacity=0.4, name='Modèle B', showscale=False)])
        st.plotly_chart(fig3d, use_container_width=True)

    # ================= EXPORTS =================
    st.divider()
    df_export = pd.DataFrame({"T":T, "M_pred":M_user, "DeltaS":deltaS, "n_T":n_T})
    df_stats = pd.DataFrame({"Paramètre":["Smax", "RCP", "TEC_max", "NRC", "Tc"], "Valeur":[Smax, RCP, TEC_max, NRC, Tc]})
    st.download_button("📥 Export Excel", data=to_excel_full(df_export, df_stats), file_name="Magnetocaloric_Expert.xlsx")
else:
    st.info("Veuillez charger un fichier CSV pour démarrer.")
