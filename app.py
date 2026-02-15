import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIGURATION PAGE =================
st.set_page_config(page_title="IA Magnétocalorique Expert", layout="wide")

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

# ================= BARRE LATÉRALE =================
with st.sidebar:
    st.header("⚙️ Configuration IA & Physique")
    nodes_m1 = st.slider("Neurones Modèle A (Principal)", 32, 256, 128, step=32)
    nodes_m2 = st.slider("Neurones Modèle B (Comparaison)", 32, 256, 64, step=32)
    st.divider()
    deltaT_tec = st.slider("Plage TEC (ΔT en K)", 1, 10, 3)
    st.info("Le Modèle A pilote les calculs thermodynamiques.")

# ================= ENTÊTE =================
col1, col2 = st.columns([1,5])
with col1:
    try:
        st.image("logo.png", width=110)
    except:
        st.markdown("### ISSAT")
with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Analyse Expert")
    st.markdown("**Élaboré par : DALHOUMI WALID**")

st.divider()

# ================= CHARGEMENT DES DONNÉES =================
file = st.file_uploader("Charger le fichier CSV (Colonnes: T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # ================= ENTRAÎNEMENT MODÈLE A =================
    with st.spinner('IA en cours d\'apprentissage...'):
        H_known = np.array([1, 2, 3])
        Tg, Hg = np.meshgrid(T, H_known)
        X = np.column_stack([Tg.ravel(), Hg.ravel()])
        y = M_matrix.T.ravel()

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(nodes_m1, nodes_m1), max_iter=5000, random_state=42)
        model.fit(X_scaled, y_scaled)

    # ================= PRÉDICTION & CALCULS =================
    st.subheader("🔮 Prédiction à Champ Personnalisé")
    H_user = st.number_input("Champ cible (Tesla)", 0.1, 10.0, 5.0, 0.5)

    X_u = scaler_X.transform(np.column_stack([T, np.full_like(T, H_user)]))
    M_u = scaler_y.inverse_transform(model.predict(X_u).reshape(-1,1)).ravel()

    # --- Thermodynamique ---
    dM_dT = [np.gradient(m, T) for m in [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_u]]
    deltaS = np.trapezoid(dM_dT, x=[1, 2, 3, H_user], axis=0)
    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # --- Paramètres Experts ---
    indices_arr = np.where(np.abs(deltaS) >= Smax/2)[0]
    if len(indices_arr) > 1:
        FWHM = T[indices_arr[-1]] - T[indices_arr[0]]
        RCP = Smax * FWHM
    else:
        FWHM, RCP = 0, 0
    
    RC = np.trapezoid(np.abs(deltaS), T)
    NRC = RCP / H_user if H_user != 0 else 0
    TEC = [np.mean(np.abs(deltaS)[(T >= t - deltaT_tec/2) & (T <= t + deltaT_tec/2)]) for t in T]
    TEC_max = np.max(TEC)

    # ================= AFFICHAGE MÉTRIQUES =================
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ΔS Max", f"{Smax:.4f}")
    m2.metric("RCP", f"{RCP:.2f}")
    m3.metric(f"TEC ({deltaT_tec}K)", f"{TEC_max:.4f}")
    m4.metric("NRC", f"{NRC:.2f}")
    m5.metric("Tc (K)", f"{Tc:.1f}")

    # ================= ONGLETS (TABS) =================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Courbes M & H/M", "❄️ Entropie & TEC", "🧲 Arrott Fit", "🧬 Comparaison 3D"])

    with tab1:
        st.subheader("📈 Analyse Magnétique Multi-Champs")
        col_g1, col_g2 = st.columns(2)
        champs_list = [1, 2, 3, H_user]
        couleurs = ['black', 'red', 'green', 'cyan']
        M_list = [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_u]

        with col_g1:
            fig_m, ax_m = plt.subplots(figsize=(5, 4))
            for m, h, c in zip(M_list, champs_list, couleurs):
                label = f"{h}T (Exp)" if h <= 3 else f"{h}T (IA)"
                ax_m.plot(T, m, label=label, color=c, lw=1.5)
            ax_m.set_xlabel("T (K)"); ax_m.set_ylabel("M (emu/g)")
            ax_m.set_title("Magnétisation M(T)"); ax_m.legend(fontsize='small'); st.pyplot(fig_m)

        with col_g2:
            fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
            for m, h, c in zip(M_list, champs_list, couleurs):
                mask = m > 0.1
                ax_hm.plot(T[mask], h / m[mask], color=c, lw=1.5)
            ax_hm.set_xlabel("T (K)"); ax_hm.set_ylabel("H/M (T.g/emu)")
            ax_hm.set_title("Susceptibilité Inverse H/M(T)"); st.pyplot(fig_hm)

        st.divider()
        st.subheader("🔍 Analyse de Curie-Weiss (Phase Paramagnétique)")
        mask_para = T > (Tc + 10)
        if any(mask_para) and len(T[mask_para]) > 5:
            T_fit = T[mask_para].reshape(-1, 1)
            Y_fit = (H_user / M_u[mask_para]).reshape(-1, 1)
            reg_cw = LinearRegression().fit(T_fit, Y_fit)
            pente = float(reg_cw.coef_[0][0])
            theta_p = -float(reg_cw.intercept_[0]) / pente
            mu_eff = np.sqrt(8 * (1/pente)) if pente > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Temp. Curie Para (θp) :** {theta_p:.1f} K")
            c2.info(f"**Moment Effectif (μeff) :** {mu_eff:.2f} μB")
            c3.success(f"**R² du Fit Para :** {reg_cw.score(T_fit, Y_fit):.4f}")
        else:
            st.warning("Plage de température insuffisante après Tc pour l'analyse Curie-Weiss.")

    with tab2:
        fig_th, ax_th = plt.subplots(figsize=(6, 3.5))
        ax_th.plot(T, np.abs(deltaS), label="|ΔS|", color='blue', lw=2)
        ax_th.plot(T, TEC, label=f"TEC({deltaT_tec}K)", color='orange', ls='--')
        ax_th.set_title("Variation d'Entropie vs Température"); ax_th.legend(); st.pyplot(fig_th)

    with tab3:
        st.subheader("Analyse de Transition & Fit Linéaire $y=ax+b$")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Arrott Plot (Banerjee Criterion)**")
            fig_ar, ax_ar = plt.subplots(figsize=(5, 4))
            mask = (M_u > 1e-6)
            X_f, Y_f = (M_u[mask]**2).reshape(-1, 1), (H_user / M_u[mask]).reshape(-1, 1)
            if len(X_f) > 1:
                reg = LinearRegression().fit(X_f, Y_f)
                ax_ar.scatter(X_f, Y_f, alpha=0.3, s=10)
                ax_ar.plot(X_f, reg.predict(X_f), color='red')
                ax_ar.set_xlabel("$M^2$"); ax_ar.set_ylabel("$H/M$"); st.pyplot(fig_ar)
                p = float(reg.coef_[0][0]); i = float(reg.intercept_[0])
                st.success(f"Équation : **y = {p:.4e}x + {i:.4f}**")
                st.info(f"Transition de **{'2ème' if p > 0 else '1er'} ordre** suggérée.")

        with c2:
            st.markdown("**Master Curve (Scaling Universel)**")
            if len(indices_arr) > 1:
                t_r1, t_r2 = T[indices_arr[0]], T[indices_arr[-1]]
                theta = np.where(T <= Tc, -(T-Tc)/(t_r1-Tc+1e-6), (T-Tc)/(t_r2-Tc+1e-6))
                fig_ms, ax_ms = plt.subplots(figsize=(5, 4))
                ax_ms.plot(theta, np.abs(deltaS)/Smax, color='green', lw=2)
                ax_ms.set_xlabel(r"$\theta$"); ax_ms.set_ylabel(r"$\Delta S / \Delta S_{max}$"); st.pyplot(fig_ms)

    with tab4:
        st.subheader("🧬 Comparaison Surfaces 3D (Modèle A vs B)")
        model_B = MLPRegressor(hidden_layer_sizes=(nodes_m2, nodes_m2), max_iter=3000, random_state=1).fit(X_scaled, y_scaled)
        H_sr = np.linspace(0.1, H_user, 30)
        Ts_g, Hs_g = np.meshgrid(T, H_sr)
        X_sf = scaler_X.transform(np.column_stack([Ts_g.ravel(), Hs_g.ravel()]))
        ZA = scaler_y.inverse_transform(model.predict(X_sf).reshape(-1,1)).reshape(len(H_sr), len(T))
        ZB = scaler_y.inverse_transform(model_B.predict(X_sf).reshape(-1,1)).reshape(len(H_sr), len(T))
        fig3d = go.Figure(data=[go.Surface(z=ZA, x=T, y=H_sr, colorscale='Viridis', name='A'),
                                go.Surface(z=ZB, x=T, y=H_sr, colorscale='Reds', opacity=0.4, showscale=False)])
        fig3d.update_layout(scene=dict(xaxis_title='T', yaxis_title='H', zaxis_title='M'), height=600)
        st.plotly_chart(fig3d, use_container_width=True)

    # ================= EXPORTS =================
    st.divider()
    df_ex = pd.DataFrame({"T":T, "M_pred":M_u, "DeltaS":deltaS})
    df_st = pd.DataFrame({"Paramètre":["Smax", "RCP", "TEC_max", "NRC", "Tc"], "Valeur":[Smax, RCP, TEC_max, NRC, Tc]})
    st.download_button("📥 Télécharger Résultats (Excel)", data=to_excel_full(df_ex, df_st), file_name="Magneto_Analyse_IA.xlsx")
