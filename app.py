import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIG PAGE =================
st.set_page_config(page_title="IA Magnétocalorique Expert", layout="wide")

# ================= FONCTIONS TECHNIQUES =================
def to_excel_full(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data_Predictions', index=False)
        df_stats.to_excel(writer, sheet_name='Thermo_Params', index=False)
    return output.getvalue()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration IA")
    nodes_m1 = st.slider("Neurones Modèle A", 32, 256, 128, step=32)
    nodes_m2 = st.slider("Neurones Modèle B", 32, 256, 64, step=32)
    st.divider()
    st.subheader("Paramètres Physiques")
    deltaT_tec = st.slider("Plage TEC (ΔT en K)", 1, 10, 3)

# ================= HEADER =================
st.markdown("## 🧲 IA Magnétocalorique Expert : M(T,H) & Thermodynamique")
st.markdown("**Analyse complète : ΔS, RCP, RC, TEC, NRC & Arrott**")
st.divider()

file = st.file_uploader("Charger CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # --- Entraînement Modèle A ---
    with st.spinner('Entraînement IA...'):
        H_known = np.array([1, 2, 3])
        Tg, Hg = np.meshgrid(T, H_known)
        X = np.column_stack([Tg.ravel(), Hg.ravel()])
        y = M_matrix.T.ravel()
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_sc = scaler_X.fit_transform(X)
        y_sc = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
        
        model = MLPRegressor(hidden_layer_sizes=(nodes_m1, nodes_m1), max_iter=5000, random_state=42)
        model.fit(X_sc, y_sc)

    # --- Prédiction & Thermodynamique ---
    H_user = st.sidebar.number_input("Champ Cible (T)", 0.1, 10.0, 5.0)
    X_u = scaler_X.transform(np.column_stack([T, np.full_like(T, H_user)]))
    M_u = scaler_y.inverse_transform(model.predict(X_u).reshape(-1,1)).ravel()

    # Calcul ΔS
    dM_dT = [np.gradient(m, T) for m in [M_matrix[:,0], M_matrix[:,1], M_matrix[:,2], M_u]]
    deltaS = np.trapezoid(dM_dT, x=[1, 2, 3, H_user], axis=0)
    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # --- NOUVEAUX PARAMÈTRES (TEC, NRC, RC) ---
    # 1. RC (Refrigerant Capacity)
    RC = np.trapezoid(np.abs(deltaS), T)
    
    # 2. RCP (Relative Cooling Power)
    idx_half = np.where(np.abs(deltaS) >= Smax/2)[0]
    FWHM = (T[idx_half[-1]] - T[idx_half[0]]) if len(idx_half) > 1 else 0
    RCP = Smax * FWHM
    
    # 3. TEC (Temperature Averaged Entropy Change)
    TEC = []
    for t_c in T:
        mask = (T >= t_c - deltaT_tec/2) & (T <= t_c + deltaT_tec/2)
        TEC.append(np.mean(np.abs(deltaS[mask])) if any(mask) else 0)
    TEC_max = np.max(TEC)
    
    # 4. NRC (Normalized RCP)
    NRC = RCP / H_user if H_user != 0 else 0

    # --- AFFICHAGE MÉTRIQUES ---
    st.subheader(f"📊 Paramètres Thermodynamiques à {H_user}T")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ΔS Max (J/kg·K)", f"{Smax:.4f}")
    m2.metric("RCP (J/kg)", f"{RCP:.2f}")
    m3.metric(f"TEC({deltaT_tec}K)", f"{TEC_max:.4f}")
    m4.metric("NRC (J/kg·T)", f"{NRC:.2f}")
    m5.metric("Tc (K)", f"{Tc:.1f}")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Aimantation", "❄️ Entropie & TEC", "🧲 Arrott & Master", "🧬 Comparaison 3D"])

    with tab1:
        st.line_chart(pd.DataFrame({f"{H_user}T": M_u, "3T": M_matrix[:,2]}, index=T))

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(T, np.abs(deltaS), label="|ΔS|", color='blue')
        ax2.plot(T, TEC, label=f"TEC({deltaT_tec}K)", color='orange', linestyle='--')
        ax2.set_ylabel("Entropy change"); ax2.legend(); st.pyplot(fig2)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Banerjee Criterion (Arrott Plot)**")
            fig3, ax3 = plt.subplots()
            ax3.plot(M_u**2, H_user/M_u)
            ax3.set_xlabel("$M^2$"); ax3.set_ylabel("$H/M$")
            st.pyplot(fig3)
            # Logique simple de l'ordre
            pente = np.polyfit(M_u**2, H_user/M_u, 1)[0]
            st.write("Ordre suggéré :", "**2ème ordre**" if pente > 0 else "**1er ordre**")
            
        with col_b:
            st.markdown("**Master Curve**")
            st.info("Scaling universel affiché dans l'export final.")

    with tab4:
        # --- Comparaison Modèle A & B ---
        model_B = MLPRegressor(hidden_layer_sizes=(nodes_m2, nodes_m2), max_iter=3000, random_state=1).fit(X_sc, y_sc)
        H_range = np.linspace(0.1, H_user, 30)
        T_g, H_g = np.meshgrid(T, H_range)
        X_flat = scaler_X.transform(np.column_stack([T_g.ravel(), H_g.ravel()]))
        
        Z_A = scaler_y.inverse_transform(model.predict(X_flat).reshape(-1,1)).reshape(len(H_range), len(T))
        Z_B = scaler_y.inverse_transform(model_B.predict(X_flat).reshape(-1,1)).reshape(len(H_range), len(T))

        fig3d = go.Figure(data=[
            go.Surface(z=Z_A, x=T, y=H_range, colorscale='Viridis', name='Modèle A'),
            go.Surface(z=Z_B, x=T, y=H_range, colorscale='Reds', opacity=0.5, name='Modèle B', showscale=False)
        ])
        fig3d.update_layout(scene=dict(xaxis_title='T', yaxis_title='H', zaxis_title='M'), height=600)
        st.plotly_chart(fig3d, use_container_width=True)

    # --- EXPORT ---
    st.download_button("📥 Télécharger Résultats Excel", 
                       data=to_excel_full(pd.DataFrame({"T":T, "M_pred":M_u, "DeltaS":deltaS}), 
                                          pd.DataFrame({"Metric":["Smax","RCP","TEC_max","NRC","Tc"], "Val":[Smax,RCP,TEC_max,NRC,Tc]})),
                       file_name="Analyse_IA_Expert.xlsx")
else:
    st.info("👋 Bienvenue ! Veuillez charger vos données expérimentales (CSV) pour commencer l'analyse IA.")
