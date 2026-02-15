import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from io import BytesIO

# ====== PAGE CONFIG ======
st.set_page_config(page_title="IA Magnétocalorique Expert", layout="wide")

# ====== FONCTIONS ======
def to_excel(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, index=False, sheet_name='Data & Predictions')
        df_stats.to_excel(writer, index=False, sheet_name='Thermo Params')
    return output.getvalue()

def plot_to_pdf(fig):
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches='tight')
    return output.getvalue()

# ====== HEADER ======
col_logo, col_title = st.columns([1,5])
with col_logo:
    try:
        st.image("logo.png", width=80)
    except:
        st.markdown("### 🧲")  # emoji aimant si logo absent
with col_title:
    st.markdown("## IA Magnétocalorique - Analyse Expert")
    st.markdown("**Développeur : DALHOUMI WALID**")

st.divider()

# ====== UPLOAD CSV ======
file = st.file_uploader("Charger CSV (colonnes: Temperature/H_1T..H_5T)")

if file:
    data = pd.read_csv(file).dropna()
    st.write("Colonnes détectées:", data.columns)

    # ==== Lecture Temperature ====
    if "T" in data.columns:
        T = data["T"].values
    elif "Temperature" in data.columns:
        T = data["Temperature"].values
    else:
        T = data.iloc[:,0].values

    # ==== Lecture M (premiers 3 champs) ====
    H_cols = [col for col in data.columns if "H_" in col or "M_" in col]
    if len(H_cols) < 3:
        st.error("CSV doit contenir au moins 3 colonnes de M/H.")
    else:
        M_matrix = data[H_cols[:3]].values
        H_known = np.array([1,2,3])  # champs correspondants aux colonnes

    # ===== NN TRAINING ======
    Tg, Hg = np.meshgrid(T, H_known)
    X = np.column_stack([Tg.ravel(), Hg.ravel()])
    y = M_matrix.T.ravel()

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    nodes_nn = st.sidebar.slider("Neurones NN", 32, 256, 128, step=32)
    model = MLPRegressor(hidden_layer_sizes=(nodes_nn,nodes_nn), max_iter=5000, random_state=42)
    model.fit(X_scaled, y_scaled)

    # ====== PREDICTION ======
    H_pred = st.number_input("Champ pour prédiction (Tesla)", 0.1, 10.0, 5.0, 0.5)
    X_pred = scaler_X.transform(np.column_stack([T, np.full_like(T,H_pred)]))
    M_pred = scaler_y.inverse_transform(model.predict(X_pred).reshape(-1,1)).ravel()

    # ====== ΔS & Paramètres ======
    dM_dT = [np.gradient(M_matrix[:,i], T) for i in range(3)]
    dM_dT.append(np.gradient(M_pred, T))
    dM_dT_stack = np.vstack(dM_dT)  # shape (4, nT)
    H_all = np.append(H_known, H_pred)
    deltaS = np.trapz(dM_dT_stack, x=H_all, axis=0)

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    indices = np.where(np.abs(deltaS) >= Smax/2)[0]
    FWHM = T[indices[-1]] - T[indices[0]] if len(indices)>1 else 0
    RCP = Smax*FWHM
    RC = np.trapz(np.abs(deltaS), T)
    NRC = RCP/H_pred if H_pred!=0 else 0

    # ====== AFFICHAGE METRIQUES ======
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("ΔS Max", f"{Smax:.3f}")
    m2.metric("RCP", f"{RCP:.2f}")
    m3.metric("RC", f"{RC:.2f}")
    m4.metric("NRC", f"{NRC:.2f}")
    m5.metric("Tc (K)", f"{Tc:.1f}")

    # ====== COURBES ======
    tab1, tab2 = st.tabs(["📈 Magnétisation & Arrott", "❄️ ΔS & Master Curve"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(5,4))
        for i, M in enumerate(M_matrix.T):
            ax1.plot(T, M, label=f"{H_known[i]}T Exp")
        ax1.plot(T, M_pred, label=f"{H_pred}T IA", color='cyan', lw=2)
        ax1.set_xlabel("T (K)"); ax1.set_ylabel("M (emu/g)"); ax1.legend(fontsize='small')
        st.pyplot(fig1)

        # Arrott Plot
        fig2, ax2 = plt.subplots(figsize=(5,4))
        mask = M_pred>1e-6
        X_ar = (M_pred[mask]**2).reshape(-1,1)
        Y_ar = (H_pred/M_pred[mask]).reshape(-1,1)
        reg = LinearRegression().fit(X_ar,Y_ar)
        ax2.scatter(X_ar,Y_ar, alpha=0.3, s=10)
        ax2.plot(X_ar, reg.predict(X_ar), color='red')
        ax2.set_xlabel("M²"); ax2.set_ylabel("H/M"); st.pyplot(fig2)
        st.info(f"Arrott linear fit: y={float(reg.coef_[0][0]):.4e}x + {float(reg.intercept_[0]):.4f}")

    with tab2:
        fig3, ax3 = plt.subplots(figsize=(5,4))
        ax3.plot(T, np.abs(deltaS), label="|ΔS|", color='blue', lw=2)
        theta = (T-Tc)/FWHM if FWHM>0 else T-Tc
        ax3.plot(theta, np.abs(deltaS)/Smax, label="Master Curve", color='green', lw=2)
        ax3.set_xlabel("T-Tc ou θ"); ax3.set_ylabel("ΔS / ΔSmax")
        ax3.legend(fontsize='small'); st.pyplot(fig3)

    # ====== EXPORT ======
    df_ex = pd.DataFrame({"T":T,"M_pred":M_pred,"ΔS":deltaS})
    df_st = pd.DataFrame({"Param":["Smax","RCP","RC","NRC","Tc"],"Valeur":[Smax,RCP,RC,NRC,Tc]})
    st.download_button("📥 Télécharger Excel", data=to_excel(df_ex, df_st), file_name="Magneto_IA.xlsx")
