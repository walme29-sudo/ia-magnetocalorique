import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="IA Magnétocalorique Expert PRO", layout="wide")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration IA & Matériau")

    nodes_m1 = st.slider("Neurones Modèle A", 32, 256, 128, step=32)
    nodes_m2 = st.slider("Neurones Modèle B", 32, 256, 64, step=32)
    deltaT_tec = st.slider("Fenêtre TEC (K)", 1, 10, 3)

    alloy_type = st.selectbox(
        "Type d’alliage",
        ["Gd-based", "LaFeSi", "MnFePAs", "Heusler"]
    )

# ================= VALEURS THEORIQUES =================
theoretical_db = {
    "Gd-based": {"Smax": 10, "RCP": 400},
    "LaFeSi": {"Smax": 20, "RCP": 450},
    "MnFePAs": {"Smax": 18, "RCP": 350},
    "Heusler": {"Smax": 12, "RCP": 300}
}

# ================= HEADER =================
st.title("🧲 IA Magnétocalorique – Version Expert Scientifique")

# ================= LOAD DATA =================
file = st.file_uploader("Charger CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:

    data = pd.read_csv(file).dropna()
    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # ===== TRAIN IA =====
    H_known = np.array([1,2,3])
    Tg, Hg = np.meshgrid(T, H_known)

    X = np.column_stack([Tg.ravel(), Hg.ravel()])
    y = M_matrix.T.ravel()

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    model = MLPRegressor(hidden_layer_sizes=(nodes_m1,nodes_m1),
                         max_iter=5000, random_state=42)
    model.fit(Xs, ys)

    # ===== USER FIELD =====
    H_user = st.number_input("Champ cible (Tesla)", 0.1, 10.0, 5.0)

    Xu = scaler_X.transform(np.column_stack([T, np.full_like(T,H_user)]))
    M_u = scaler_y.inverse_transform(
        model.predict(Xu).reshape(-1,1)
    ).ravel()

    # ===== THERMO CALCULATIONS =====
    dM_dT = [np.gradient(m,T) for m in [M_matrix[:,0],M_matrix[:,1],M_matrix[:,2],M_u]]
    deltaS = np.trapezoid(dM_dT, x=[1,2,3,H_user], axis=0)

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    indices = np.where(np.abs(deltaS)>=Smax/2)[0]

    if len(indices)>1:
        FWHM = T[indices[-1]]-T[indices[0]]
        RCP = Smax*FWHM
    else:
        FWHM=0
        RCP=0

    RC = np.trapezoid(np.abs(deltaS),T)
    q = RC
    NRC = RCP/H_user if H_user!=0 else 0

    # ===== ESTIMATION ΔTad =====
    Cp = 400  # valeur moyenne J/kg.K
    deltaTad = Smax*H_user/Cp

    # ===== EXPOSANT CRITIQUE n(T) =====
    n_exp = np.gradient(np.log(np.abs(deltaS)+1e-6),
                        np.log(H_user+1e-6))

    # ================= METRICS =================
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("ΔSmax", f"{Smax:.2f}")
    col2.metric("RCP", f"{RCP:.1f}")
    col3.metric("RC (q)", f"{q:.1f}")
    col4.metric("ΔTad estimé", f"{deltaTad:.2f} K")
    col5.metric("Tc", f"{Tc:.1f} K")

    # ================= TABLEAU COMPARATIF =================
    st.subheader("📊 Comparaison IA vs Valeurs Théoriques")

    theo = theoretical_db[alloy_type]

    df_compare = pd.DataFrame({
        "Paramètre":["ΔSmax","RCP"],
        "IA":[Smax,RCP],
        "Théorique":[theo["Smax"],theo["RCP"]],
        "Écart %":[
            100*(Smax-theo["Smax"])/theo["Smax"],
            100*(RCP-theo["RCP"])/theo["RCP"]
        ]
    })

    st.dataframe(df_compare,use_container_width=True)

    # ================= TABS =================
    tab1,tab2,tab3 = st.tabs([
        "❄ Entropie & Refroidissement",
        "🧲 Arrott",
        "🌐 Surface 3D"
    ])

    # ===== TAB 1 =====
    with tab1:
        fig,ax = plt.subplots(figsize=(7,4))

        ax.plot(T,np.abs(deltaS),label="|ΔS|",lw=2)
        ax.axhline(Smax,color='red',ls='--',label="ΔSmax")
        ax.axvspan(T[indices[0]] if len(indices)>1 else Tc,
                   T[indices[-1]] if len(indices)>1 else Tc,
                   alpha=0.2,label="FWHM")

        ax.set_xlabel("T (K)")
        ax.set_ylabel("ΔS")
        ax.legend()
        st.pyplot(fig)

        # Courbe RCP & q
        fig2,ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(T,np.abs(deltaS),label="ΔS")
        ax2.fill_between(T,np.abs(deltaS),alpha=0.3)
        ax2.set_title("Visualisation graphique de RC (q)")
        st.pyplot(fig2)

    # ===== TAB 2 =====
    with tab2:
        mask = M_u>1e-6
        Xf = (M_u[mask]**2).reshape(-1,1)
        Yf = (H_user/M_u[mask]).reshape(-1,1)

        if len(Xf)>1:
            reg = LinearRegression().fit(Xf,Yf)

            fig_ar,ax_ar = plt.subplots(figsize=(6,4))
            ax_ar.scatter(Xf,Yf,s=10)
            ax_ar.plot(Xf,reg.predict(Xf),color='red')
            ax_ar.set_xlabel("M²")
            ax_ar.set_ylabel("H/M")
            st.pyplot(fig_ar)

            pente = reg.coef_[0][0]
            st.info(f"Transition {'2ème ordre' if pente>0 else '1er ordre'}")

    # ===== TAB 3 =====
    with tab3:
        H_range = np.linspace(0.1,H_user,40)
        Tg,Hg = np.meshgrid(T,H_range)

        Xsurf = scaler_X.transform(
            np.column_stack([Tg.ravel(),Hg.ravel()])
        )

        Z = scaler_y.inverse_transform(
            model.predict(Xsurf).reshape(-1,1)
        ).reshape(len(H_range),len(T))

        fig3d = go.Figure(data=[go.Surface(z=Z,x=T,y=H_range)])
        fig3d.update_layout(scene=dict(
            xaxis_title='T (K)',
            yaxis_title='H (T)',
            zaxis_title='M'
        ),height=600)

        st.plotly_chart(fig3d,use_container_width=True)

    # ================= EXPORT =================
    df_export = pd.DataFrame({
        "T":T,
        "M_pred":M_u,
        "DeltaS":deltaS,
        "n(T)":n_exp
    })

    st.download_button(
        "📥 Télécharger Résultats Excel",
        data=df_export.to_csv(index=False),
        file_name="Magneto_Expert_PRO.csv"
    )
