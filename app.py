import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from io import BytesIO

# ================= PAGE CONFIG =================
st.set_page_config(page_title="IA Magnetocalorique PRO", layout="wide")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration IA")

    nodes_A = st.slider("Neurones Modèle A", 32, 256, 128, step=32)
    nodes_B = st.slider("Neurones Modèle B", 32, 256, 64, step=32)
    deltaT_tec = st.slider("Fenêtre TEC (K)", 1, 10, 3)

    alloy = st.selectbox(
        "Type Alliage",
        ["Gd-based", "LaFeSi", "MnFePAs", "Heusler"]
    )

theoretical_db = {
    "Gd-based": {"Smax": 10, "RCP": 400},
    "LaFeSi": {"Smax": 20, "RCP": 450},
    "MnFePAs": {"Smax": 18, "RCP": 350},
    "Heusler": {"Smax": 12, "RCP": 300}
}

# ================= TITLE =================
st.title("🧲 IA Magnétocalorique – Version PRO Complète")

file = st.file_uploader("Upload CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:

    data = pd.read_csv(file).dropna()
    T = data["T"].values
    M_exp = data[["M_1T", "M_2T", "M_3T"]].values

    # ===== TRAIN IA =====
    H_known = np.array([1, 2, 3])
    Tg, Hg = np.meshgrid(T, H_known)

    X = np.column_stack([Tg.ravel(), Hg.ravel()])
    y = M_exp.T.ravel()

    scalerX = StandardScaler()
    scalerY = StandardScaler()

    Xs = scalerX.fit_transform(X)
    ys = scalerY.fit_transform(y.reshape(-1, 1)).ravel()

    modelA = MLPRegressor(
        hidden_layer_sizes=(nodes_A, nodes_A),
        max_iter=5000,
        random_state=42
    )
    modelA.fit(Xs, ys)

    modelB = MLPRegressor(
        hidden_layer_sizes=(nodes_B, nodes_B),
        max_iter=3000,
        random_state=1
    )
    modelB.fit(Xs, ys)

    # ===== USER FIELD =====
    H_user = st.number_input("Champ cible (T)", 0.1, 10.0, 5.0)

    Xu = scalerX.transform(
        np.column_stack([T, np.full_like(T, H_user)])
    )

    M_pred = scalerY.inverse_transform(
        modelA.predict(Xu).reshape(-1, 1)
    ).ravel()

    # ===== THERMODYNAMICS =====
    dMdT = [
        np.gradient(m, T)
        for m in [M_exp[:, 0], M_exp[:, 1], M_exp[:, 2], M_pred]
    ]

    deltaS = np.trapezoid(
        dMdT,
        x=[1, 2, 3, H_user],
        axis=0
    )

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    idx = np.where(np.abs(deltaS) >= Smax / 2)[0]

    if len(idx) > 1:
        FWHM = T[idx[-1]] - T[idx[0]]
        RCP = Smax * FWHM
    else:
        FWHM = 0
        RCP = 0

    RC = np.trapezoid(np.abs(deltaS), T)
    q = RC

    Cp = 400
    deltaTad = Smax * H_user / Cp

    n_exp = np.gradient(
        np.log(np.abs(deltaS) + 1e-6),
        np.log(H_user + 1e-6)
    )

    # ===== METRICS =====
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ΔSmax", f"{Smax:.2f}")
    m2.metric("RCP", f"{RCP:.1f}")
    m3.metric("RC (q)", f"{q:.1f}")
    m4.metric("ΔTad", f"{deltaTad:.2f} K")
    m5.metric("Tc", f"{Tc:.1f} K")

    # ===== COMPARISON TABLE =====
    st.subheader("📊 Comparaison IA vs Théorie")

    theo = theoretical_db[alloy]

    df_comp = pd.DataFrame({
        "Paramètre": ["ΔSmax", "RCP"],
        "IA": [Smax, RCP],
        "Théorie": [theo["Smax"], theo["RCP"]],
        "Écart %": [
            100 * (Smax - theo["Smax"]) / theo["Smax"],
            100 * (RCP - theo["RCP"]) / theo["RCP"]
        ]
    })

    st.dataframe(df_comp, use_container_width=True)

    # ===== TABS =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Magnétisation",
        "❄ Thermodynamique",
        "🧲 Arrott & Scaling",
        "🌐 Surfaces 3D",
        "🖼 Heatmap HD"
    ])

    # ================= TAB 1 =================
    with tab1:
        fig1, ax1 = plt.subplots()

        colors = ["black", "red", "green", "blue"]
        M_list = [M_exp[:, 0], M_exp[:, 1], M_exp[:, 2], M_pred]
        H_list = [1, 2, 3, H_user]

        for m, h, c in zip(M_list, H_list, colors):
            ax1.plot(T, m, label=f"{h}T", color=c)

        ax1.legend()
        ax1.set_xlabel("T")
        ax1.set_ylabel("M")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()

        for m, h, c in zip(M_list, H_list, colors):
            mask = m > 0.1
            ax2.plot(T[mask], h / m[mask], color=c)

        ax2.set_xlabel("T")
        ax2.set_ylabel("H/M")
        st.pyplot(fig2)

    # ================= TAB 2 =================
    with tab2:
        figS, axS = plt.subplots()
        axS.plot(T, np.abs(deltaS))
        axS.fill_between(T, np.abs(deltaS), alpha=0.3)
        st.pyplot(figS)

        TEC = [
            np.mean(np.abs(deltaS)[
                (T >= t - deltaT_tec / 2) &
                (T <= t + deltaT_tec / 2)
            ])
            for t in T
        ]

        figTEC, axTEC = plt.subplots()
        axTEC.plot(T, TEC)
        st.pyplot(figTEC)

        figN, axN = plt.subplots()
        axN.plot(T, n_exp)
        st.pyplot(figN)

    # ================= TAB 3 =================
    with tab3:
        mask = M_pred > 1e-6
        Xf = (M_pred[mask] ** 2).reshape(-1, 1)
        Yf = (H_user / M_pred[mask]).reshape(-1, 1)

        if len(Xf) > 1:
            reg = LinearRegression().fit(Xf, Yf)
            figA, axA = plt.subplots()
            axA.scatter(Xf, Yf, s=10)
            axA.plot(Xf, reg.predict(Xf), color="red")
            st.pyplot(figA)

    # ================= TAB 4 =================
    with tab4:
        H_range = np.linspace(0.1, H_user, 40)
        Tg, Hg = np.meshgrid(T, H_range)

        Xsurf = scalerX.transform(
            np.column_stack([Tg.ravel(), Hg.ravel()])
        )

        Z = scalerY.inverse_transform(
            modelA.predict(Xsurf).reshape(-1, 1)
        ).reshape(len(H_range), len(T))

        fig3d = go.Figure(
            data=[go.Surface(z=Z, x=T, y=H_range)]
        )

        st.plotly_chart(fig3d, use_container_width=True)

    # ================= TAB 5 =================
    with tab5:
        Hfine = np.linspace(0.1, H_user, 100)
        Tfine = np.linspace(T.min(), T.max(), 100)

        Tg, Hg = np.meshgrid(Tfine, Hfine)

        Xhd = scalerX.transform(
            np.column_stack([Tg.ravel(), Hg.ravel()])
        )

        Mhd = scalerY.inverse_transform(
            modelA.predict(Xhd).reshape(-1, 1)
        ).reshape(len(Hfine), len(Tfine))

        figHD, axHD = plt.subplots(figsize=(7, 5), dpi=300)
        cont = axHD.contourf(Tfine, Hfine, Mhd, levels=60)
        figHD.colorbar(cont)
        st.pyplot(figHD)

        buf = BytesIO()
        figHD.savefig(buf, format="png", dpi=300)
        buf.seek(0)

        st.download_button(
            "📥 Télécharger PNG HD",
            data=buf,
            file_name="Structure_HD.png"
        )

    # ================= EXPORT =================
    df_export = pd.DataFrame({
        "T": T,
        "M_pred": M_pred,
        "DeltaS": deltaS,
        "n(T)": n_exp
    })

    st.download_button(
        "📥 Télécharger CSV",
        data=df_export.to_csv(index=False),
        file_name="Magneto_PRO.csv"
    )
