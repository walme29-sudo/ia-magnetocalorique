import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="IA Magnétocalorique - ISSAT", layout="wide")

# ================= EXPORT FUNCTIONS =================
def to_excel_full(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data_Predictions')
        df_stats.to_excel(writer, sheet_name='Thermo_Parameters', index=False)
    return output.getvalue()

def plot_to_pdf(fig):
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches='tight')
    return output.getvalue()

# ================= HEADER =================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    try:
        st.image("logo.png", width=120)
    except:
        st.info("ISSAT Kasserine")

with col_title:
    st.title("🧲 IA Magnétocalorique - Neural Network Global")
    st.markdown("**Développeur : DALHOUMI WALID**")

st.divider()

# ================= FILE UPLOAD =================
file = st.file_uploader("Charger fichier CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:

    data = pd.read_csv(file).dropna()
    required = ["T", "M_1T", "M_2T", "M_3T"]

    if not all(col in data.columns for col in required):
        st.error("Colonnes requises : T, M_1T, M_2T, M_3T")
        st.stop()

    T = data["T"].values
    M_matrix = data[["M_1T", "M_2T", "M_3T"]].values
    H_values = np.array([1,2,3])

    # ================= GLOBAL NN =================
    T_grid, H_grid = np.meshgrid(T, H_values)

    X = np.column_stack([T_grid.ravel(), H_grid.ravel()])
    y = M_matrix.T.ravel()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(128,128),
        activation='relu',
        solver='adam',
        max_iter=6000,
        random_state=42
    )

    model.fit(X_scaled, y_scaled)

    # ================= PREDICTION 5T =================
    H_pred = np.full_like(T, 5)
    X_pred = np.column_stack([T, H_pred])
    X_pred_scaled = scaler_X.transform(X_pred)

    M_5T = scaler_y.inverse_transform(
        model.predict(X_pred_scaled).reshape(-1,1)
    ).ravel()

    # ================= CALCUL ΔS =================
    dM_dT_1 = np.gradient(M_matrix[:,0], T)
    dM_dT_2 = np.gradient(M_matrix[:,1], T)
    dM_dT_3 = np.gradient(M_matrix[:,2], T)
    dM_dT_5 = np.gradient(M_5T, T)

    deltaS = np.trapezoid(
        [dM_dT_1, dM_dT_2, dM_dT_3, dM_dT_5],
        x=[1,2,3,5],
        axis=0
    )

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # ================= RCP =================
    indices = np.where(np.abs(deltaS) >= Smax/2)[0]
    if len(indices) > 1:
        RCP = Smax * (T[indices[-1]] - T[indices[0]])
    else:
        RCP = 0

    # ================= RC =================
    RC = np.trapezoid(np.abs(deltaS), T)

    # ================= EXPOSANT n =================
    H_list = np.array([1,2,3,5])
    DeltaS_H = []

    for H in H_list:
        X_temp = np.column_stack([T, np.full_like(T, H)])
        X_temp_scaled = scaler_X.transform(X_temp)
        M_temp = scaler_y.inverse_transform(
            model.predict(X_temp_scaled).reshape(-1,1)
        ).ravel()
        dM_dT = np.gradient(M_temp, T)
        DeltaS_H.append(np.max(np.abs(dM_dT)))

    DeltaS_H = np.array(DeltaS_H)
    coeffs = np.polyfit(np.log(H_list), np.log(DeltaS_H), 1)
    n_exponent = coeffs[0]

    # ================= METRICS =================
    st.subheader("Paramètres Thermodynamiques")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ΔS Max", f"{Smax:.4f}")
    c2.metric("RCP", f"{RCP:.2f}")
    c3.metric("RC", f"{RC:.2f}")
    c4.metric("n exponent", f"{n_exponent:.3f}")
    c5.metric("Tc (K)", f"{Tc:.1f}")

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs([
        "📈 Magnetisation",
        "❄ ΔS Curve",
        "🧲 Arrott & Master"
    ])

    with tab1:
        df_m = pd.DataFrame({
            "1T": M_matrix[:,0],
            "2T": M_matrix[:,1],
            "3T": M_matrix[:,2],
            "5T (NN)": M_5T
        }, index=T)
        st.line_chart(df_m, height=350)

    with tab2:
        df_ds = pd.DataFrame({"ΔS (1→5T)": deltaS}, index=T)
        st.line_chart(df_ds, height=350)

    with tab3:

        st.subheader("Arrott Plot (H/M vs M²)")

        fig_arrott, ax_arrott = plt.subplots(figsize=(6,4))
        for i,H in enumerate([1,2,3]):
            M = M_matrix[:,i]
            ax_arrott.plot(M**2, H/M, label=f"{H} T")

        ax_arrott.set_xlabel("M²")
        ax_arrott.set_ylabel("H/M")
        ax_arrott.legend()
        st.pyplot(fig_arrott)

        st.subheader("Master Curve Universelle")

        deltaS_norm = deltaS / np.max(np.abs(deltaS))
        theta = (T - Tc) / (np.max(T) - np.min(T))

        fig_master, ax_master = plt.subplots(figsize=(6,4))
        ax_master.plot(theta, deltaS_norm)
        ax_master.set_xlabel("θ (Température réduite)")
        ax_master.set_ylabel("ΔS / ΔSmax")
        st.pyplot(fig_master)

        colA,colB = st.columns(2)
        colA.download_button(
            "📥 Arrott PDF",
            data=plot_to_pdf(fig_arrott),
            file_name="Arrott_Plot.pdf",
            mime="application/pdf"
        )

        colB.download_button(
            "📥 Master Curve PDF",
            data=plot_to_pdf(fig_master),
            file_name="Master_Curve.pdf",
            mime="application/pdf"
        )

    # ================= EXPORT EXCEL =================
    st.subheader("Téléchargement Excel")

    df_export = pd.DataFrame({
        "T": T,
        "M_1T": M_matrix[:,0],
        "M_2T": M_matrix[:,1],
        "M_3T": M_matrix[:,2],
        "M_5T_NN": M_5T,
        "DeltaS": deltaS
    })

    df_stats = pd.DataFrame({
        "Parameter": ["DeltaS Max","RCP","RC","n exponent","Tc"],
        "Value": [Smax,RCP,RC,n_exponent,Tc]
    })

    st.download_button(
        "📥 Télécharger Excel Complet",
        data=to_excel_full(df_export, df_stats),
        file_name="Magnetocaloric_Final_Walid.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Veuillez charger un fichier CSV.")
