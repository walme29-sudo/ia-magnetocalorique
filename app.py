import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="IA Magnétocalorique", layout="wide")

# ================= EXPORT FUNCTIONS =================
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

# ================= HEADER =================
col1, col2 = st.columns([1,5])

with col1:
    try:
        st.image("logo.png", width=110)
    except:
        st.write("ISSAT")

with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Neural Network Global")
    st.markdown("**DALHOUMI WALID**")

st.divider()

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload CSV (T, M_1T, M_2T, M_3T)", type=["csv"])

if file:
    data = pd.read_csv(file).dropna()
    required = ["T","M_1T","M_2T","M_3T"]

    if not all(col in data.columns for col in required):
        st.error("CSV must contain: T, M_1T, M_2T, M_3T")
        st.stop()

    T = data["T"].values
    M_matrix = data[["M_1T","M_2T","M_3T"]].values

    # ================= GLOBAL NN =================
    H_values = np.array([1,2,3])
    T_grid_init, H_grid_init = np.meshgrid(T, H_values)

    X = np.column_stack([T_grid_init.ravel(), H_grid_init.ravel()])
    y = M_matrix.T.ravel()

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    model = MLPRegressor(hidden_layer_sizes=(128,128),
                         activation='relu',
                         solver='adam',
                         max_iter=6000,
                         random_state=42)

    model.fit(X_scaled, y_scaled)

    # ================= USER CHAMP =================
    st.subheader("🔮 Champ de prédiction personnalisé")

    H_user = st.number_input(
        "Choisir le champ magnétique (Tesla)",
        min_value=0.1,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

    # ================= PREDICTION =================
    X_user = np.column_stack([T, np.full_like(T,H_user)])
    X_user_scaled = scaler_X.transform(X_user)

    M_user = scaler_y.inverse_transform(
        model.predict(X_user_scaled).reshape(-1,1)
    ).ravel()

    # ================= ΔS =================
    dM1 = np.gradient(M_matrix[:,0],T)
    dM2 = np.gradient(M_matrix[:,1],T)
    dM3 = np.gradient(M_matrix[:,2],T)
    dM_user = np.gradient(M_user,T)

    deltaS = np.trapezoid([dM1,dM2,dM3,dM_user],
                         x=[1,2,3,H_user],
                         axis=0)

    Smax = np.max(np.abs(deltaS))
    Tc = T[np.argmax(np.abs(deltaS))]

    # ================= RCP & RC =================
    indices = np.where(np.abs(deltaS)>=Smax/2)[0]
    RCP = Smax*(T[indices[-1]]-T[indices[0]]) if len(indices)>1 else 0
    RC = np.trapezoid(np.abs(deltaS),T)

    # ================= n(T) =================
    H_list_full = [1,2,3,H_user]
    DeltaS_matrix = []

    for H in H_list_full:
        if H in [1,2,3]:
            idx = int(H)-1
            dM_dT = np.gradient(M_matrix[:,idx],T)
        else:
            dM_dT = np.gradient(M_user,T)
        DeltaS_matrix.append(np.abs(dM_dT))

    DeltaS_matrix = np.array(DeltaS_matrix)

    n_T = []
    for i in range(len(T)):
        try:
            y_vals = DeltaS_matrix[:,i]
            if np.all(y_vals>0):
                coeffs = np.polyfit(np.log(H_list_full),np.log(y_vals),1)
                n_T.append(coeffs[0])
            else:
                n_T.append(np.nan)
        except:
            n_T.append(np.nan)

    n_T = np.array(n_T)
    idx_Tc = np.argmin(np.abs(T-Tc))
    n_exponent = n_T[idx_Tc]

    # ================= METRICS =================
    st.subheader("Thermodynamic Parameters")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ΔS Max", f"{Smax:.4f}")
    c2.metric("RCP", f"{RCP:.2f}")
    c3.metric("RC", f"{RC:.2f}")
    c4.metric("n (at Tc)", f"{n_exponent:.3f}")
    c5.metric("Tc (K)", f"{Tc:.1f}")

    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Magnetisation","❄ ΔS & n(T)","🧲 Arrott & Master", "🧬 Surface 3D"])

    with tab1:
        df_m = pd.DataFrame({
            "1T":M_matrix[:,0],
            "2T":M_matrix[:,1],
            "3T":M_matrix[:,2],
            f"{H_user:.1f}T (NN)":M_user
        }, index=T)
        st.line_chart(df_m)

    with tab2:
        fig_ds, ax_ds = plt.subplots(figsize=(5,3))
        ax_ds.plot(T,deltaS)
        ax_ds.set_xlabel("T (K)")
        ax_ds.set_ylabel("ΔS")
        st.pyplot(fig_ds)

        fig_n, ax_n = plt.subplots(figsize=(5,3))
        ax_n.plot(T,n_T)
        ax_n.axvline(Tc, color='red', linestyle='--')
        ax_n.set_xlabel("T (K)")
        ax_n.set_ylabel("n(T)")
        st.pyplot(fig_n)

    with tab3:
        st.subheader("Arrott Plot (H/M vs M²)")
        fig_arrott, ax_arrott = plt.subplots(figsize=(5,3))
        H_plot = [1,2,3,H_user]
        M_all = [M_matrix[:,0],M_matrix[:,1],M_matrix[:,2],M_user]
        for H, M in zip(H_plot, M_all):
            valid = M != 0
            ax_arrott.plot(M[valid]**2, H/M[valid], label=f"{H:.1f}T")
        ax_arrott.legend()
        st.pyplot(fig_arrott)

        st.subheader("Master Curve Multi-H")
        fig_master, ax_master = plt.subplots(figsize=(5,3))
        for H, M in zip(H_plot, M_all):
            dM_dT = np.gradient(M,T)
            DeltaS_temp = np.abs(dM_dT)
            DeltaS_norm = DeltaS_temp/np.max(DeltaS_temp)
            indices_half = np.where(DeltaS_norm>=0.5)[0]
            if len(indices_half) > 1:
                T_r1, T_r2 = T[indices_half[0]], T[indices_half[-1]]
                theta = np.where(T < Tc, -(T-Tc)/(T_r1-Tc), (T-Tc)/(T_r2-Tc))
                ax_master.plot(theta, DeltaS_norm, label=f"{H:.1f}T")
        ax_master.legend()
        st.pyplot(fig_master)

    with tab4:
        st.subheader("🧬 Surface 3D M(T,H) Interactive")
        
        # Génération de la surface
        H_surface = np.linspace(0.1, H_user, 30)
        T_surface = T
        T_grid, H_grid = np.meshgrid(T_surface, H_surface)
        
        X_surf = np.column_stack([T_grid.ravel(), H_grid.ravel()])
        X_surf_scaled = scaler_X.transform(X_surf)
        M_surf = scaler_y.inverse_transform(model.predict(X_surf_scaled).reshape(-1,1)).reshape(len(H_surface), len(T_surface))

        # Graphique Plotly
        fig_3d = go.Figure(data=[go.Surface(z=M_surf, x=T_surface, y=H_surface, colorscale='Viridis')])
        fig_3d.update_layout(
            scene=dict(xaxis_title='T (K)', yaxis_title='H (T)', zaxis_title='M'),
            width=800, height=600, margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # ================= EXPORT EXCEL =================
    st.subheader("Download Results")
    df_export = pd.DataFrame({
        "T":T, "M_1T":M_matrix[:,0], "M_2T":M_matrix[:,1], "M_3T":M_matrix[:,2],
        f"M_{H_user:.1f}T_NN":M_user, "DeltaS":deltaS, "n(T)":n_T
    })
    df_stats = pd.DataFrame({
        "Parameter":["DeltaS Max","RCP","RC","n(Tc)","Tc"],
        "Value":[Smax,RCP,RC,n_exponent,Tc]
    })
    st.download_button("📥 Full Excel File", data=to_excel_full(df_export,df_stats), file_name="Magnetocaloric_Final.xlsx")

