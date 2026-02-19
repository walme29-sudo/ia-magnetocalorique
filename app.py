import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="IA Magnétocalorique - Comparaison Multi-Matériaux", layout="wide")

def to_excel(df_list, sheet_names):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, name in zip(df_list, sheet_names):
            df.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()

# ================= HEADER =================
st.title("🧲 IA Magnétocalorique - Comparaison Scientifique")
st.markdown("### Comparaison MnFeP, Heusler, etc.")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration")
    files = st.file_uploader("Charger plusieurs CSV", type=["csv"], accept_multiple_files=True)
    nodes = st.slider("Neurones IA", 64, 512, 256)
    h_target = st.number_input("Champ cible (T)", 0.1, 10.0, 5.0)
    cp_const = st.number_input("Cp (J/kg.K)", 100, 1000, 450)

# ================= ANALYSE =================
if files:

    materials_results = {}
    summary_table = []

    for file in files:

        name = file.name.split(".")[0]
        df = pd.read_csv(file).dropna()

        cols = df.columns.tolist()
        t_col = next((c for c in cols if c.lower() in ['t', 'temp']), cols[0])
        m_cols = [c for c in cols if 'M_' in c]

        if len(m_cols) < 2:
            continue

        T = df[t_col].values
        M_matrix = df[m_cols].values
        H_vals = np.array([float(''.join(c for c in col if c.isdigit() or c=='.')) for col in m_cols])

        # ===== IA TRAINING =====
        T_g, H_g = np.meshgrid(T, H_vals)
        X = np.column_stack([T_g.ravel(), H_g.ravel()])
        y = M_matrix.T.ravel()

        sc_X, sc_y = StandardScaler(), StandardScaler()
        X_s = sc_X.fit_transform(X)
        y_s = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(nodes, nodes),
                             max_iter=3000,
                             random_state=42)
        model.fit(X_s, y_s)

        # ===== PREDICTION =====
        X_p = sc_X.transform(np.column_stack([T, np.full_like(T, h_target)]))
        M_pred = sc_y.inverse_transform(model.predict(X_p).reshape(-1, 1)).ravel()

        dM_dT = np.gradient(M_pred, T)

        # Maxwell Integration
        h_steps = np.linspace(0, h_target, 20)
        gradients = []

        for hi in h_steps:
            Xi = sc_X.transform(np.column_stack([T, np.full_like(T, hi)]))
            Mi = sc_y.inverse_transform(model.predict(Xi).reshape(-1, 1)).ravel()
            gradients.append(np.gradient(Mi, T))

        ds = np.abs(np.trapezoid(gradients, x=h_steps, axis=0))

        s_max = np.max(ds)
        tc = T[np.argmax(ds)]
        dt_ad = (T * ds) / cp_const

        # ===== RCP =====
        idx_half = np.where(ds >= s_max/2)[0]
        if len(idx_half) > 1:
            fwhm = T[idx_half[-1]] - T[idx_half[0]]
            rcp = s_max * fwhm
        else:
            fwhm, rcp = 0, 0

        # ===== MASTER CURVE =====
        if len(idx_half) > 1:
            tr1, tr2 = T[idx_half[0]], T[idx_half[-1]]
            theta = np.where(T <= tc,
                             -(T - tc)/(tr1 - tc + 1e-5),
                             (T - tc)/(tr2 - tc + 1e-5))
        else:
            theta = T - tc

        materials_results[name] = {
            'T': T,
            'ds': ds,
            'dt_ad': dt_ad,
            's_max': s_max,
            'rcp': rcp,
            'tc': tc,
            'theta': theta
        }

        summary_table.append([name, round(s_max,2), round(fwhm,2), round(rcp,2), round(tc,2)])

    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 ΔS Comparaison",
        "❄️ ΔTad Comparaison",
        "📊 RCP vs Matériau",
        "📁 Export"
    ])

    # -------- ΔS --------
    with tab1:
        fig1, ax1 = plt.subplots()
        for name, data in materials_results.items():
            ax1.plot(data['T'], data['ds'], label=name)
        ax1.set_xlabel("T (K)")
        ax1.set_ylabel("ΔS (J/kg.K)")
        ax1.legend()
        st.pyplot(fig1)

    # -------- ΔTad --------
    with tab2:
        fig2, ax2 = plt.subplots()
        for name, data in materials_results.items():
            ax2.plot(data['T'], data['dt_ad'], label=name)
        ax2.set_xlabel("T (K)")
        ax2.set_ylabel("ΔTad (K)")
        ax2.legend()
        st.pyplot(fig2)

    # -------- RCP --------
    with tab3:
        fig3, ax3 = plt.subplots()
        names = list(materials_results.keys())
        rcps = [materials_results[n]['rcp'] for n in names]
        ax3.bar(names, rcps)
        ax3.set_ylabel("RCP (J/kg)")
        st.pyplot(fig3)

        summary_df = pd.DataFrame(summary_table,
                                  columns=["Matériau", "ΔSmax", "FWHM", "RCP", "Tc"])
        st.dataframe(summary_df)

    # -------- EXPORT --------
    with tab4:
        summary_df = pd.DataFrame(summary_table,
                                  columns=["Matériau", "ΔSmax", "FWHM", "RCP", "Tc"])
        st.download_button("📥 Télécharger Résumé Excel",
                           to_excel([summary_df], ["Résumé"]),
                           "Comparaison_Magnetocalorique.xlsx")

else:
    st.info("Charge au moins deux matériaux pour comparer.")
