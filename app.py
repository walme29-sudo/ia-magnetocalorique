import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from io import BytesIO

# ================= CONFIGURATION PAGE =================
st.set_page_config(page_title="IA Magnétocalorique Expert v3", layout="wide", page_icon="🧲")

def to_excel_full(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data_Predictions', index=False)
        df_stats.to_excel(writer, sheet_name='Thermo_Params', index=False)
    return output.getvalue()

# ================= ENTÊTE =================
col1, col2 = st.columns([1,5])
with col1:
    st.markdown("## 🏛️ ISSAT")
with col2:
    st.markdown("## 🧲 IA Magnétocalorique - Analyse Expert")
    st.markdown("**Développeur : DALHOUMI WALID** | Correction Erreur de Matrice Vide")

st.divider()

# ================= BARRE LATÉRALE =================
with st.sidebar:
    st.header("⚙️ Configuration")
    nodes = st.slider("Neurones (MLP)", 64, 512, 256, step=64)
    deltaT_tec = st.slider("Fenêtre TEC (K)", 1, 10, 5)
    st.divider()
    st.info("💡 Conseil : Assurez-vous que votre CSV contient une colonne 'T' et au moins deux colonnes 'M_xT'.")

# ================= CHARGEMENT & VALIDATION =================
file = st.file_uploader("Charger le fichier CSV", type=["csv"])

if file:
    # Lecture du fichier
    df_raw = pd.read_csv(file).dropna()
    
    # Détection des colonnes
    cols = df_raw.columns.tolist()
    t_col = next((c for c in cols if c.lower() in ['t', 'temperature', 'temp']), None)
    m_cols = [c for c in cols if 'M_' in c or 'm_' in c]

    if not t_col or len(m_cols) < 2:
        st.error(f"❌ Erreur de format. Trouvé : Temp='{t_col}', Champs Magnétiques={m_cols}")
        st.stop()

    # Extraction des données numériques
    try:
        T = df_raw[t_col].values
        M_matrix = df_raw[m_cols].values
        # Extraction des valeurs H (ex: M_1T -> 1.0)
        H_known = np.array([float(''.join(c for c in col if c.isdigit() or c=='.')) for col in m_cols])
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des chiffres des colonnes : {e}")
        st.stop()

    # ================= PRÉPARATION IA =================
    # Création de la grille d'entraînement
    T_grid, H_grid = np.meshgrid(T, H_known)
    X = np.column_stack([T_grid.ravel(), H_grid.ravel()])
    y = M_matrix.T.ravel()

    # Vérification anti-crash (ValueError)
    if X.size == 0 or y.size == 0:
        st.error("❌ La matrice de données est vide. Vérifiez votre fichier CSV.")
        st.stop()

    # Entraînement avec Scaling
    with st.spinner('L\'IA apprend les propriétés du matériau...'):
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(nodes, nodes), max_iter=3000, random_state=42)
        model.fit(X_scaled, y_scaled)
        
        score_r2 = r2_score(y_scaled, model.predict(X_scaled))

    # ================= PRÉDICTIONS =================
    st.subheader("🔮 Analyse de Phase & Prédiction")
    h_target = st.number_input("Champ cible pour prédiction (Tesla)", 0.1, 15.0, 5.0)
    
    X_target = scaler_X.transform(np.column_stack([T, np.full_like(T, h_target)]))
    M_pred = scaler_y.inverse_transform(model.predict(X_target).reshape(-1,1)).ravel()

    # ================= CALCULS THERMODYNAMIQUES =================
    # Intégration de Maxwell
    dM_dT_list = [np.gradient(df_raw[c].values, T) for c in m_cols]
    dM_dT_list.append(np.gradient(M_pred, T))
    H_all = np.append(H_known, h_target)
    
    deltaS = np.trapz(dM_dT_list, x=H_all, axis=0)
    abs_ds = np.abs(deltaS)
    s_max = np.max(abs_ds)
    tc = T[np.argmax(abs_ds)]
    
    # RCP & TEC
    idx_half = np.where(abs_ds >= s_max/2)[0]
    fwhm = (T[idx_half[-1]] - T[idx_half[0]]) if len(idx_half) > 1 else 0
    rcp = s_max * fwhm

    # ================= INTERFACE GRAPHIQUE =================
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ΔS Max", f"{s_max:.4f} J/kgK")
    m2.metric("Tc (Curie)", f"{tc:.1f} K")
    m3.metric("RCP", f"{rcp:.2f}")
    m4.metric("Score IA (R²)", f"{score_r2:.3f}")

    tab1, tab2, tab3 = st.tabs(["📊 Magnétisation", "❄️ Thermodynamique", "🧬 Analyse 3D"])

    with tab1:
        fig_m = go.Figure()
        for i, col in enumerate(m_cols):
            fig_m.add_scatter(x=T, y=df_raw[col], name=f"{H_known[i]}T Exp", mode='markers')
        fig_m.add_scatter(x=T, y=M_pred, name=f"{h_target}T IA", line=dict(color='red', width=3))
        st.plotly_chart(fig_m, use_container_width=True)

    with tab2:
        
        c1, c2 = st.columns(2)
        fig_ds, ax_ds = plt.subplots()
        ax_ds.fill_between(T, 0, abs_ds, color='blue', alpha=0.2)
        ax_ds.plot(T, abs_ds, color='blue', lw=2)
        ax_ds.set_title(f"Entropie Magnétique à {h_target}T")
        ax_ds.set_xlabel("T (K)"); ax_ds.set_ylabel("|ΔS| (J/kgK)")
        c1.pyplot(fig_ds)
        
        # Arrott Plot
        fig_ar, ax_ar = plt.subplots()
        ax_ar.plot(M_pred**2, h_target/(M_user := M_pred + 1e-9), color='purple')
        ax_ar.set_title("Arrott Plot (Critère de Banerjee)")
        ax_ar.set_xlabel("M²"); ax_ar.set_ylabel("H/M")
        c2.pyplot(fig_ar)

    with tab3:
        # Surface 3D
        H_range = np.linspace(min(H_known), h_target, 20)
        T_m, H_m = np.meshgrid(T, H_range)
        X_s = scaler_X.transform(np.column_stack([T_m.ravel(), H_m.ravel()]))
        Z = scaler_y.inverse_transform(model.predict(X_s).reshape(-1,1)).reshape(20, len(T))
        
        fig3d = go.Figure(data=[go.Surface(z=Z, x=T, y=H_range, colorscale='Viridis')])
        fig3d.update_layout(title="Surface de Magnétisation Apprise", scene=dict(xaxis_title='T', yaxis_title='H', zaxis_title='M'))
        st.plotly_chart(fig3d, use_container_width=True)

    # ================= EXPORT =================
    res_df = pd.DataFrame({"T": T, "M_IA": M_pred, "DeltaS": deltaS})
    stats_df = pd.DataFrame({"Paramètre": ["Smax", "Tc", "RCP", "R2"], "Valeur": [s_max, tc, rcp, score_r2]})
    st.download_button("📥 Télécharger Rapport Excel", to_excel_full(res_df, stats_df), "Resultats_IA_Magneto.xlsx")

else:
    st.info("👋 Veuillez uploader un fichier CSV contenant les colonnes T et M_xT pour démarrer.")
