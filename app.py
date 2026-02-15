import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO

# Configuration Expert
st.set_page_config(page_title="IA Magnétocalorique Pro", layout="wide")
plt.rcParams.update({'font.size': 10, 'lines.linewidth': 1.5})

# ====== FONCTIONS DE CALCUL ======
def calculate_scaling_laws(H_list, Smax_list):
    """Calcule l'exposant n via une régression log-log : ln(Smax) = n*ln(H) + C"""
    log_H = np.log(np.array(H_list)).reshape(-1, 1)
    log_S = np.log(np.array(Smax_list))
    model_lin = LinearRegression().fit(log_H, log_S)
    return model_lin.coef_[0], model_lin.score(log_H, log_S)

def get_master_curve(T, deltaS, Tc, fwhm):
    """Calcule la température réduite theta pour la courbe universelle."""
    if fwhm == 0: return T - Tc
    theta = np.where(T <= Tc, -(T - Tc) / (T[0] - Tc), (T - Tc) / (T[-1] - Tc))
    return theta

# ====== HEADER ======
st.title("🧲 IA Magnétocalorique : Analyse de Phase Expert")
st.markdown("---")

# ====== SIDEBAR ======
with st.sidebar:
    st.header("📂 Données & Modèle")
    file = st.file_uploader("Fichier CSV", type=["csv"])
    st.divider()
    nodes = st.select_slider("Puissance IA (Neurones)", options=[64, 128, 256, 512], value=256)
    target_H = st.number_input("Prédire pour H (Tesla)", 0.0, 10.0, 5.0, 0.5)

if file:
    df = pd.read_csv(file).dropna()
    cols = df.columns.tolist()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        t_col = st.selectbox("Température (K)", cols)
        m_cols = st.multiselect("Champs M(H) dispo", [c for c in cols if c != t_col])

    if len(m_cols) >= 3:
        T = df[t_col].values
        # Extraction des valeurs de H
        H_vals = []
        for c in m_cols:
            nums = ''.join(filter(lambda x: x.isdigit() or x=='.', c))
            H_vals.append(float(nums) if nums else 1.0)

        # ====== IA ENGINE ======
        with st.spinner("Entraînement de l'IA sur les cycles magnétiques..."):
            X_data = []
            y_data = []
            for i, h in enumerate(H_vals):
                for t, m in zip(T, df[m_cols[i]]):
                    X_data.append([t, h])
                    y_data.append(m)
            
            X_train = np.array(X_data)
            y_train = np.array(y_data)
            
            sc_X, sc_y = StandardScaler(), StandardScaler()
            X_s = sc_X.fit_transform(X_train)
            y_s = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            model = MLPRegressor(hidden_layer_sizes=(nodes, nodes), max_iter=3000, random_state=42)
            model.fit(X_s, y_s)
            
            # Prédiction sur champ cible
            X_target = sc_X.transform(np.column_stack([T, np.full_like(T, target_H)]))
            M_pred = sc_y.inverse_transform(model.predict(X_target).reshape(-1, 1)).ravel()

        # ====== CALCULS THERMO ======
        # Calcul de dM/dT pour tous les champs (exp + pred)
        dM_dT_all = [np.gradient(df[c].values, T) for c in m_cols]
        dM_dT_all.append(np.gradient(M_pred, T))
        all_H_sorted = sorted(H_vals + [target_H])
        
        # Delta Sm par intégration de Maxwell
        deltaS_target = np.trapz(dM_dT_all, x=all_H_sorted, axis=0)
        abs_ds = np.abs(deltaS_target)
        s_max = np.max(abs_ds)
        tc_idx = np.argmax(abs_ds)
        tc = T[tc_idx]
        
        # FWHM & RCP
        idx_half = np.where(abs_ds >= s_max/2)[0]
        fwhm = T[idx_half[-1]] - T[idx_half[0]] if len(idx_half) > 1 else 0
        rcp = s_max * fwhm

        # ====== INTERFACE RESULTATS ======
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ΔS Max", f"{s_max:.3f} J/kg·K")
        m2.metric("Tc (Curie)", f"{tc:.1f} K")
        m3.metric("RCP", f"{rcp:.2f}")
        m4.metric("R² IA", f"{r2_score(y_s, model.predict(X_s)):.3f}")

        st.divider()
        t1, t2, t3 = st.tabs(["📈 Physique M(T, H)", "❄️ Thermodynamique ΔS", "🧬 Master Curve & Exposants"])

        with t1:
            col_a, col_b = st.columns(2)
            fig_m, ax_m = plt.subplots()
            for i, col in enumerate(m_cols):
                ax_m.plot(T, df[col], 'o', ms=3, alpha=0.4, label=f"{H_vals[i]}T Exp")
            ax_m.plot(T, M_pred, 'r-', lw=2, label=f"{target_H}T IA")
            ax_m.set_ylabel("M (emu/g)"); ax_m.set_xlabel("T (K)"); ax_m.legend()
            col_a.pyplot(fig_m)
            
            # Arrott Plot simplifié
            fig_ar, ax_ar = plt.subplots()
            ax_ar.plot(M_pred**2, target_H/M_pred, 'b-')
            ax_ar.set_title("Arrott Plot (IA)"); ax_ar.set_xlabel("M²"); ax_ar.set_ylabel("H/M")
            col_b.pyplot(fig_ar)

        with t2:
            fig_s, ax_s = plt.subplots(figsize=(8, 4))
            ax_s.fill_between(T, 0, abs_ds, color='blue', alpha=0.1)
            ax_s.plot(T, abs_ds, color='blue', lw=2, label=f"ΔH = {target_H}T")
            ax_s.axvline(tc, color='red', ls='--', label=f"Tc = {tc}K")
            ax_s.set_ylabel("|ΔS_m| (J/kg·K)"); ax_s.set_xlabel("T (K)"); ax_s.legend()
            st.pyplot(fig_s)
            

        with t3:
            ca, cb = st.columns(2)
            # Master Curve
            theta = get_master_curve(T, abs_ds, tc, fwhm)
            fig_ma, ax_ma = plt.subplots()
            ax_ma.plot(theta, abs_ds/s_max, 'g-', lw=2)
            ax_ma.set_title("Master Curve (Normalisée)"); ax_ma.set_xlabel("θ (Temp. Réduite)"); ax_ma.set_ylabel("ΔS / ΔSmax")
            ca.pyplot(fig_ma)
            
            # Lois d'échelle (Scaling Law)
            # Pour l'exemple, on simule le scaling sur les champs dispos
            s_max_list = [np.max(np.abs(np.trapz(dM_dT_all[:i+1], x=sorted(H_vals)[:i+1], axis=0))) for i in range(len(H_vals))]
            n_exp, r2_n = calculate_scaling_laws(H_vals, s_max_list)
            
            cb.write("### Analyse des Exposants")
            cb.info(f"Exposant critique **n** : {n_exp:.3f}")
            cb.write(f"Confiance (R²) : {r2_n:.4f}")
            cb.write("> Un n ≈ 0.66 indique un modèle de champ moyen (3D Heisenberg ≈ 0.64).")
            

        # Export Excel complet
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            res_df = pd.DataFrame({"T(K)": T, "M_IA": M_pred, "dS_IA": deltaS_target, "Theta": theta})
            res_df.to_excel(writer, sheet_name='Predictions')
        st.download_button("📥 Export Données Expert (Excel)", output.getvalue(), "Analyse_Magneto_Expert.xlsx")
    else:
        st.error("L'analyse nécessite au moins 3 colonnes de magnétisation pour établir les lois d'échelle.")
