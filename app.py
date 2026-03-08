import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from io import BytesIO
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import io
import zipfile
import matplotlib.ticker as ticker
import plotly.express as px
st.set_page_config(layout="wide")
# ==========================================
# 1. CONFIGURATION ET SESSION STATE
# ==========================================
st.set_page_config(page_title="Plateforme IA URMAN", layout="wide")

if 'page' not in st.session_state:
    st.session_state['page'] = 'accueil'
if 'history' not in st.session_state:
    st.session_state.history = []
# --- FONCTIONS DE CALCUL (À mettre au début) ---
def calculate_metrics(T_vec, dS_vec, H_max):
    idx_max = np.argmax(dS_vec)
    dS_max = dS_vec[idx_max]
    tc = T_vec[idx_max]   
    half_max = dS_max / 2
    idx_above = np.where(dS_vec >= half_max)[0]
    if len(idx_above) > 1:
        fwhm = T_vec[idx_above[-1]] - T_vec[idx_above[0]]
        rcp = dS_max * fwhm
        rc = np.trapezoid(dS_vec[idx_above], T_vec[idx_above]) # Intégrale réelle
    else:
        fwhm, rcp, rc = 0, 0, 0  
    nrc = rc / H_max if H_max > 0 else 0
    return dS_max, tc, fwhm, rcp, rc, nrc
def get_image_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    return buf.getvalue()
def calculate_Cp(T, dS):
    # Relation thermodynamique : Cp = T * (dDeltaS / dT)
    d_dS_dT = np.gradient(dS, T)
    cp_mag = T * d_dS_dT
    # Lissage pour un rendu "Publication" sans bruit
    return savgol_filter(cp_mag, window_length=15, polyorder=3)
# Bloc de configuration pour un style scientifique "fin" et élégant
# ==========================================
# CONFIGURATION STYLE SCIENTIFIQUE PRO
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", # Rendu LaTeX élégant
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.linewidth": 1.5,      # Cadre plus épais
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "xtick.major.width": 1.2,
    "ytick.major.size": 6,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",    # Graduations vers l'intérieur
    "ytick.direction": "in",
    "xtick.top": True,          # Graduations en haut
    "ytick.right": True,        # Graduations à droite
    "lines.linewidth": 1.5,
    "legend.fontsize": 10,
    "legend.frameon": False,    # Pas de cadre pour la légende
    "figure.dpi": 300           # Résolution d'impression
})
def petit_titre(texte):
    return st.markdown(f"**<p style='font-size:18px; margin-bottom:-10px;'>{texte}</p>**", unsafe_allow_html=True)
def moyen_titre(texte):
    return st.markdown(f"**<p style='font-size:20px; margin-bottom:-10px;'>{texte}</p>**", unsafe_allow_html=True)
# 1. Créez un dégradé de couleurs basé sur le nombre de courbes
# 'turbo' ou 'jet' imitent parfaitement votre photo de référence
# ================= 1. LECTURE ET NETTOYAGE ROBUSTE =================
def load_magneto_csv(file):
    # CRUCIAL : On remet le curseur au début du fichier pour éviter EmptyDataError
    file.seek(0)
    try:
        data = pd.read_csv(file, sep=",", decimal=".")
        if len(data.columns) < 2: raise ValueError
    except:
        file.seek(0) # On rembobine encore avant de tester le point-virgule
        data = pd.read_csv(file, sep=";", decimal=",")
    data = data.dropna()
    data = data.groupby("T").mean().reset_index().sort_values("T").reset_index(drop=True)
    M_cols = [c for c in data.columns if c.startswith("M_")]
    for col in M_cols:
        if len(data) > 15:
            data[col] = savgol_filter(data[col], window_length=15, polyorder=2)
    H_known = np.array([float(c.replace("M_","").replace("T","")) for c in M_cols])
    return data["T"].values, H_known, data[M_cols].values, data
# ================= 2. OUTILS PHYSIQUES =================
def power_law(h, a, n):
    return a * np.power(h, n)
def excel_export(df_main, df_stats):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Predictions', index=False)
        df_stats.to_excel(writer, sheet_name='Physique', index=False)
    return output.getvalue()


def apply_scientific_theme(ax, x_label="", y_label=""):
    """Applique un style 'Physical Review' à un axe Matplotlib."""
    # 1. Configuration des bordures (Spines)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 2. Graduations (Ticks) : Direction intérieure et présence sur les 4 côtés
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.2, top=True, right=True)
    ax.minorticks_on() # Active les petites graduations
    
    # 3. Labels avec police Serif et support LaTeX
    ax.set_xlabel(x_label, fontsize=12, fontfamily='serif')
    ax.set_ylabel(y_label, fontsize=12, fontfamily='serif')
    
    # 4. Suppression du cadre de la légende par défaut si elle existe
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_linewidth(0.0)
# Style Matplotlib Global
plt.rcParams.update({
    "lines.linewidth": 1.2, "font.family": "serif", "figure.dpi": 150,
    "xtick.direction": "in", "ytick.direction": "in", "axes.grid": False
})
# --- PAGE A : ACCUEIL ---
if st.session_state['page'] == 'accueil':
    col1, col2 = st.columns([1, 5])
    with col1:
        try: st.image("logo.png", width=180)
        except: st.title("🏫")
    with col2:
        st.markdown("## 🔬 Plateforme IA pour la Caractérisation Magnétocalorique")
        st.markdown("**Unité de Recherche Matériaux Avancés et Nanotechnologies (URMAN)**")
        st.write("ISSAT Kasserine, Université de Kairouan, Tunisie")
    st.divider()
    st.markdown("### 👋 Bienvenue")
    st.write("**Développeur :** DALHOUMI WALID")
    st.write("**Projet :** Analyse Avancée par Réseaux de Neurones (PFE)")
    st.info("Veuillez charger votre fichier de mesures magnétiques pour commencer.")
    file = st.file_uploader("Charger le fichier CSV (Format: T, M_1T, M_2T...)", type=["csv"])
    if file:
        st.session_state['uploaded_file'] = file
        st.session_state['last_file_name'] = file.name
        st.success(f"Fichier '{file.name}' prêt pour l'analyse !")
        if st.button("Lancer l'Analyse Avancée 🚀"):
            st.session_state['page'] = 'analyses'
            st.rerun()
# --- PAGE B : ANALYSES ---
elif st.session_state['page'] == 'analyses':
    if 'uploaded_file' not in st.session_state:
        st.error("Aucun fichier trouvé. Veuillez repasser par l'accueil.")
        st.stop()
    # --- INITIALISATION DE L'ÉTAT (À mettre tout en haut) ---
    if 'history' not in st.session_state:
        st.session_state.history = []
    # Barre Latérale (Sidebar)
    with st.sidebar:
        if st.button("🏠 Retour à l'accueil"):
            st.session_state['page'] = 'accueil'
            st.rerun()
        st.header("⚙️ Configuration")
        nodes_nn = st.slider("Nombre de Neurones (MLP)", 32, 256, 128)
        h_max_sim = st.number_input("Champ Max Simulation (T)", 0.0001, 15.0, 7.0)
        deltaT_tec = st.slider("Fenêtre TEC (?T en K)", 1, 10, 3)
        st.divider()
        if st.button("💾 Sauvegarder l'analyse actuelle"):
            # On vérifie si Tc existe (preuve que le calcul est fait)
            if 'Tc' in st.session_state:
                # On récupère le nom du fichier stocké ou "Inconnu"
                nom_fichier = st.session_state.get('last_file_name', "Inconnu")
                st.session_state.history.append({
                "Date": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Matériau": nom_fichier,
                "Tc (K)": round(st.session_state.Tc, 2),
                "dS_max": round(st.session_state.dS_max_val, 4),
                "RCP": round(st.session_state.rcp, 2)
                })
                st.success(f"Analyse de {nom_fichier} ajoutée !")
            else:
                st.warning("⚠️ Veuillez d'abord charger un fichier et lancer la simulation.")
        # Affichage de l'historique dans la sidebar
        if 'history' in st.session_state and len(st.session_state.history) > 0:
            st.subheader("📜 Historique des mesures")
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True)
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
        # Récupération des données
        file = st.session_state['uploaded_file']
        T, H_known, M_exp, data = load_magneto_csv(file)
    if file:
        # ON ENREGISTRE LE NOM DU FICHIER DANS LE STATE
        st.session_state.last_file_name = file.name
        # --- CHARGEMENT ---
        T, H_known, M_exp, data = load_magneto_csv(file)
        # --- ENTRAÎNEMENT IA ---
        with st.spinner("L'IA apprend la thermodynamique du matériau..."):
            # Grille d'entraînement dense
            H_dense_steps = np.linspace(H_known.min(), H_known.max(), 15)
            f_interp = interp1d(H_known, M_exp, axis=1, kind="linear", fill_value="extrapolate")
            M_train_dense = f_interp(H_dense_steps).T.ravel()
            Tg_train, Hg_train = np.meshgrid(T, H_dense_steps)
            X_train = np.column_stack([Tg_train.ravel(), Hg_train.ravel()])
            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_scaled = scaler_X.fit_transform(X_train)
            y_scaled = scaler_y.fit_transform(M_train_dense.reshape(-1,1)).ravel()
            model = MLPRegressor(hidden_layer_sizes=(nodes_nn, nodes_nn//2), activation="tanh", max_iter=3000, random_state=42)
            model.fit(X_scaled, y_scaled)
        # --- 1. GÉNÉRATION DES PRÉDICTIONS IA ---
            T_dense = np.linspace(T.min(), T.max(), 300) 
            H_sim = np.linspace(0.1, h_max_sim, 20)
            Tg, Hg = np.meshgrid(T_dense, H_sim)
            X_sim = scaler_X.transform(np.column_stack([Tg.ravel(), Hg.ravel()]))
            M_sim = scaler_y.inverse_transform(model.predict(X_sim).reshape(-1,1)).reshape(Hg.shape)
            # --- 2. CALCULS THERMODYNAMIQUES (Le moteur du PFE) ---
            dS_grid = np.zeros_like(M_sim)
            for i in range(len(H_sim)):
                # On calcule la dérivée pour chaque champ magnétique
                dS_grid[i, :] = np.gradient(M_sim[i, :], T_dense) * H_sim[i]
            # On récupère la courbe au champ maximum pour les calculs de performance
            dS_final = np.abs(dS_grid[-1, :]) 
            # --- 3. EXTRACTION DES MÉTRIQUES DE PERFORMANCE ---
            # On appelle la fonction définie en haut pour avoir Tc, RCP, RC, NRC
            dS_max_val, Tc, fwhm, rcp, rc, nrc = calculate_metrics(T_dense, dS_final, H_sim.max())
            # Calcul du TEC (Fenêtre de température utile)
            window = int(len(T_dense) * (deltaT_tec / (T_dense.max() - T_dense.min())))
            if window < 1: window = 1
            tec_list = [np.mean(dS_final[i:i+window]) for i in range(len(dS_final)-window)]
            tec = np.max(tec_list) if tec_list else 0
            # Calcul de la capacité calorique magnétique
            Cp_mag = calculate_Cp(T_dense, dS_final)
            # Calcul de la précision de l'IA
            y_pred_scaled = model.predict(X_scaled)
            r2 = r2_score(y_scaled, y_pred_scaled)
            # Variable pour la sidebar (évite les erreurs)
            Smax = dS_max_val
            # Préparation du tableau de données pour l'export Excel
            df_out = pd.DataFrame({
                "Température (K)": T_dense, 
                "M IA (emu/g)": M_sim[-1, :], 
                "Delta S (J/kg.K)": dS_final,
                "Delta Cp (J/kg.K)": Cp_mag
            }) 
            # --- À AJOUTER JUSTE AVANT LES ONGLETS ---
            st.session_state.Tc = Tc
            st.session_state.dS_max_val = dS_max_val
            st.session_state.rcp = rcp
    # CSS pour forcer l'affichage des onglets sur deux lignes
    st.markdown("""
        <style>
        /* 1. Espace entre les deux lignes et alignement global */
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: wrap;
            gap: 20px !important; /* Augmente l'espace horizontal entre onglets */
            row-gap: 15px !important; /* AJOUTE l'espace vertical entre les deux lignes */
            border-bottom: none;
        }
        /* 2. Style de chaque onglet */
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 15px !important;
            padding-right: 15px !important;
            background-color: #f0f2f6; /* Optionnel : un léger fond gris pour mieux les voir */
            border-radius: 8px; /* Bords arrondis pour un look moderne */
        }
        /* 3. On cache l'indicateur flottant qui bug */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        /* 4. Style de l'onglet actif (Ligne rouge et texte) */
        .stTabs [aria-selected="true"] {
            border-bottom: 4px solid #ff4b4b !important; /* Ligne rouge plus épaisse */
            color: #ff4b4b !important;
            background-color: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)
    # ================= 4. ONGLETS DE VISUALISATION =================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Magnétisation",
        "📊 Caractérisation Magnétique",
        "📈 Surface S 3D",
        "❄️ Entropie & Capacité",
        "📈 Arrott & Phase",
        "📋 Données & Export",
        "🔬 Scaling Universel",
        "🤖 Performance IA",
        "🔬 Physique Avancée"
    ])
    with tab1:
        c1, c2 = st.columns([1, 1], gap="xlarge")
        with c1:
            moyen_titre("Magnétisation M(T) : Comparaison IA")
            fig_mt, ax_mt = plt.subplots(figsize=(5, 4))
            # 1. Tracé des données expérimentales (en arrière-plan)
            colors = plt.cm.turbo(np.linspace(0, 1, len(H_known)))
            for i, h in enumerate(H_known):
                ax_mt.plot(T, M_exp[:, i], 'o', color=colors[i], alpha=0.2, ms=2)
            # 2. TON AJOUT : Prédiction IA au Champ Max (Contraste Noir)
            # M_sim[-1, :] correspond à la simulation pour h_max_sim
            ax_mt.plot(T_dense, M_sim[-1, :], 'k--', lw=1.8, 
                       label=f"IA Max ({H_sim[-1]:.1f} T)")
            # 3. Formatage scientifique
            ax_mt.set_xlabel("Température $T$ (K)")
            ax_mt.set_ylabel("Aimantation $M$ (emu/g)")
            ax_mt.legend(fontsize=8, loc='upper right', frameon=True)
            # Optionnel : Ajouter une grille très légère pour la lecture
            ax_mt.grid(True, linestyle=':', alpha=0.4) 
            fig_mt.tight_layout()
            st.pyplot(fig_mt)
            # --- BOUTON DE TÉLÉCHARGEMEN
            st.download_button("💾 Télécharger Graphe Magnétisation M(T) : Comparaison IA", 
                                data=get_image_bytes(fig_mt), 
                                file_name="Magnétisation M(T) : Comparaison IA.png",
                                key="btn_r2_mt")
            # --- BARRE DE PRÉCISION R² ---
            # On affiche le R² avec une couleur dynamique (vert si > 0.99)
            color_r2 = "normal" if r2 > 0.98 else "inverse"
            st.metric(label="Précision de l'apprentissage (R²)", value=f"{r2:.5f}", delta="Excellent" if r2 > 0.99 else "Bon")
            # Utilise st.info ou st.write avec ton texte entre guillemets UNIQUEMENT
            st.info("💡 **Note scientifique :** Un $R^2$ proche de 1.00000 indique que l'IA a parfaitement capturé la physique du matériau.")
        with c2:
            moyen_titre("Isothermes M(H) - IA")
            fig_mh, ax_mh = plt.subplots(figsize=(5, 4))
            n_curves = 8 
            indices = np.linspace(0, len(T_dense) - 1, n_curves, dtype=int)
            colors_h = plt.cm.viridis(np.linspace(0, 1, n_curves))
            for i, idx in enumerate(indices):
                ax_mh.plot(H_sim, M_sim[:, idx], 'o-', color=colors_h[i], 
                           label=f"{T_dense[idx]:.0f} K", ms=3, lw=1)
            ax_mh.set_xlabel("Champ H (T)")
            ax_mh.set_ylabel("M (emu/g)")
            ax_mh.legend(fontsize=7, ncol=2)
            fig_mh.tight_layout()
            st.pyplot(fig_mh)
            # --- BOUTON DE TÉLÉCHARGEMEN
            st.download_button("💾 Télécharger Graphe Isothermes M(H) - IA", 
                                data=get_image_bytes(fig_mh), 
                                file_name="Isothermes M(H) - IA.png",
                                key="btn_r2_mh")
    with tab2:
        c1, c2 = st.columns([1, 1], gap="xlarge")
        # --- COLONNE 1 : ANALYSE PHYSIQUE DE BASE ---
        with c1:
            moyen_titre("📊 Caractérisation Magnétique")
            # 1. Courbe dM/dT (Localisation de Tc)
            petit_titre("🌡️ Localisation de Tc")
            fig_deriv, ax_deriv = plt.subplots(figsize=(5, 4))
            dM_dT = np.gradient(M_sim[-1, :], T_dense)
            ax_deriv.plot(T_dense, dM_dT, 'r-', lw=2, label="Dérivée IA")
            idx_tc = np.argmin(dM_dT)
            tc_val = T_dense[idx_tc]
            ax_deriv.axvline(tc_val, color='k', linestyle='--', alpha=0.7)
            ax_deriv.set_xlabel("Température $T$ (K)")
            ax_deriv.set_ylabel("$dM/dT$ (emu/g.K)")
            ax_deriv.legend()
            st.pyplot(fig_deriv)
            # 2. Courbe Delta T Adiabatique
            petit_titre("🌡️ Réponse Thermique")
            fig_dt, ax_dt = plt.subplots(figsize=(5, 4))
            cp_val = 300 # Valeur moyenne pour les ferromagnétiques
            delta_T_ad = - (T_dense / cp_val) * dS_final
            ax_dt.plot(T_dense, delta_T_ad, color='crimson', lw=2)
            ax_dt.set_xlabel("Température $T$ (K)")
            ax_dt.set_ylabel("$\Delta T_{ad}$ (K)")
            st.pyplot(fig_dt)
            st.download_button("💾 Télécharger Graphe Localisation de Tc", 
                                data=get_image_bytes(fig_deriv), 
                                file_name="Localisation de Tc.png",
                                key="btn_r2_deriv")
            st.download_button("💾 Télécharger Graphe Réponse Thermique", 
                                data=get_image_bytes(fig_dt), 
                                file_name="Réponse Thermique.png",
                                key="btn_r2_dt")
        # --- COLONNE 2 : ANALYSE THERMODYNAMIQUE AVANCÉE ---
        with c2:
            moyen_titre("⚙️ Efficacité et Sensibilité")
            # 1. Courbe d(Delta S) / dM (Nature de la transition)
            petit_titre("Nature de la transition")
            fig_dsdm, ax_dsdm = plt.subplots(figsize=(5, 4))
            M_sim_max = M_sim[-1, :]
            deriv_ds_dm = np.gradient(dS_final, M_sim_max)
            ax_dsdm.plot(T_dense, deriv_ds_dm, color='purple', lw=2)
            ax_dsdm.set_xlabel("Température $T$ (K)")
            ax_dsdm.set_ylabel(r"$d\Delta S_{mag}/dM$")
            st.pyplot(fig_dsdm)
            # 2. Courbe Delta S / Delta H (Efficacité du champ)
            petit_titre("Efficacité du champ")
            fig_dsdh, ax_dsdh = plt.subplots(figsize=(5, 4))
            delta_H = H_sim[-1] - H_sim[0]
            efficacite = dS_final / delta_H
            ax_dsdh.plot(T_dense, efficacite, color='darkorange', lw=2)
            ax_dsdh.set_xlabel("Température $T$ (K)")
            ax_dsdh.set_ylabel(r"$\Delta S_{mag}/\Delta H$ (J/kg.K.T)")
            st.pyplot(fig_dsdh)
            st.download_button("💾 Télécharger Graphe Nature de la transition", 
                                data=get_image_bytes(fig_dsdm), 
                                file_name="Nature de la transition.png",
                                key="btn_r2_dsdm")
            st.download_button("💾 Télécharger Graphe Efficacité du champ", 
                                data=get_image_bytes(fig_dsdh), 
                                file_name="Efficacité du champ.png",
                                key="btn_r2_dsdh")                    
    with tab3:
        col_a, col_b = st.columns([1, 1], gap="xlarge")
        with col_a:
            moyen_titre("📈 Surface S 3D")
            # Création de la figure
            fig_3d = go.Figure(data=[go.Surface(
            z=np.abs(dS_grid), 
            x=T_dense, 
            y=H_sim, 
            colorscale='Viridis',
            hovertemplate='T: %{x}K<br>H: %{y}T<br>|ΔS|: %{z:.2f}<extra></extra>'
            )])
            # --- AUGMENTATION DE LA TAILLE ET MISE EN PAGE ---
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Température T (K)',
                    yaxis_title='Champ H (T)',
                    zaxis_title='|ΔS| (J/kg.K)',
                    aspectratio=dict(x=1, y=1, z=0.7) # Ajuste les proportions de la boîte 3D
                ),
                width=500,  # Largeur forcée
                height=300, # Hauteur augmentée pour agrandir le "carreau"
                margin=dict(l=0, r=0, b=0, t=40) # Réduit les marges blanches autour
            )
            # Affichage
            st.plotly_chart(fig_3d, use_container_width=True)
            # --- BOUTON DE TÉLÉCHARGEMENT HTML ---
            # Utile pour garder l'interactivité (rotation, zoom) dans le rapport
            buffer = io.StringIO()
            fig_3d.write_html(buffer, include_plotlyjs='cdn')
            html_bytes = buffer.getvalue().encode()
            st.download_button(
                label="📥 Télécharger la Surface 3D interactive (HTML)",
                data=html_bytes,
                file_name="surface_3d_interactive.html",
                mime="text/html",
                key="btn_download_3d"
            )
        with col_b:
            petit_titre("Heatmap de l'Effet Magnétocalorique")
            
            fig_heat = px.imshow(
            np.abs(dS_grid),
            x=T_dense,
            y=H_sim,
            aspect="auto", # Très important pour remplir l'espace
            labels=dict(x="Température (K)", y="Champ H (T)", color="|ΔS|"),
            color_continuous_scale='Viridis'
            )

            fig_heat.update_layout(
                height=300, # Augmente la hauteur
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(family="serif", size=14)
            )

            st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()
        
        
    with tab4:
        col_a, col_b = st.columns([1, 1], gap="xlarge")
        with col_a:
            moyen_titre("Analyse Thermodynamique: ΔS, RCP & Tc")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            
            # Trace tes données
            ax2.plot(T_dense, dS_final, color='black', lw=1.5, label=r"$|\Delta S_{mag}|$ (IA)")
            
            # LE SECRET DU LOOK PRO : Activer les ticks secondaires et intérieurs
            ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True)
            ax2.minorticks_on() # Ajoute les petites graduations intermédiaires
            
            # Labels avec LaTeX propre
            ax2.set_xlabel(r"Temperature $T$ (K)")
            ax2.set_ylabel(r"$|\Delta S_{mag}|$ (J$\cdot$kg$^{-1}\cdot$K$^{-1}$)")
            ax2.fill_between(T_dense, 0, dS_final, alpha=0.1, color='blue') 
            # 2. Calcul et tracé de la zone RCP (Rectangle)
            half_max = dS_max_val / 2
            idx_above = np.where(dS_final >= half_max)[0] 
            if len(idx_above) > 1:
                T_low, T_high = T_dense[idx_above[0]], T_dense[idx_above[-1]]
                # Dessin du rectangle RCP (orange semi-transparent)
                rect_rcp = plt.Rectangle((T_low, 0), T_high - T_low, dS_max_val, 
                                          color='orange', alpha=0.2, label=f"Zone RCP ({rcp:.1f} J/kg)")
                ax2.add_patch(rect_rcp)
                # Flèche pour la largeur à mi-hauteur (FWHM)
                ax2.annotate('', xy=(T_low, half_max), xytext=(T_high, half_max),
                             arrowprops=dict(arrowstyle='<->', color='black', lw=1))
                ax2.text((T_low + T_high)/2, half_max * 1.05, f'FWHM = {fwhm:.1f} K', 
                         ha='center', fontsize=7, fontweight='bold')
            # 3. Ligne verticale pour la Température de Curie (Tc)
            ax2.axvline(Tc, color='red', linestyle='--', lw=1.2, label=f"Tc = {Tc:.1f} K")
            # 4. Point d'entropie maximale
            ax2.scatter(Tc, dS_max_val, color='red', s=30, zorder=5)
            ax2.text(Tc, dS_max_val * 1.02, f'{dS_max_val:.3f}', ha='center', color='red', fontsize=9)
            # Mise en forme scientifique
            ax2.set_xlabel("Température (K)")
            ax2.set_ylabel(r"$|\Delta S_{mag}|$ (J/kg·K)")
            ax2.set_ylim(0, dS_max_val * 1.3) # On laisse de la place en haut pour les textes
            ax2.legend(fontsize=8, loc='upper right', frameon=True)
            fig2.tight_layout()
            st.pyplot(fig2)
            st.download_button(
                label="💾 Télécharger Graphe Analyse Thermodynamique: ΔS,RCP & Tc", 
                data=get_image_bytes(fig2), 
                file_name=f"Analyse Thermodynamique: ΔS,RCP & Tc.png",
                key="btn_ds_rcp_tc")
            st.info(f"""
            **Interprétation Automatique :**
            * Le matériau présente un pic d'entropie à **{Tc:.1f} K**, ce qui correspond à sa température de transition.
            * La capacité de refroidissement (**RCP**) est de **{rcp:.2f} J/kg**.
            * La largeur de travail (**FWHM**) de **{fwhm:.1f} K** indique l'étendue thermique de l'effet magnétocalorique.
            """)
            # Bouton de téléchargement haute résolution
            # On récupère le nom du fichier de manière sécurisée
            nom_pour_save = st.session_state.get('last_file_name', "Resultat").replace(".csv", "")
        with col_b:
            moyen_titre("Capacité Calorique Magnétique ΔCp")
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            # Utilisation de T_dense pour correspondre à la taille de Cp_mag
            ax3.plot(T_dense, Cp_mag, color='red', lw=1)
            ax3.axhline(0, color='black', lw=0.5, ls='--')
            ax3.set_xlabel("T (K)"); ax3.set_ylabel(r"$\Delta C_{mag}$ (J/kg·K)")
            fig3.tight_layout()
            st.pyplot(fig3)
            st.download_button("💾 Télécharger Graphe Capacité Calorique Magnétique ΔCp", 
                               data=get_image_bytes(fig3), 
                               file_name="Cp_mag.png",
                               key="btn_cp")
    with tab5:
        c3, c4 = st.columns([1, 1], gap="xlarge")
        with c3:
            moyen_titre("Arrott Plots (M² vs H/M)")
            fig_ar, ax_ar = plt.subplots()
            for i in range(0, len(H_sim), 4):
                ax_ar.plot(M_sim[i, :]**2, H_sim[i]/(M_sim[i, :] + 1e-9), label=f"{H_sim[i]:.1f}T")
                ax_ar.set_xlabel("M²"); ax_ar.set_ylabel("H/M"); st.pyplot(fig_ar)
        with c4:
            moyen_titre("Exposants Critiques")
            dS_max_list = [np.max(np.abs(row)) for row in dS_grid]
            try:
                popt, _ = curve_fit(power_law, H_sim, dS_max_list)
                n_exp = popt[1]
                st.metric("Exposant n (?Smax ? Hn)", f"{n_exp:.3f}")
                st.write("Si n ˜ 0.66 ? Modèle de Champ Moyen")
            except: st.write("Fit impossible")
            st.metric("Température de Curie (Tc)", f"{Tc:.2f} K")
        st.markdown("### 📦 Exportation Groupée")
        st.write("Téléchargez les 4 graphiques principaux de l'analyse magnétocalorique en un seul clic.")
        if st.button("🎁 Préparer le pack de graphiques"):
            # 1. Créer un buffer en mémoire pour le fichier ZIP
            buf = io.BytesIO() 
            with zipfile.ZipFile(buf, "x") as myzip:
                graphes = {
                    "1_fig_ar.png": fig_ar,
                    "2_fig_ar.png": fig_ar,
                    "3_fig_ar.png": fig_ar,
                    "4_fig_ar.png": fig_ar,
                }
                for name, fig in graphes.items():
                    # Sauvegarder chaque figure en PNG dans le ZIP
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
                    myzip.writestr(name, img_buf.getvalue())
            # 2. Bouton de téléchargement du ZIP final
            st.download_button(
                label="📥 Télécharger le dossier ZIP (4 Graphes)",
                data=buf.getvalue(),
                file_name="resultats_Arrott Plots (M² vs H/M).zip",
                mime="application/zip",
                key="btn_zip_export" # Clé unique pour éviter l'erreur DuplicateKey
            )
    with tab6:
        moyen_titre("📋 Analyse Quantitative des Performances")
        stats = pd.DataFrame({
            "Indicateur de Performance": [
                "Température de Curie (Tc)", 
                "Variation d'Entropie Max (dS_max)",
                "Relative Cooling Power (RCP)",
                "Refrigeration Capacity (RC)",
                "TEC (Temperature Every Core)",
                "NRC (Normalized RC)",
                "Largeur à mi-hauteur (FWHM)"
            ], 
            "Valeur": [
                f"{Tc:.2f} K", 
                f"{dS_max_val:.4f} J/kg.K", 
                f"{rcp:.2f} J/kg", 
                f"{rc:.2f} J/kg",
                f"{tec:.4f} J/kg.K",
                f"{nrc:.4f} J/kg.T",
                f"{fwhm:.2f} K"
            ],
            "Unité": ["K", "J/(kg·K)", "J/kg", "J/kg", "J/(kg·K)", "J/(kg·T)", "K"]
        })
        st.table(stats)
        # Bouton de sauvegarde dans l'onglet Données
        if st.button("💾 Sauvegarder dans l'historique", key="save_final"):
            # On vérifie que les calculs sont bien présents
            if 'Tc' in locals() and 'rcp' in locals():
                st.session_state.history.append({
                    "Matériau": file.name,
                    "Tc (K)": round(Tc, 2),
                    "RCP (J/kg)": round(rcp, 2),
                    "RC (J/kg)": round(rc, 2),
                    "TEC": round(tec, 4),
                    "R² IA": round(r2, 4)
                })
                st.success(f"Analyse de {file.name} sauvegardée !")
            else:
                st.error("Données de calcul introuvables.")
        st.download_button("💾 Exporter vers Excel (.xlsx)", 
                           data=excel_export(df_out, stats), 
                           file_name="Resultats_PFE.xlsx",
                           key="btn_excel")
    with tab7:
        moyen_titre("🔬 Analyse de Scaling Universel (IA-Enhanced)")
        
        # CORRECTION 1 : Abaisser le seuil à 0 pour accepter les faibles champs
        valid_indices = [i for i, h in enumerate(H_sim) if h > 0]
        
        # CORRECTION 2 : Vérifier si on a au moins une donnée
        if len(valid_indices) == 0:
            st.warning("⚠️ Aucune donnée de simulation disponible. Augmentez le Champ Max.")
        else:
            # CORRECTION 3 : Sécuriser le nombre de courbes (max 5 ou moins si pas assez de données)
            n_curves = min(5, len(valid_indices))
            selected_indices = np.unique(np.linspace(valid_indices[0], valid_indices[-1], n_curves, dtype=int))
            
            # Rendre le graphique plus large (figsize 10, 5)
            fig_u, ax_u = plt.subplots(figsize=(10, 5))
            
            for idx in selected_indices:
                h_val = H_sim[idx]
                ds_curve = np.abs(dS_grid[idx, :])
                ds_max_l = np.max(ds_curve)
                
                # Repérage de Tc local pour le scaling
                tc_l = T_dense[np.argmax(ds_curve)]
                
                # Calcul FWHM (Full Width at Half Maximum)
                half_max = ds_max_l / 2
                idx_above = np.where(ds_curve >= half_max)[0]
                
                if len(idx_above) > 1:
                    tr_low, tr_high = T_dense[idx_above[0]], T_dense[idx_above[-1]]
                    delta_tr = tr_high - tr_low
                else:
                    delta_tr = 1.0 # Sécurité contre division par zéro
                    
                # Normalisation physisque
                theta = (T_dense - tc_l) / delta_tr
                y_norm = ds_curve / ds_max_l
                
                ax_u.plot(theta, y_norm, label=f"{h_val:.3f} T")

            # Application du style "Publication" que nous avons créé
            apply_scientific_theme(ax_u, r"$\theta = (T - T_c) / \delta T_{r}$", r"$\Delta S / \Delta S_{max}$")
            
            ax_u.set_xlim([-3, 3])
            ax_u.legend(fontsize=9, frameon=False, loc='upper right')
            
            # Affichage large dans Streamlit
            st.pyplot(fig_u, use_container_width=True)
            
            # Bouton de téléchargement (placé à l'intérieur du 'else' pour éviter les erreurs)
            st.download_button(
                label="💾 Télécharger le Graphe (Haute Résolution)", 
                data=get_image_bytes(fig_u), 
                file_name="Scaling_Universel_IA.png",
                key="btn_r2_u"
            )

        st.info("💡 Le regroupement des courbes (collapse) confirme une transition du second ordre.")

    # Notes scientifiques en dehors du bloc conditionnel
    st.info("""
    **Interprétation Physique :** Si les courbes se superposent (Collapse), cela confirme l'universalité de la transition. 
    À très faible champ (< 0.5 T), des écarts peuvent apparaître à cause du régime de Rayleigh.
    """)
    with tab8:
        # Titre principal en dehors des colonnes pour qu'il soit bien centré/large
        moyen_titre("🤖 Métriques & Analyse de la fiabilité IA")
        # On définit les deux grandes colonnes principales
        main_c1, main_c2 = st.columns([1, 1], gap="xlarge")
        
        

        with main_c1:
            petit_titre("🤖 Métriques de l'Intelligence Artificielle")
            
            
           
            
            # Graphique de corrélation (Réel vs Prédit)
            fig_v, ax_v = plt.subplots(figsize=(6, 5))
            ax_v.scatter(y_scaled[::5], y_pred_scaled[::5], alpha=0.4, color='green', s=10, label="Prédictions")
            ax_v.plot([y_scaled.min(), y_scaled.max()], [y_scaled.min(), y_scaled.max()], 'r--', lw=1.5, label="Parfait")
            
            # Utilisation de ton thème scientifique personnalisé
            apply_scientific_theme(ax_v, "Valeurs Réelles (Normalisées)", "Valeurs Prédites (Normalisées)")
            ax_v.legend(frameon=False)
            
            st.pyplot(fig_v, use_container_width=True)
            st.success(f"✅ Modèle entraîné sur {len(X_train)} points.")
            # Sous-colonnes pour les chiffres clés 
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Précision du Modèle (R²)", f"{r2:.5f}")
            st.download_button("💾 Télécharger Graphe R²", 
                                data=get_image_bytes(fig_v), 
                                file_name="Performance_IA.png",
                                key="btn_r2")

        with main_c2:
            petit_titre("🔍 Analyse de la fiabilité : Résidus")
            
            # Calcul des résidus
            M_pred_train = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1,1)).ravel()
            residus = M_train_dense - M_pred_train
            
            # Graphique des résidus (plus large pour voir les oscillations)
            fig_res, ax_res = plt.subplots(figsize=(8, 5))
            scatter = ax_res.scatter(X_train[:, 0], residus, alpha=0.3, s=10, c=X_train[:, 1], cmap='plasma')
            ax_res.axhline(0, color='black', lw=1, ls='--')
            
            apply_scientific_theme(ax_res, "Température (K)", "Erreur $M_{exp} - M_{IA}$")
            
            # Ajout d'une barre de couleur pour le champ magnétique sur les résidus
            cbar = fig_res.colorbar(scatter, ax=ax_res)
            cbar.set_label('Champ H (T)', rotation=270, labelpad=15)
            
            st.pyplot(fig_res, use_container_width=True)
            
            st.info("""
            **Interprétation :** Une répartition aléatoire des points autour de la ligne pointillée (0) indique que l'IA n'a pas de biais systématique.
            """)
    with tab9:
        moyen_titre("🔬 Analyses Magnétiques Approfondies")
        c1, c2 = st.columns([1, 1], gap="xlarge")

        with c1:
            petit_titre("Inverse de la Susceptibilité $1/\chi$")
            # Calcul de chi DC (M/H) pour le champ le plus faible
            h_min_idx = 0 
            chi_dc = M_sim[h_min_idx, :] / (H_sim[h_min_idx] + 1e-9)
            inv_chi = 1 / (chi_dc + 1e-9)

            fig_chi, ax_chi = plt.subplots(figsize=(5, 4))
            ax_chi.plot(T_dense, inv_chi, 'g-', lw=1.5, label=f"H = {H_sim[h_min_idx]:.2f}T")
            
            # Ajustement linéaire sur la zone haute température (Paramagnétique)
            # On prend les 20% derniers points
            cut = int(len(T_dense) * 0.8)
            z = np.polyfit(T_dense[cut:], inv_chi[cut:], 1)
            p = np.poly1d(z)
            ax_chi.plot(T_dense[cut-20:], p(T_dense[cut-20:]), "r--", alpha=0.8, label="Fit Curie-Weiss")
            
            ax_chi.set_xlabel("Température (K)")
            ax_chi.set_ylabel("$1/\chi$ (g/emu)")
            ax_chi.legend()
            st.pyplot(fig_chi)
            st.info(f"**Température de Curie-Weiss ($\theta_p$) estimée :** {-z[1]/z[0]:.2f} K")

# Suivi historique dans la sidebar
if 'history' in st.session_state and file:
    st.sidebar.write(pd.DataFrame(st.session_state.history))        materials_results[name] = {
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

