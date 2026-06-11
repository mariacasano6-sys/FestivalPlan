import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Configurazione pagina
st.set_page_config(
    page_title="FestivalFit",
    page_icon="🎵",
    layout="wide"
)
    
@st.cache_data
def load_data():
    df_paese = pd.read_csv('data/paese_profilo.csv', index_col=0)
    df_artisti = pd.read_csv('data/artisti_profilo.csv')
    return df_paese, df_artisti



@st.cache_resource
def load_model():
    return joblib.load('models/pipeline_rf.pkl')

df_paese, df_artisti = load_data()
pipeline = load_model()

# Sidebar
st.sidebar.title("🎵 FestivalFit")
st.sidebar.subheader("Tool per promoter musicali")

pagina = st.sidebar.radio("Naviga", [
     "🗺️ Profilo paese",
     "🎤 Artist Match",
     "🌍 Clustering",
     "🔮 Predizione artista"
 ])

st.title("🎪 FestivalFit")
st.subheader("Data-driven tool per promoter musicali")

features_cluster = ['danceability', 'energy', 'valence', 'tempo',
                     'acousticness', 'speechiness', 'instrumentalness', 
                     'liveness', 'is_explicit', 'mode']

nomi_cluster = {
    0: 'High Energy Pop',
    1: 'Pop Mainstream',
    2: 'Latin Groove',
    3: 'Asian Acoustic',
    4: 'Brazilian Beat'
 }

nomi_paesi = {
     'AE': 'Emirati Arabi', 'AR': 'Argentina', 'AT': 'Austria',
     'AU': 'Australia', 'BE': 'Belgio', 'BG': 'Bulgaria',
     'BO': 'Bolivia', 'BR': 'Brasile', 'BY': 'Bielorussia',
     'CA': 'Canada', 'CH': 'Svizzera', 'CL': 'Cile',
     'CO': 'Colombia', 'CR': 'Costa Rica', 'CZ': 'Repubblica Ceca',
     'DE': 'Germania', 'DK': 'Danimarca', 'DO': 'Rep. Dominicana',
     'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egitto',
     'ES': 'Spagna', 'FI': 'Finlandia', 'FR': 'Francia',
     'GB': 'Regno Unito', 'GR': 'Grecia', 'GT': 'Guatemala',
     'HK': 'Hong Kong', 'HN': 'Honduras', 'HU': 'Ungheria',
     'ID': 'Indonesia', 'IE': 'Irlanda', 'IL': 'Israele',
     'IN': 'India', 'IS': 'Islanda', 'IT': 'Italia',
     'JP': 'Giappone', 'KR': 'Corea del Sud', 'KZ': 'Kazakistan',
     'LT': 'Lituania', 'LU': 'Lussemburgo', 'LV': 'Lettonia',
     'MA': 'Marocco', 'MX': 'Messico', 'MY': 'Malaysia',
     'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Olanda',
     'NO': 'Norvegia', 'NZ': 'Nuova Zelanda', 'PA': 'Panama',
     'PE': 'Perù', 'PH': 'Filippine', 'PK': 'Pakistan',
     'PL': 'Polonia', 'PT': 'Portogallo', 'PY': 'Paraguay',
     'RO': 'Romania', 'SA': 'Arabia Saudita', 'SE': 'Svezia',
     'SG': 'Singapore', 'SK': 'Slovacchia', 'SV': 'El Salvador',
     'TH': 'Tailandia', 'TR': 'Turchia', 'TW': 'Taiwan',
     'UA': 'Ucraina', 'US': 'Stati Uniti', 'UY': 'Uruguay',
     'VE': 'Venezuela', 'VN': 'Vietnam', 'ZA': 'Sud Africa'
 }

colori_cluster = {
     'High Energy Pop': '#FF6600',
     'Pop Mainstream': '#4488FF',
     'Latin Groove': '#FF4444',
     'Asian Acoustic': '#44BB44',
     'Brazilian Beat': '#FF9900'
 }

# ===========================
# PAGINA 1 - PROFILO PAESE
# ===========================

if pagina == "🗺️ Profilo paese":
    st.header("🗺️ Profilo sonoro per paese")
        
    paese_nome = st.selectbox( "Scegli un paese", 
        options= sorted(nomi_paesi.keys(), key=lambda x: nomi_paesi[x]),
        format_func=lambda x: nomi_paesi.get(x, x)
    )
    profilo = df_paese.loc[paese_nome]
    cluster_id = int(profilo['cluster'])
    cluster_name = nomi_cluster[cluster_id]
    colore = colori_cluster[cluster_name]
        
    st.subheader(f"Cluster: **{cluster_name}**")

    # Paesi che appartengono allo stesso cluster
    paesi_cluster = df_paese[df_paese['cluster'] == cluster_id].index.tolist()
    nomi_cluster_paesi = sorted([nomi_paesi.get(p, p) for p in paesi_cluster])
    st.markdown(f"🌍 **Paesi in questo cluster ({len(paesi_cluster)}):**")
    st.markdown(f"{', '.join(nomi_cluster_paesi)}")

    # Grafico a linee con i profili medi di tutti i cluster
    # Grafico a linee con i profili medi normalizzati di tutti i cluster
    st.subheader("📈 Confronto profili medi tra cluster")

    scaler_linee = MinMaxScaler()
    df_paese_norm_linee = pd.DataFrame(
        scaler_linee.fit_transform(df_paese[features_cluster]),
        index=df_paese.index,
        columns=features_cluster
    )
    df_paese_norm_linee['cluster'] = df_paese['cluster']

    medie_cluster = df_paese_norm_linee.groupby('cluster')[features_cluster].mean()
    medie_cluster.index = medie_cluster.index.map(nomi_cluster)

    fig_linee = go.Figure()
    for cname, row in medie_cluster.iterrows():
        fig_linee.add_trace(go.Scatter(
            x=features_cluster,
            y=row.values,
            mode='lines+markers',
            name=cname,
            line=dict(color=colori_cluster[cname], width=3 if cname == cluster_name else 1.5),
            opacity=1.0 if cname == cluster_name else 0.4
        ))

    fig_linee.update_layout(
        title='Valori medi delle features per cluster',
        xaxis_title='Feature',
        yaxis_title='Valore medio (normalizzato 0-1)',
        showlegend=True
    )
    st.plotly_chart(fig_linee, use_container_width=True)

    # Normalizziamo tutto df_paese con MinMaxScaler
    scaler_viz = MinMaxScaler()
    df_paese_norm = pd.DataFrame(
         scaler_viz.fit_transform(df_paese[features_cluster]),
         index=df_paese.index,
         columns=features_cluster
    )

    # Prendi il profilo normalizzato del paese scelto
    profilo_norm = df_paese_norm.loc[paese_nome]

    #Media Globale normalizzata
    media_norm = df_paese_norm.mean()

    #Valori normalizzati del paese
    valori = df_paese_norm.loc[paese_nome][features_cluster].tolist()
    valori += valori[:1] 

    fig = go.Figure(data=go.Scatterpolar(
         r=valori,
         theta=features_cluster + [features_cluster[0]],
         fill='toself',
         fillcolor=f'rgba{tuple(int(colore.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
         line=dict(color=colore)
    ))

    fig.update_layout(
         polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
         title=f'Profilo sonoro — {nomi_paesi[paese_nome]} ({cluster_name})',
         showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================
# PAGINA 2 - ARTIST MATCH
# ==================================

elif pagina == "🎤 Artist Match":
    st.header("🎤 Artist Match")
    st.subheader("Trova gli artisti più compatibili per il tuo festival")

    # Selezione paese
    paese_nome = st.selectbox(
         "Scegli il paese del festival",
         options=sorted(nomi_paesi.keys(), key=lambda x: nomi_paesi[x]),
         format_func=lambda x: nomi_paesi.get(x, x)
    )

    profilo = df_paese.loc[paese_nome]
    cluster_id = int(profilo['cluster'])
    cluster_name = nomi_cluster[cluster_id]
    colore = colori_cluster[cluster_name]

    st.info(f"🎵 Il pubblico di **{nomi_paesi[paese_nome]}** appartiene al cluster **{cluster_name}**")

    # Filtra artisti dello stesso cluster
    artisti_cluster = df_artisti[df_artisti['cluster'] == cluster_id].sort_values('popularity', ascending=False).copy()

    st.write(f"**{len(artisti_cluster)} artisti** compatibili con questo mercato")

    #Slider Popolarità
    pop_min = st.slider("Popolarità minima", 0,100,50)
    artisti_filtrati = artisti_cluster[artisti_cluster['popularity'] >= pop_min].copy()
    
    st.write(f"**{len(artisti_filtrati)} artisti** con popolarità ≥ {pop_min}")

    # Multiselect per scegliere la line-up
    artisti_scelti = st.multiselect(
         "🎤 Scegli gli artisti per la tua line-up",
         options=artisti_filtrati['artists'].tolist()
    )

    # Mostra tabella artisti filtrati
    st.dataframe(
         artisti_filtrati[['artists', 'popularity', 'cluster_name'] + features_cluster]\
         .reset_index(drop=True),
         use_container_width=True
    )

    # Mostra line-up scelta
    if artisti_scelti:
        st.subheader("🎪 La tua line-up")
        lineup = artisti_filtrati[artisti_filtrati['artists'].isin(artisti_scelti)]
        st.dataframe(lineup[['artists', 'popularity'] + features_cluster],
                    use_container_width=True)

        # Grafico radar per confrontare gli artisti scelti
        st.subheader("📊 Confronto profili sonori")
        
        fig = go.Figure()

        for _, artista in lineup.iterrows():
            # Normalizziamo i valori
            valori = []
            for f in features_cluster:
                min_val = df_artisti[f].min()
                max_val = df_artisti[f].max()
                val_norm = (artista[f] - min_val) / (max_val - min_val)
                valori.append(round(max(0, min(1, val_norm)), 3))
            valori += valori[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=valori,
                theta=features_cluster + [features_cluster[0]],
                fill='toself',
                name=artista['artists'],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Confronto profili sonori — line-up',
            showlegend=True
    )

        st.plotly_chart(fig, use_container_width=True)


# ==============================
# PAGINA 3 - CLUSTERING
# =============================

elif pagina == "🌍 Clustering":
    st.header("🌍 Clustering mondiale")
    st.subheader("I 72 paesi raggruppati per profilo sonoro")

    # Tabella paesi per cluster
    for cluster_id, cluster_name in nomi_cluster.items():
        paesi_cluster = df_paese[df_paese['cluster'] == cluster_id].index.tolist()
        nomi = [nomi_paesi.get(p, p) for p in paesi_cluster]

        st.markdown(f"### {cluster_name}")
        st.markdown(f"**{len(paesi_cluster)} paesi:** {', '.join(sorted(nomi))}")
        st.markdown("---")

    # Radar chart per ogni cluster
    st.subheader("📊 Profilo sonoro per cluster")

    fig = go.Figure()

    # Normalizza 
    scaler_viz = MinMaxScaler()
    df_paese_norm = pd.DataFrame(
        scaler_viz.fit_transform(df_paese[features_cluster]),
        index=df_paese.index,
        columns=features_cluster
    )

    for cluster_id, cluster_name in nomi_cluster.items():
        colore = colori_cluster[cluster_name]

        media_cluster = df_paese_norm[df_paese['cluster'] == cluster_id].mean()
        valori = media_cluster.tolist()
        valori += valori[:1]

        fig.add_trace(go.Scatterpolar(
            r=valori,
            theta=features_cluster + [features_cluster[0]],
            fill='toself',
            name=cluster_name,
            line=dict(color=colore),
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Profilo sonoro medio per cluster',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    
# # ===========================
# # PAGINA 4 - PREDIZIONE ARTISTA
# # ===========================
elif pagina == "🔮 Predizione artista":
    st.header("🔮 Predizione artista")
    st.subheader("Scopri dove funziona un artista nel mondo")

    artisti_ordinati = df_artisti.dropna(subset=['artists'])\
        .sort_values('popularity', ascending=False)['artists'].tolist()

    nome_artista = st.multiselect(
        "Cerca un artista",
        options=artisti_ordinati,
        max_selections=1
    )

    if nome_artista:
        artista = df_artisti[df_artisti['artists'] == nome_artista[0]].iloc[0]
        input_data = pd.DataFrame([artista[features_cluster].values],
                                  columns=features_cluster)
        cluster_pred = int(pipeline.predict(input_data)[0])
        cluster_pred_name = nomi_cluster[cluster_pred]
        colore_pred = colori_cluster[cluster_pred_name]

        # Cluster predetto
        st.success(f"**{nome_artista[0]}** appartiene al cluster: **{cluster_pred_name}**")

        # Paesi del cluster
        paesi_cluster = df_paese[df_paese['cluster'] == cluster_pred].index.tolist()
        nomi_consigliati = sorted([nomi_paesi.get(p, p) for p in paesi_cluster])
        
        st.markdown(f"🌍 **Mercati consigliati ({len(paesi_cluster)} paesi):**")
        st.markdown(f"{', '.join(nomi_consigliati)}")

        # Radar chart del cluster
        st.subheader(f"📊 Profilo sonoro del cluster {cluster_pred_name}")

        # Normalizzazione
        scaler_viz = MinMaxScaler()
        df_paese_norm = pd.DataFrame(
            scaler_viz.fit_transform(df_paese[features_cluster]),
            index=df_paese.index,
            columns=features_cluster
        )

        # Media normalizzata del cluster
        media_cluster = df_paese_norm[df_paese['cluster'] == cluster_pred].mean()
        valori = media_cluster.tolist()
        valori += valori[:1]

        fig = go.Figure(data=go.Scatterpolar(
            r=valori,
            theta=features_cluster + [features_cluster[0]],
            fill='toself',
            fillcolor=f'rgba{tuple(int(colore_pred.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=colore_pred)
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f'Profilo sonoro — {cluster_pred_name}',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

