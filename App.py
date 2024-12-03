#Mis à jour 02/12/2024
import warnings
import io
import base64
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash import Dash, dcc, html, Input, Output, callback_context, ALL

from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objs as go

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import networkx as nx
from fa2_modified import ForceAtlas2
from community import community_louvain
from dash import html
# Ignorer les avertissements de convergence
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def create_kmeans(df, columns, n_clusters=3):
    try:
        X = df[columns].dropna().values
        
        # Limiter le nombre de points pour accélérer le calcul
        if len(X) > 10000:
            X = X[:10000]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        fig_raw = px.scatter(x=X[:, 0], y=X[:, 1], color=labels,
                             labels={'x': columns[0], 'y': columns[1]},
                             title=f"K-means (raw) - {n_clusters} clusters")
        
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        labels_normalized = kmeans.fit_predict(X_normalized)
        
        fig_normalized = px.scatter(x=X_normalized[:, 0], y=X_normalized[:, 1], color=labels_normalized,
                                    labels={'x': f'{columns[0]} (normalized)', 'y': f'{columns[1]} (normalized)'},
                                    title=f"K-means (normalized) - {n_clusters} clusters")
        
        return fig_raw, fig_normalized
    except Exception as e:
        print(f"Erreur dans create_kmeans: {e}")
        return px.scatter(title="Erreur dans le calcul K-means"), px.scatter(title="Erreur dans le calcul K-means")

        
def Ccreate_heatmap(df, columns):
    numeric_columns = ['iyear','alternative', 'claimed', 'crit1', 'doubtterr', 'extended', 'individual', 'ishostkid', 'multiple', 'natlty1', 'ndays', 'nhostkid', 'nhours', 'nkill', 'nperps', 'nreleased', 'nwound', 'property', 'propvalue', 'specificity', 'success', 'suicide', 'vicinity', 'crit2', 'crit3', 'guncertain1', 'nhostkidus', 'nkillter', 'nkillus', 'nperpcap', 'nwoundte', 'nwoundus', 'ransom', 'ransomamt']

    # Sélectionner les colonnes qui sont à la fois dans le DataFrame et dans numeric_columns
    available_columns = list(set(df.columns) & set(numeric_columns))

    if not available_columns:
        print("Aucune colonne numérique valide trouvée.")
        return {}, {}  # Retourner des figures vides si aucune colonne valide n'est trouvée

    # Créer un nouveau DataFrame avec seulement les colonnes numériques disponibles
    numeric_df = df[available_columns]

    # Convertir en type numérique et gérer les valeurs manquantes
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    # Calculer les corrélations
    pearson_corr = numeric_df.corr(method='pearson')
    spearman_corr = numeric_df.corr(method='spearman')

    # Créer les heatmaps
    pearson_heatmap = px.imshow(
        pearson_corr,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        labels=dict(color="Coeff de corrélation"),
        title="Corrélation de Pearson"
    )
    
    spearman_heatmap = px.imshow(
        spearman_corr,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        labels=dict(color="Coeff de corrélation"),
        title="Corrélation de Spearman"
    )

        
    # Mise à jour des mises en page
    for heatmap in [pearson_heatmap, spearman_heatmap]:
        heatmap.update_layout(
            width=550,
            height=550,
            coloraxis_colorbar=dict(
                title="Coeff de corrélation",
                titleside="top",
                ticks="outside",
                tickfont=dict(size=12),
                thickness=10,
                len=0.6,
                yanchor="top",
                y=-0.5,
                xanchor="center",
                x=0.5,
                orientation="h"
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )

    return pearson_heatmap, spearman_heatmap

def load_and_prepare_data(file_path):
    try:
        # Supprime temporairement les avertissements de copie
        pd.options.mode.chained_assignment = None

        # Liste des colonnes à garder (inchangée)
        columns_to_keep = ['alternative','location', 'attacktype1_txt', 'city', 'claimed', 'country_txt', 'crit1', 'doubtterr', 'extended', 'gname', 'iday', 'imonth', 'individual', 'ishostkid', 'iyear', 'latitude', 'longitude', 'multiple', 'natlty1', 'ndays', 'nhostkid', 'nhours', 'nkill', 'nperps', 'nreleased', 'nwound', 'property', 'propextent_txt', 'propvalue', 'provstate', 'region_txt','region', 'specificity', 'success', 'suicide', 'targtype1_txt', 'vicinity', 'weaptype1_txt','claimmode_txt', 'crit2', 'crit3', 'guncertain1', 'hostkidoutcome_txt', 'kidhijcountry', 'motive', 'natlty1_txt', 'nhostkidus', 'nkillter', 'nkillus', 'nperpcap', 'nwoundte', 'nwoundus', 'propcomment', 'ransom', 'ransomamt', 'scite1', 'targsubtype1_txt', 'weapdetail', 'weapsubtype1_txt']

        # Chargement des données
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False, usecols=columns_to_keep)
        print(f"DataFrame chargé avec succès. Shape: {df.shape}")

        # Remplacement des valeurs 0 par 1 pour le mois et le jour
        df.loc[df['imonth'] == 0, 'imonth'] = 1
        df.loc[df['iday'] == 0, 'iday'] = 1

        # Création de la colonne datetime
        def create_date(year, month, day):
            try:
                return pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                return pd.NaT

        df['date'] = df.apply(lambda row: create_date(row['iyear'], row['imonth'], row['iday']), axis=1)

        # Conversion des colonnes numériques et gestion des valeurs manquantes
        df['nkill'] = pd.to_numeric(df['nkill'], errors='coerce').fillna(0)
        df['nwound'] = pd.to_numeric(df['nwound'], errors='coerce').fillna(0)
        df['casualties'] = df['nkill'] + df['nwound']
        df['propvalue'] = pd.to_numeric(df['propvalue'], errors='coerce')

        # Liste des colonnes numériques potentielles (inchangée)
        numeric_columns = ['iyear','alternative', 'claimed', 'crit1', 
                           'doubtterr', 'extended','individual','ishostkid',
                           'multiple','natlty1','ndays','nhostkid','nhours',
                           'nkill','nperps','nreleased','nwound','property',
                           'propvalue','specificity','success','suicide',
                           'vicinity','crit2','crit3','guncertain1',
                           'nhostkidus','nkillter','nkillus',
                           'nperpcap','nwoundte','nwoundus',
                           'ransom','ransomamt']

        # Sélectionner les colonnes qui sont à la fois dans le DataFrame et dans numeric_columns
        available_numeric_columns = list(set(df.columns) & set(numeric_columns))

        if not available_numeric_columns:
            print("Aucune colonne numérique valide trouvée.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        # Créer un nouveau DataFrame avec seulement les colonnes numériques disponibles
        numeric_df = df[available_numeric_columns].copy()

        # Convertir en type numérique et gérer les valeurs manquantes
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
            numeric_df[col].fillna(numeric_df[col].median(), inplace=True)

        # Gestion des valeurs manquantes pour les colonnes catégorielles
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')

        # Nettoyage des données pour le clustering
        df_cleaned = df.dropna(subset=['nkill', 'nwound']).copy()

        # Normalisation des données pour le clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_cleaned[['nkill', 'nwound']])

        # Application de K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_cleaned['cluster'] = kmeans.fit_predict(scaled_data)

        # Calcul des corrélations
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')

        # Division des colonnes en paquets de 10
        columns = pearson_corr.columns

        # Réactive les avertissements
        pd.options.mode.chained_assignment = 'warn'

        return df_cleaned, numeric_df, pearson_corr, spearman_corr, columns

    except Exception as e:
        print(f"Erreur lors du chargement ou du traitement des données: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []        

# Exemple d'utilisation de la fonction
#df, numeric_df, pearson_corr, spearman_corr, column_packages = load_and_prepare_data('globalterrorismdb_0718dist.csv')



def create_kmeans(df, columns, n_clusters=3):
        # Select only the relevant columns and drop rows with NaN values
        X = df[columns].dropna()
        
        if X.empty:
            raise ValueError("No valid data available for clustering.")
        
        # K-means without normalization
        kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster_raw'] = np.nan  # Initialize the cluster column
        df.loc[X.index, 'cluster_raw'] = kmeans_raw.fit_predict(X)
    
        # K-means with normalization
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        kmeans_normalized = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster_normalized'] = np.nan  # Initialize the cluster column
        df.loc[X.index, 'cluster_normalized'] = kmeans_normalized.fit_predict(X_scaled)
        
        # Create scatter plots
        fig_raw = px.scatter(df, x=columns[0], y=columns[1], color='cluster_raw', 
                             title=f"Clustering K-means (n={n_clusters}) sans normalisation")
        fig_normalized = px.scatter(df, x=columns[0], y=columns[1], color='cluster_normalized', 
                                    title=f"Clustering K-means (n={n_clusters}) avec normalisation")
        
        return fig_raw, fig_normalized



# Chargement des données
columns_to_keep = [
    'alternative', 'attacktype1_txt', 'city', 'claimed', 'country_txt',
    'crit1', 'doubtterr', 'extended', 'gname', 'iday', 'imonth',
    'individual', 'ishostkid', 'iyear', 'latitude', 'longitude',
    'multiple', 'natlty1', 'ndays', 'nhostkid', 'nhours',
    'nkill', 'nperps', 'nreleased', 'nwound', 'property',
    'propextent_txt', 'propvalue', 'provstate', 'region_txt',
    'region', 'specificity', 'success', 'suicide',
    'targtype1_txt', 'vicinity', 'weaptype1_txt','claimmode_txt',
    'crit2', 'crit3', 'guncertain1',
    'hostkidoutcome_txt', 'kidhijcountry', 'natlty1_txt',
    'nhostkidus', 'nkillter',
    'nkillus', 'nperpcap',
    'nwoundte',
    'nwoundus',
    'propcomment',
    'ransom',
    'ransomamt'
]

# Chargement du DataFrame
dfp = pd.read_csv('globalterrorismdb_0718dist.csv',
                 encoding='ISO-8859-1',
                 low_memory=False,
                 usecols=columns_to_keep)

# Définition des colonnes numériques pour PCA et t-SNE
numeric_columns = ['iyear','alternative', 'claimed', 'crit1', 'doubtterr', 'extended', 'individual', 'ishostkid', 'multiple', 'natlty1', 'ndays', 'nhostkid', 'nhours', 'nkill', 'nperps', 'nreleased', 'nwound', 'property', 'propvalue', 'specificity', 'success', 'suicide', 'vicinity', 'crit2', 'crit3', 'guncertain1', 'nhostkidus', 'nkillter', 'nkillus', 'nperpcap', 'nwoundte', 'nwoundus', 'ransom', 'ransomamt']

# Préparation des données pour PCA
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(dfp[numeric_columns])

# Normalisation des données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(imputed_data)

# Application de PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Ajout des résultats PCA au DataFrame
dfp['PCA1'] = pca_result[:, 0]
dfp['PCA2'] = pca_result[:, 1]

# Application de t-SNE sur un sous-échantillon
def compute_tsne(data):
    tsne = TSNE(n_components=2, perplexity=min(30, len(data) // 10), random_state=42)  # Ajustement de perplexité
    return tsne.fit_transform(data)

# Échantillonnage aléatoire pour t-SNE
sample_size = min(3000, len(dfp))
sample_df = dfp.sample(n=sample_size, random_state=42)
tsne_result = compute_tsne(scaled_data[:sample_size])  # Utiliser les données normalisées

# Ajout des résultats t-SNE au DataFrame échantillonné
sample_df['tSNE1'] = tsne_result[:, 0]
sample_df['tSNE2'] = tsne_result[:, 1]
    
# Fonction pour créer le graphique du réseau sans nœuds isolés

def create_network(df, target_column, min_degree):
    # Créer le graphe
    g = nx.from_pandas_edgelist(df, source='gname', target=target_column)

    # Filtrer les nœuds avec un degré minimum
    g = g.subgraph([n for n, d in g.degree() if d >= min_degree])

    # Utiliser ForceAtlas2 pour le layout
    forceatlas2 = ForceAtlas2(scalingRatio=2.0, gravity=1.0)
    positions = forceatlas2.forceatlas2_networkx_layout(g, pos=None, iterations=1000)

    # Préparer les données pour Plotly
    edge_x, edge_y = [], []
    for edge in g.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = [], []
    node_colors = []
    
    for node in g.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        
        # Déterminer la couleur du nœud en fonction de la cible sélectionnée
        if node in df['gname'].values:
            node_colors.append('#fc8d62')  # Orange pour les groupes terroristes
        elif target_column == "country_txt":
            node_colors.append('lightblue')
        elif target_column == "region_txt":
            node_colors.append('#ff9999')
        elif target_column == "attacktype1_txt":
            node_colors.append('#ffff99')
        elif target_column == "targtype1_txt":
            node_colors.append('lightgreen')
        elif target_column == "weaptype1_txt":
            node_colors.append('#800080')  # Violet pour les types d'armes
        elif target_column == "kidhijcountry":
            node_colors.append('#FFC0CB')  # Rose clair pour le pays d'enlèvement
        elif target_column == "hostkidoutcome_txt":
            node_colors.append('#00FFFF')  # Cyan pour l'issue des otages

    # Créer la figure Plotly
    fig = go.Figure()
    
    # Ajouter les arêtes
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                             line=dict(width=0.5, color='#888'),
                             hoverinfo='none',
                             mode='lines'))
    
    # Ajouter les nœuds
    node_sizes = [g.degree(node) * 3 for node in g.nodes()]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(size=node_sizes, color=node_colors, line_width=2)
    ))

    # Ajouter les étiquettes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='text',
        text=[node for node in g.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        hoverinfo='none'
    ))

    fig.update_layout(
        title=f"Global Terrorism Network: Groups and {target_column} (Filtered)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig    
# Exemple d'utilisation de la fonction
df, numeric_df, pearson_corr, spearman_corr, column_packages = load_and_prepare_data('globalterrorismdb_0718dist.csv')
#print("Shape of numeric_df:", numeric_df.shape)
#print("Columns of numeric_df:", numeric_df.columns.tolist())


# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.server.config['TIMEOUT'] = 600  # 5 minutes
# Création de l'interface utilisateur avec plusieurs onglets
app.layout = dbc.Container([
    html.Div(id='dummy-input', children=[], style={'display': 'none'}),
    dcc.Location(id='url', refresh=False),
    dbc.Tabs([
        dbc.Tab(label="Analyses principales", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Type d'analyse", className="card-title"),
                            dcc.Dropdown(
                                id='analysis-type',
                                options=[
                                    {'label': 'Analyse géographique', 'value': 'geo'},
                                    {'label': 'Analyse temporelle', 'value': 'time'},
                                    {'label': "Analyse des types d'attaques", 'value': 'attack'},
                                    {'label': "Analyse des armes", 'value': 'weapon'},
                                    {'label': "Analyse des victimes", 'value': 'casualties'},
                                    {'label': "Analyse des groupes terroristes", 'value': 'groups'},
                                    {'label': "Analyse des cibles", 'value': 'targets'},
                                    {'label': "Analyse des dommages", 'value': 'damage'}
                                ],
                                value='geo',
                                clearable=False,
                                className="mb-3"
                            ),
                            html.Div(id='sub-dropdown-container')
                        ])
                    ], className="mb-4 shadow-sm")
                ], md=3),
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label='Graphique principal', children=[
                            dcc.Graph(id='main-graph')
                        ]),
                        dbc.Tab(label='Heatmaps', children=[
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Corrélation de Pearson", className="text-center"),
                                    dcc.Graph(id='Ppearson-heatmap', style={'height': '400px'})
                                ], md=6),
                                dbc.Col([
                                    html.H5("Corrélation de Spearman", className="text-center"),
                                    dcc.Graph(id='Sspearman-heatmap', style={'height': '300px'})
                                ], md=6)
                            ])
                        ]),
                        dcc.Tab(label='K-means', children=[
                            dcc.Dropdown(
                                id='Kkmeans-n-clusters',
                                options=[{'label': i, 'value': i} for i in range(1, 6)],
                                value=3,
                                style={'width': '50%', 'margin': '10px','font-weight': 'bold'}
                            ),
                            dcc.Graph(id='kmeans-graph-raw'),
                            dcc.Graph(id='kmeans-graph-normalized')
                        ]),
                    ], className="mb-4")
                ], md=8)
            ])
        ]),
        dbc.Tab(label="Corrélations", children=[
            dbc.Row([
                dbc.Col(html.H1("Analyse des Corrélations du Terrorisme Global", className="text-center mb-4"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Corrélation de Pearson"),
                    dcc.Graph(id='pearson-heatmap')
                ], width=6),
                dbc.Col([
                    html.H3("Corrélation de Spearman"),
                    dcc.Graph(id='spearman-heatmap')
                ], width=6)
            ])
        ]),
        dbc.Tab(label="Clustering", children=[
            dbc.Row([
                dbc.Col(html.H1("Analyse de Clustering", className="text-center mb-4"), width=12)
            ]),
            dbc.Tabs([
                dbc.Tab(label="K-means", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='kmeans-column1-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[0],
                                placeholder="Sélectionnez la première colonne"
                            ),
                        ], width=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='kmeans-column2-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[1],
                                placeholder="Sélectionnez la deuxième colonne"
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Nombre de clusters", htmlFor="kmeans-n-clusters", style={'font-weight': 'bold'} ),
                            dcc.Dropdown(
                                id='kmeans-n-clusters',
                                options=[{'label': i, 'value': i} for i in range(2, 11)],
                                value=3,
                                style={'width': '50%', 'margin': '10px'}
                            ),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='kmeans-raw-plot')], width=6),
                        dbc.Col([dcc.Graph(id='kmeans-normalized-plot')], width=6),
                    ]),
                ]),
                dbc.Tab(label="DBSCAN", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='dbscan-column1-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[0],
                                placeholder="Sélectionnez la première colonne"
                            ),
                        ], width=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='dbscan-column2-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[1],
                                placeholder="Sélectionnez la deuxième colonne"
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Epsilon", htmlFor="dbscan-eps", style={'font-weight': 'bold'}),
                            dcc.Slider(
                                id='dbscan-eps',
                                min=0.1,
                                max=2,
                                step=0.1,
                                value=0.5,
                                marks={i/10: str(i/10) for i in range(1, 21)},
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Min Samples", htmlFor="dbscan-min-samples", style={'font-weight': 'bold'}),
                            dcc.Slider(
                                id='dbscan-min-samples',
                                min=2,
                                max=20,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(2, 21, 2)},
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([dbc.Col([dcc.Graph(id='dbscan-plot')], width=12)]),
                ]),
                dbc.Tab(label="Gaussian Mixture", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='gmm-column1-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[0],
                                placeholder="Sélectionnez la première colonne"
                            ),
                        ], width=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='gmm-column2-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_df.columns],
                                value=numeric_df.columns[1],
                                placeholder="Sélectionnez la deuxième colonne"
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Nombre de composants", htmlFor="gmm-n-components", style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                id='gmm-n-components',
                                options=[{'label': i, 'value': i} for i in range(2, 11)],
                                value=3,
                                style={'width': '50%', 'margin': '10px'}
                            ),
                        ], width=12),
                    ]),
                    dbc.Row([dbc.Col([dcc.Graph(id='gmm-plot')], width=12)]),
                ]),
            ], className="mt-4"),
        ]),
        dbc.Tab(label="Analyse Bootstrap", children=[
            dbc.Row([
                dbc.Col(html.H1("Comparaison avec Distribution Normale et KDE", className="text-center mb-4"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='bootstrap-kde-column-dropdown',
                        options=[{'label': col, 'value': col} for col in numeric_df.columns],
                        value=numeric_df.columns[0] if len(numeric_df.columns) > 0 else None,
                        clearable=False,
                        style={'width': '50%', 'padding': '10px', 'fontSize': 16}
                    ),
                    dcc.Graph(id='bootstrap-kde-plot')
                ], width=12)
            ]),
        ]),
        
        dbc.Tab(label="Analyse PCA et t-SNE", children=[
            dbc.Row([
                dbc.Col(html.H1("Analyse PCA et t-SNE", className="text-center mb-4"), width=12)
            ]),
            
            # Nouvelle ligne pour la sélection de la colonne
            dbc.Row([
                dbc.Col([
                    html.Label("Sélectionnez l'attribut de couleur :", style={'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=[
                            {'label': 'Pays', 'value': 'country_txt'},
                            {'label': 'Région', 'value': 'region_txt'},
                            {'label': "Type d'attaque", 'value': 'attacktype1_txt'},
                            {'label': 'Ville', 'value': 'city'},
                            {'label': 'Groupe terroriste', 'value': 'gname'},
                            {'label': 'Type de cible', 'value': 'targtype1_txt'},
                            {'label': "Type d'arme", 'value': 'weaptype1_txt'},
                            {'label': "Pays de l\'enlèvement", 'value': 'kidhijcountry'},
                            {'label': 'Issue de la prise d\'otages', 'value': 'hostkidoutcome_txt'},
                            {'label': 'Étendue des dégâts', 'value': 'propextent_txt'},
                            {'label': 'Province/État', 'value': 'provstate'},
                            {'label': 'Mode de revendication', 'value': 'claimmode_txt'},
                            {'label': 'Sous-type de cible', 'value': 'targsubtype1_txt'},
                        ],
                        value='region_txt',
                        clearable=False,
                        style={'width': '60%', 'padding': '10px', 'fontSize': 16}
                    ),
                ], width=4)
            ]),

            # Ligne pour les graphiques PCA et t-SNE
            dbc.Row([
                dbc.Col(dcc.Graph(id='pca-scatter'), width=6),  # Graphique pour PCA
                dbc.Col(dcc.Graph(id='tsne-scatter'), width=6)   # Graphique pour t-SNE
            ])
        ]),
        dbc.Tab(label='Graphe Réseau', children=[
            html.Div([
                html.Label("Select Target Attribute:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='target-dropdown',
                    options=[
                        {'label': 'Countries', 'value': 'country_txt'},
                        {'label': 'Regions', 'value': 'region_txt'},
                        {'label': 'Attack Types', 'value': 'attacktype1_txt'},
                        {'label': 'Target Types', 'value': 'targtype1_txt'},
                        {'label': 'Weapon Types', 'value': 'weaptype1_txt'},
                        {'label': "Kidnapping Country", "value": "kidhijcountry"},
                        {'label': "Hostage Outcome", "value": "hostkidoutcome_txt"},
                    ],
                    value='region_txt',
                    clearable=False,
                    style={'width': '50%'}
                ),
                
                html.Label("Minimum Degree of Nodes:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='min-degree-slider',
                    min=1,
                    max=20,
                    value=10,
                    marks={i: str(i) for i in range(1, 21)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Div(id='legend-container', style={'margin-top': '20px'})
            ], style={'width': '50%', 'margin': 'auto'}),
            
            dcc.Graph(id='network-graph', style={'height':'80vh'})  # Increased height for the graph
        ])
    ])
])

@app.callback(
    Output('pca-scatter', 
           'figure'),
    [Input('color-dropdown', 
           'value')]
)
def update_pca_graph(color_column):
    fig = px.scatter(
        dfp,
        x='PCA1',
        y='PCA2',
        color=color_column,
        hover_data=['country_txt', 
                    'gname', 
                    'attacktype1_txt'],
        labels={'PCA1': 
                "First Principal Component", 
                "PCA2": 
                "Second Principal Component"}
    )
    
    fig.update_layout(title=f"PCA Scatter Plot (colored by {color_column})")
    
    return fig

@app.callback(
    Output('tsne-scatter',"figure"),
     [Input('color-dropdown', "value")]
)
def update_tsne_graph(color_column):
    fig = px.scatter(
        sample_df,
        x='tSNE1',
        y='tSNE2',
        color=color_column,
        hover_data=['country_txt', 
                    'gname', 
                    'attacktype1_txt'],
        labels={'tSNE1': 
                "t-SNE Component 1", 
                "tSNE2": 
                "t-SNE Component 2"}
    )
    
    fig.update_layout(title=f"t-SNE Scatter Plot (colored by {color_column})")
    
    return fig

# Callback pour mettre à jour les graphiques en fonction de l'analyse exécutée

@app.callback(
    [Output('network-graph', 'figure'),
     Output('legend-container', 'children')],
    [Input('target-dropdown', 'value'),
     Input('min-degree-slider', 'value')]
)
def update_network_graph(selected_target, min_degree):
    fig = create_network(df, selected_target, min_degree)
    
    # Create the legend based on selected target
    legend_items = [
        html.Div([
            html.Div(style={'backgroundColor': '#fc8d62', 'width': 20, 'height': 20}),
            html.Span("Terrorist Groups", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "5px"})
    ]
    
    if selected_target == "country_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': 'lightblue', 'width': 20, 'height': 20}),
            html.Span("Countries", style={"margin-left": "5px"}),
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "region_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': '#ff9999', 'width': 20, 'height': 20}),
            html.Span("Regions", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "attacktype1_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': '#ffff99', 'width': 20, 'height': 20}),
            html.Span("Attack Types", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "targtype1_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': 'lightgreen', 'width': 20, 'height': 20}),
            html.Span("Target Types", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "weaptype1_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': '#800080', 'width': 20, 'height': 20}),
            html.Span("Weapon Types", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "kidhijcountry":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': '#FFC0CB', 'width': 20, 'height': 20}),
            html.Span("Kidnapping Country", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    elif selected_target == "hostkidoutcome_txt":
        legend_items.append(html.Div([
            html.Div(style={'backgroundColor': '#00FFFF', 'width': 20, 'height': 20}),
            html.Span("Hostage Outcome", style={"margin-left": "5px"})
        ], style={"display": "flex", "align-items": "center"}))
    
    return fig, legend_items
    
# Fonction pour créer une heatmap
def create_heatmap(corr_matrix, columns, title):
    fig = px.imshow(corr_matrix.loc[columns, columns],
                    labels=dict(x="Variables", y="Variables", color="Corrélation"),
                    x=columns,
                    y=columns,
                    color_continuous_scale="RdBu_r")
    
    fig.update_layout(
        width=650,
        height=700,
        title=title,
        coloraxis_colorbar=dict(
            title="Coeff de corrélation",
            titleside="top",
            ticks="outside",
            tickfont=dict(size=12),
            thickness=10,
            len=0.7,
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        margin=dict(l=20, r=50, t=50, b=100)  # Augmenté en bas pour la légende
    )
    
    return fig
# kmeans
from dash.exceptions import PreventUpdate
import time

@app.callback(
    [Output('kmeans-raw-plot', 'figure'),
     Output('kmeans-normalized-plot', 'figure')],
    [Input('kmeans-column1-dropdown', 'value'),
     Input('kmeans-column2-dropdown', 'value'),
     Input('kmeans-n-clusters', 'value')]
)
def update_kmeans_plots(column1, column2, n_clusters):
    if not column1 or not column2:
        raise PreventUpdate
    
    try:
        start_time = time.time()
        fig_raw, fig_normalized = create_kmeans(df, [column1, column2], n_clusters)
        end_time = time.time()
        
        if end_time - start_time > 30:  # Si le calcul prend plus de 30 secondes
            print("Le calcul K-means a pris trop de temps.")
            return px.scatter(title="Calcul trop long"), px.scatter(title="Calcul trop long")
        
        return fig_raw, fig_normalized
    except Exception as e:
        print(f"Erreur lors de la création des graphiques K-means : {e}")
        return px.scatter(title="Erreur dans le calcul"), px.scatter(title="Erreur dans le calcul")
#DBSCAN : 
@app.callback(
    Output('gmm-plot', 'figure'),
    [Input('gmm-column1-dropdown', 'value'),
     Input('gmm-column2-dropdown', 'value'),
     Input('gmm-n-components', 'value')]
)
def update_gmm_plot(column1, column2, n_components):
    if column1 and column2:
        X = df[[column1, column2]].dropna().values
        gmm = BayesianGaussianMixture(n_components=n_components,  init_params='random', random_state=42)
        labels = gmm.fit_predict(X)
        
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels,
                         labels={'x': column1, 'y': column2},
                         title=f"Gaussian Mixture (n={n_components})")
        return fig
    return {}


@app.callback(
    Output('dbscan-plot', 'figure'),
    [Input('dbscan-column1-dropdown', 'value'),
     Input('dbscan-column2-dropdown', 'value'),
     Input('dbscan-eps', 'value'),
     Input('dbscan-min-samples', 'value')]
)
def update_dbscan_plot(column1, column2, eps, min_samples):
    if column1 and column2:
        try:
            X = df[[column1, column2]].dropna().values
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels,
                             labels={'x': column1, 'y': column2},
                             title=f"DBscan (eps={eps}, min_samples={min_samples})")
            return fig
        except Exception as e:
            print(f"Error in DBSCAN: {e}")
            return px.scatter(title="Error in DBSCAN calculation")
    return px.scatter(title="Please select both columns")
    
# Callbacks pour les heatmaps
@app.callback(
    Output('pearson-heatmap', 'figure'),
    Input('dummy-input', 'children')
)
def update_pearson_heatmap(dummy):
    return create_heatmap(pearson_corr, pearson_corr.columns, "Corrélation de Pearson")

@app.callback(
    Output('spearman-heatmap', 'figure'),
    Input('dummy-input', 'children')
)
def update_spearman_heatmap(dummy):
    return create_heatmap(spearman_corr, spearman_corr.columns, "Corrélation de Spearman")

# Callback pour l'analyse bootstrap avec KDE
@app.callback(
    Output('bootstrap-kde-plot', 'figure'),
    Input('bootstrap-kde-column-dropdown', 'value')
)
def update_bootstrap_kde_plot(selected_column):
    # Données originales
    original_data = numeric_df[selected_column].dropna()
    
    # Générer des données normales
    mean = original_data.mean()
    std = original_data.std()
    generated_data = np.random.normal(mean, std, 1000)
    
    # Créer la figure avec seaborn pour histogramme et KDE
    plt.figure(figsize=(10, 6))
    
    # Histogramme des données originales avec KDE
    sns.histplot(original_data, kde=True, stat="density", label="Données Originales", color='blue', bins=30)
    
    # Histogramme des données générées avec KDE
    sns.histplot(generated_data, kde=True, stat="density", label="Distribution Normale Générée", color='orange', bins=30)
    
    plt.title(f"Comparaison de la distribution de {selected_column} avec une distribution normale")
    plt.xlabel(selected_column)
    plt.ylabel("Densité")
    
    plt.legend()
    
    # Convertir le graphique matplotlib en image pour Dash
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    # Créer une figure Plotly à partir de l'image
    fig = go.Figure(go.Image(source=f"data:image/png;base64,{data}"))
    
    return fig

@app.callback(
    Output('sub-dropdown-container', 'children'),
    Input('analysis-type', 'value')
)
def update_sub_dropdown(analysis_type):
    dropdown_options = {
        'geo': [
            {'label': 'Attaques par pays', 'value': 'country'},
            {'label': 'Attaques par région', 'value': 'region'},
            {'label': 'Carte des attaques', 'value': 'map'},
            {'label': 'Évolution géographique', 'value': 'geo_evolution'},
            {'label': 'Analyse par ville', 'value': 'city_analysis'}
        ],
        'time': [
            {'label': "Tendance annuelle", "value": "yearly"},
            {'label': "Tendance mensuelle", "value": "monthly"},
            {'label': "Jours de la semaine", "value": "weekday"},
            {'label': "Évolution temporelle par type d'attaque", "value": "time_attack_type"}
        ],
        'attack': [
            {'label': "Types d'attaques", "value": "types"},
            {'label': "Taux de réussite", "value": "success"},
            {'label': "Attaques suicides", "value": "suicide"},
            {'label': "Incidents étendus", "value": "extended"}
        ],
        'weapon': [
            {'label': "Types d'armes", "value": "types"},
            {'label': "Armes par région", "value": "by_region"},
            {'label': "Évolution des armes", "value": "evolution"},
            {'label': "Létalité des armes", "value": "lethality"}
        ],
        'casualties': [
            {'label': "Évolution annuelle", "value": "yearly"},
            {'label': "Par type d'attaque", "value": "by_attack"},
            {'label': "Par pays", "value": "by_country"},
            {'label': "Ratio morts/blessés", "value": "kill_wound_ratio"}
        ],
        'groups': [
            {'label': "Groupes les plus actifs", "value": "most_active"},
            {'label': "Évolution des groupes", "value": "evolution"},
            {'label': "Zones d'opération", "value": "areas"},
            {'label': "Méthodes préférées", "value": "methods"}
        ],
        'targets': [
            {'label': "Types de cibles", "value": "types"},
            {'label': "Cibles par région", "value": "by_region"},
            {'label': "Évolution des cibles", "value": "evolution"},
            {'label': "Cibles les plus létales", "value": "lethality"}
        ],
        'damage': [
            {'label': "Types de dommages", "value": "types"},
            {'label': "Dommages par région", "value": "by_region"},
            {'label': "Évolution des dommages", "value": "evolution"},
            {'label': "Coût des dommages", "value": "cost"}
        ]
    }
    return dcc.Dropdown(
        id={'type': 'analysis-dropdown', 'index': analysis_type},
        options=dropdown_options.get(analysis_type, []),
        value=dropdown_options.get(analysis_type, [{}])[0]['value'],
        style={'width': '90%'}
    )

@app.callback(
    Output('cluster-plot', 'src'),
    [Input('cluster-x-axis', 'value'),
     Input('cluster-y-axis', 'value')]
)
def update_cluster_plot(x_axis, y_axis):
    # Normaliser les données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[x_axis, y_axis]])

    # Appliquer K-Means
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    # Créer le graphique
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='cluster', palette='viridis')
    plt.title("Clusters de K-Means")
    
    # Convertir le graphique en image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encoder l'image en base64
    img_str = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'


@app.callback(
    [Output('main-graph', 'figure'),
     Output('Ppearson-heatmap', 'figure'),
     Output('Sspearman-heatmap', 'figure'),
     Output('kmeans-graph-raw', 'figure'),
     Output('kmeans-graph-normalized', 'figure')],
    [Input('analysis-type', 'value'),
     Input({'type': 'analysis-dropdown', 'index': ALL}, 'value'),
     Input('Kkmeans-n-clusters', 'value')]
)
def update_graphs(analysis_type, analysis_values, n_clusters):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]['prop_id'] == 'analysis-type.value':
        return px.scatter(title="Sélectionnez une analyse spécifique"), {}, {}, {}, {}
    
    analysis_value = analysis_values[0] if analysis_values else None
    if analysis_value is None:
        raise PreventUpdate

    # Define analysis functions
    analysis_functions = {
        "geo": geo_analysis,
        "time": time_analysis,
        "attack": attack_analysis,
        "weapon": weapon_analysis,
        "casualties": casualties_analysis,
        "groups": groups_analysis,
        "targets": targets_analysis,
        "damage": damage_analysis,
    }
    
    main_figure = analysis_functions[analysis_type](analysis_value)
    
    heatmap_columns = {
        'geo': ['specificity', 'vicinity', 'nkill', 'nwound'],
        'time': ['ndays', 'nhours', 'extended', 'nkill', 'nwound'],
        'attack': ['crit1', 'doubtterr', 'success', 'suicide', 'nkill', 'nwound', 'multiple', 'individual'],
        'weapon': ['nkill', 'nwound'],
        'casualties': ['nkill', 'nwound', 'propvalue', 'property'],
        'groups': ['nkill', 'nwound', 'ishostkid', 'nhostkid', 'nreleased'],
        'targets': ['nkill', 'nwound', 'natlty1'],
        'damage': ['nkill', 'nwound', 'propvalue', 'property']
    }
    
    columns = heatmap_columns.get(analysis_type, [])
    
    if columns:
        pearson_heatmap, spearman_heatmap = Ccreate_heatmap(df, columns)
        
        try:
            kmeans_figure_raw, kmeans_figure_normalized = create_kmeans(df, columns[:2], n_clusters)
        except ValueError as e:
            print(f"Error during KMeans clustering: {e}")
            kmeans_figure_raw, kmeans_figure_normalized = {}, {}
        
    else:
        pearson_heatmap, spearman_heatmap = {}, {}
        kmeans_figure_raw, kmeans_figure_normalized = {}, {}
    
    return main_figure, pearson_heatmap, spearman_heatmap, kmeans_figure_raw, kmeans_figure_normalized
    
        # Définir les colonnes pour la heatmap et le K-means en fonction du type d'analyse
    heatmap_columns = {
        'geo': ['latitude', 'longitude', 'region', 'specificity', 'vicinity','nkill', 'nwound'],
        'time': ['iyear', 'imonth', 'iday', 'ndays', 'nhours', 'extended','nkill', 'nwound'],
        'attack': ['crit1', 'doubtterr', 'success', 'suicide','nkill', 'nwound','multiple','individual'],
        'weapon': ['nkill', 'nwound','weaptype1'],
        'casualties': ['nkill', 'nwound', 'propvalue','propextent','property'],
        'groups': ['nkill', 'nwound','ishostkid','nhostkid','nreleased'],
        'targets': ['nkill', 'nwound','targtype1','natlty1'],
        'damage': ['nkill', 'nwound', 'propvalue','propextent','property']
    }

    columns = heatmap_columns.get(analysis_type, [])
    
    if columns:
        heatmap_figure = Ccreate_heatmap(df, columns)
        kmeans_figure = create_kmeans(df, columns[:2])  # Utiliser les deux premières colonnes pour K-means
    else:
        heatmap_figure = {}
        kmeans_figure = {}

    return main_figure, heatmap_figure, kmeans_figure
    
# Fonctions d'analyse géographique
# Options d'analyse géographique pour l'interface utilisateur


# Fonction d'analyse géographique complète
def geo_analysis(analysis_value):
    
    if analysis_value == "country":
        country_counts = df['country_txt'].value_counts().nlargest(20)
        return px.bar(x=country_counts.index, y=country_counts.values,
                      title="Top 20 des pays par nombre d'attaques",
                      labels={'x': "Pays", "y": "Nombre d'attaques"})

    elif analysis_value == "region":
        region_counts = df['region_txt'].value_counts()
        return px.pie(values=region_counts.values, names=region_counts.index,
                      title="Répartition des attaques par région")

    elif analysis_value == "map":
        return px.scatter_geo(df.sample(n=min(10000, len(df))),
                              lat='latitude',
                              lon='longitude',
                              color='region_txt',
                              hover_name='country_txt',
                              hover_data=['provstate', 'city', 'location'],
                              size='nkill',
                              projection='natural earth',
                              title="Carte mondiale des attaques terroristes")

    elif analysis_value == "geo_evolution":
        geo_evolution = df.groupby(['iyear', "region_txt"]).size().reset_index(name='count')
        return px.line(geo_evolution, x='iyear', y='count', color='region_txt',
                       title="Évolution des attaques par région au fil du temps")

    

    elif analysis_value == "city_analysis":
        if 'city' in df.columns:
            city_counts = df['city'].value_counts().nlargest(20)
            return px.bar(x=city_counts.index, y=city_counts.values,
                          title="Top 20 des villes par nombre d'attaques",
                          labels={'x': "Ville", "y": "Nombre d'attaques"})
        else:
            return px.bar(x=[], y=[], title="Données sur les villes non disponibles")

    else:
        return px.bar(x=[], y=[], title="Analyse non reconnue")
        
        
        
# Fonctions d'analyse temporelle
def time_analysis(analysis_value):

    if analysis_value == "yearly":
        yearly_attacks = df.groupby('iyear').size().reset_index(name='count')
        return px.line(yearly_attacks, x='iyear', y='count', 
                       title="Évolution annuelle du nombre d'attaques", 
                       labels={'iyear': "Année", "count": "Nombre d'attaques"})

    elif analysis_value == "monthly":
        df['month'] = df['imonth'].map({1:'Janv.', 2:'Févr.', 3:'Mars', 4:'Avr.', 5:'Mai', 6:'Juin', 
                                        7:'Juil.', 8:'Août', 9:'Sept.', 10:'Oct.', 11:'Nov.', 12:'Déc.'})
        monthly_attacks = df.groupby('month').size().reindex(['Janv.', 'Févr.', 'Mars', 'Avr.', 'Mai', 'Juin', 
                                                              'Juil.', 'Août', 'Sept.', 'Oct.', 'Nov.', 'Déc.'])
        return px.bar(x=monthly_attacks.index, y=monthly_attacks.values, 
                      title="Répartition mensuelle des attaques", 
                      labels={'x': "Mois", "y": "Nombre d'attaques"})

    elif analysis_value == "weekday":
        df['date'] = pd.to_datetime({
            'year': df['iyear'],
            'month': df['imonth'],
            'day': df['iday']
        })
        df['weekday'] = df['date'].dt.dayofweek
        weekday_attacks = df['weekday'].value_counts().sort_index()
        days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        return px.bar(x=days, y=weekday_attacks.values, 
                      title="Répartition des attaques par jour de la semaine", 
                      labels={'x':'Jour', "y":"Nombre d'attaques"})

    elif analysis_value == "time_attack_type":
        time_attack = df.groupby(['iyear','attacktype1_txt']).size().reset_index(name='count')
        return px.line(time_attack, x='iyear', y='count', color='attacktype1_txt', 
                       title="Évolution des types d'attaques au fil du temps")
        
# Fonctions d'analyse des attaques
def attack_analysis(analysis_value):
    
    if analysis_value == "types":
        attack_types = df['attacktype1_txt'].value_counts()
        return px.pie(values=attack_types.values, names=attack_types.index,
                      title="Répartition des types d'attaques")

    elif analysis_value == "success":
        success_rate = df['success'].value_counts(normalize=True) * 100
        return px.bar(x=success_rate.index.map({0: 'Échec', 1: 'Succès'}), y=success_rate.values,
                      title="Taux de réussite des attaques",
                      labels={'x': 'Résultat', 'y': 'Pourcentage'})

    elif analysis_value == "suicide":
        suicide_rate = df['suicide'].value_counts(normalize=True) * 100
        return px.bar(x=suicide_rate.index.map({0: 'Non-suicide', 1: 'Suicide'}), y=suicide_rate.values,
                      title="Proportion d'attaques suicides",
                      labels={'x': 'Type d\'attaque', 'y': 'Pourcentage'})

    elif analysis_value == "extended":
        extended_rate = df['extended'].value_counts(normalize=True) * 100
        return px.bar(x=extended_rate.index.map({0: 'Non-étendu', 1: 'Étendu'}), y=extended_rate.values,
                      title="Proportion d'incidents étendus",
                      labels={'x': 'Type d\'incident', 'y': 'Pourcentage'})

# Fonctions d'analyse des armes
def weapon_analysis(analysis_value):
    if analysis_value == "types":
        weapon_types = df['weaptype1_txt'].value_counts()
        return px.pie(values=weapon_types.values, names=weapon_types.index,
                      title="Répartition des types d'armes utilisées")

    elif analysis_value == "by_region":
        weapon_region = df.groupby(['region_txt','weaptype1_txt']).size().reset_index(name='count')
        return px.bar(weapon_region, x='region_txt', y='count', color='weaptype1_txt',
                      title="Types d'armes utilisées par région",
                      labels={'region_txt':'Région', 'count':'Nombre d\'attaques'})

    elif analysis_value == "evolution":
        weapon_evolution = df.groupby(['iyear','weaptype1_txt']).size().reset_index(name='count')
        return px.line(weapon_evolution, x='iyear', y='count', color='weaptype1_txt',
                       title="Évolution des types d'armes utilisées au fil du temps",
                       labels={'iyear':'Année', 'count':'Nombre d\'attaques'})

    elif analysis_value == "lethality":
        weapon_lethality = df.groupby('weaptype1_txt')['nkill'].mean().sort_values(ascending=False)
        return px.bar(x=weapon_lethality.index, y=weapon_lethality.values,
                      title="Létalité moyenne par type d'arme",
                      labels={'x':'Type d\'arme', 'y':'Nombre moyen de morts'})

# Fonctions pour l'analyse des victimes
def casualties_analysis(analysis_value):
    if analysis_value == "yearly":
        yearly_casualties = df.groupby('iyear')['casualties'].sum().reset_index()
        return px.line(yearly_casualties, x='iyear', y='casualties', 
                       title="Évolution annuelle du nombre de victimes", 
                       labels={'iyear':'Année', 'casualties':'Nombre de victimes'})

    elif analysis_value == "by_attack":
        attack_casualties = df.groupby('attacktype1_txt')['casualties'].sum().sort_values(ascending=False)
        return px.bar(x=attack_casualties.index, y=attack_casualties.values, 
                      title="Nombre de victimes par type d'attaque", 
                      labels={'x': 'Type d\'attaque', 'y': 'Nombre de victimes'})

    elif analysis_value == "by_country":
        country_casualties = df.groupby('country_txt')['casualties'].sum().nlargest(20)
        return px.bar(x=country_casualties.index, y=country_casualties.values, 
                      title="Top 20 des pays les plus touchés (en nombre de victimes)", 
                      labels={'x': 'Pays', 'y': 'Nombre de victimes'})

    elif analysis_value == "kill_wound_ratio":
        # Calculer le ratio morts/blessés
        df['kill_wound_ratio'] = df['nkill'] / (df['nwound'] + 1)  # Ajouter 1 pour éviter la division par zéro
        ratio_by_attack = df.groupby('attacktype1_txt')['kill_wound_ratio'].mean().sort_values(ascending=False)
        return px.bar(x=ratio_by_attack.index, y=ratio_by_attack.values, 
                      title="Ratio moyen morts/blessés par type d'attaque", 
                      labels={'x': 'Type d\'attaque', 'y': 'Ratio morts/blessés'})
        
# Fonctions pour l'analyse des groupes terroristes
def groups_analysis(analysis_value):

    if analysis_value == "most_active":
        top_groups = df['gname'].value_counts().nlargest(20)
        return px.bar(x=top_groups.index, y=top_groups.values, 
                      title="Top 20 des groupes terroristes les plus actifs", 
                      labels={'x': 'Groupe', 'y': 'Nombre d\'attaques'})

    elif analysis_value == "evolution":
        group_evolution = df.groupby(['iyear', 'gname']).size().reset_index(name='count')
        top_groups = df['gname'].value_counts().nlargest(10).index
        group_evolution = group_evolution[group_evolution['gname'].isin(top_groups)]
        return px.line(group_evolution, x='iyear', y='count', color='gname', 
                       title="Évolution des 10 groupes les plus actifs au fil du temps", 
                       labels={'iyear': 'Année', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "areas":
        group_areas = df.groupby(['gname', 'region_txt']).size().reset_index(name='count')
        top_groups = df['gname'].value_counts().nlargest(10).index
        group_areas = group_areas[group_areas['gname'].isin(top_groups)]
        return px.bar(group_areas, x='gname', y='count', color='region_txt', 
                      title="Zones d'opération des 10 groupes les plus actifs", 
                      labels={'gname': 'Groupe', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "methods":
        group_methods = df.groupby(['gname', 'attacktype1_txt']).size().reset_index(name='count')
        top_groups = df['gname'].value_counts().nlargest(10).index
        group_methods = group_methods[group_methods['gname'].isin(top_groups)]
        return px.bar(group_methods, x='gname', y='count', color='attacktype1_txt', 
                      title="Méthodes préférées des 10 groupes les plus actifs", 
                      labels={'gname': 'Groupe', 'count': 'Nombre d\'attaques'})
        
# Fonctions pour l'analyse des cibles
def targets_analysis(analysis_value):

    if analysis_value == "types":
        target_types = df['targtype1_txt'].value_counts()
        return px.pie(values=target_types.values, names=target_types.index,
                      title="Répartition des types de cibles")

    elif analysis_value == "by_region":
        target_region = df.groupby(['region_txt', 'targtype1_txt']).size().reset_index(name='count')
        return px.bar(target_region, x='region_txt', y='count', color='targtype1_txt',
                      title="Types de cibles par région",
                      labels={'region_txt': 'Région', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "evolution":
        target_evolution = df.groupby(['iyear', 'targtype1_txt']).size().reset_index(name='count')
        return px.line(target_evolution, x='iyear', y='count', color='targtype1_txt',
                       title="Évolution des types de cibles au fil du temps",
                       labels={'iyear': 'Année', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "lethality":
        target_lethality = df.groupby('targtype1_txt')['nkill'].mean().sort_values(ascending=False)
        return px.bar(x=target_lethality.index, y=target_lethality.values,
                      title="Létalité moyenne par type de cible",
                      labels={'x': 'Type de cible', 'y': 'Nombre moyen de morts'})
# Fonctions pour l'analyse des dommages
def damage_analysis(analysis_value):
    if analysis_value == "types":
        damage_types = df['propextent_txt'].value_counts()
        return px.pie(values=damage_types.values, names=damage_types.index,
                      title="Répartition des types de dommages")

    elif analysis_value == "by_region":
        damage_region = df.groupby(['region_txt', 'propextent_txt']).size().reset_index(name='count')
        return px.bar(damage_region, x='region_txt', y='count', color='propextent_txt',
                      title="Types de dommages par région",
                      labels={'region_txt': 'Région', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "evolution":
        damage_evolution = df.groupby(['iyear', 'propextent_txt']).size().reset_index(name='count')
        return px.line(damage_evolution, x='iyear', y='count', color='propextent_txt',
                       title="Évolution des types de dommages au fil du temps",
                       labels={'iyear': 'Année', 'count': 'Nombre d\'attaques'})

    elif analysis_value == "cost":
        df_with_value = df[df['propvalue'] > 0]
        damage_cost = df_with_value.groupby('propextent_txt')['propvalue'].mean().sort_values(ascending=False)
        return px.bar(x=damage_cost.index, y=damage_cost.values,
                      title="Coût moyen des dommages par type",
                      labels={'x': 'Type de dommage', 'y': 'Coût moyen (USD)'})

# Lancement de l'application
#app.run_server(debug=True, host="0.0.0.0")

if __name__ == '__main__':
    app.run_server(debug=True)
