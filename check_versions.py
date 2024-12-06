import warnings
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import networkx as nx
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash import Dash, Input, Output, callback_context, ALL
from dash.dependencies import Input, Output, ALL
from dash.exceptions import PreventUpdate
import plotly as plotly  # Importer le module principal de Plotly
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

# Afficher les versions des bibliothèques utilisées
print("Versions des bibliothèques :")
print(f"Numpy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")  # Utilisation de matplotlib pour obtenir la version correctement
print(f"Seaborn: {sns.__version__}")
print(f"NetworkX: {nx.__version__}")
print(f"Dash: {dash.__version__}")
print(f"Dash Bootstrap Components: {dbc.__version__}")
print(f"Plotly: {plotly.__version__}")  # Correction ici pour obtenir la version de Plotly
print(f"Scikit-learn: {KMeans.__module__.split('.')[0]} {KMeans.__module__.split('.')[1]}")  # Affiche la version de scikit-learn

# Ignore les avertissements de convergence lors de l'utilisation des modèles statistiques
warnings.filterwarnings("ignore", category=ConvergenceWarning)