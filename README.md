# Analyse Avancée des Données du Terrorisme Mondial

## 🌍 Présentation du Projet

### Contexte
Ce projet de recherche vise à analyser les dynamiques globales du terrorisme en utilisant des techniques de data science.

### Objectifs
- Cartographier les tendances géographiques et temporelles des actes terroristes
- Identifier les facteurs de risque et les modèles émergents
- Développer des visualisations interactives et des insights exploitables

## 🛠 Technologies et Méthodologies


### Stack Technologique

Ce projet utilise un ensemble de bibliothèques Python pour l'analyse de données, la visualisation et le développement d'applications web. Voici les principales bibliothèques utilisées :

- **Langages** : Python 3.9+
  
- **Analyse de Données** :
  - **Pandas** : Pour la manipulation et l'analyse des données.
  - **NumPy** : Pour les opérations sur les tableaux et les calculs numériques.

- **Visualisation** :
  - **Matplotlib** : Pour la création de graphiques statiques.
  - **Seaborn** : Pour des visualisations statistiques améliorées.
  - **Plotly** : Pour des visualisations interactives et dynamiques.
  - **Dash** : Framework pour construire des applications web interactives avec des visualisations.

- **Analyse de Réseau** :
  - **NetworkX** : Pour la création et l'analyse de graphes.
  - **fa2_modified (ForceAtlas2)** : Pour la visualisation de réseaux avec l'algorithme ForceAtlas2.
  - **community (pour la détection de communautés)** : Pour identifier des communautés dans des graphes.

- **Machine Learning et Analyse Statistique** :
  - **Scikit-learn** : Pour le clustering (KMeans, DBSCAN), la réduction de dimension (PCA, t-SNE), et d'autres algorithmes d'apprentissage machine.

### Techniques Avancées
- Clustering (KMeans, DBSCAN)
- Réduction de dimensionnalité (PCA, t-SNE)
- Analyse de réseau
- Modèles de mélange gaussien

## 📊 Fonctionnalités Principales

1. **Analyse Exploratoire des Données**
   - Nettoyage et prétraitement des données
   - Analyse statistique descriptive
   - Identification des tendances et anomalies

2. **Visualisations Interactives**
   - Cartes géographiques dynamiques
   - Graphiques de clustering
   - Réseaux d'interactions terroristes

3. **Modélisation Prédictive**
   - Identification des zones à risque
   - Analyse des facteurs de propagation

## 🚀 Installation et Configuration

### Prérequis
- Python 3.9+
- Environnement virtuel recommandé

### Étapes d'Installation
```bash
# Cloner le dépôt
git clone https://github.com/ndouralbert/ProjetAnalyseDeDonnees.git

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# venv\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

## 📁 Structure du Projet

```
ProjetAnalyseDeDonnees/
│
├── app.py                  # Application Dash principale
├── README.md
└── requirements.txt        # Dépendances du projet
```
## 📊 Données

Pour télécharger les données du terrorisme (fichier globalterrorismdb_0718dist.csv), veuillez visiter le lien suivant :
[Global Terrorism Database sur Kaggle](https://www.kaggle.com/datasets/START-UMD/gtd)

Lien du Projet : [https://github.com/ndouralbert/ProjetAnalyseDeDonnees](https://github.com/ndouralbert/ProjetAnalyseDeDonnees)


## 🔍 Méthodologie de Recherche

### Collecte de Données
- Source principale : Global Terrorism Database (GTD)
- Période couverte : 1970-2017
- Critères de sélection : Tous les incidents terroristes enregistrés dans la base de données GTD

### Approche Méthodologique
1. Nettoyage et prétraitement des données
2. Analyse exploratoire
3. Clustering 
4. Réduction de dimension
5. Distribution des données
6. Graphe de réseau



## 📞 Contact 🤝

- Albert NDOUR - albert.ndour@etu.univ-lyon1.fr
- Chaimae DARDOURI - chaimae.dardouri@etu.univ-lyon1.fr
- FOUSSENI SALAMI CISSE TIDJANI - tidjani.fousseni-salami-cisse@etu.univ-lyon1.fr

