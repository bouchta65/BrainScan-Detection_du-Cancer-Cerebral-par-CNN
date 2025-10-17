import streamlit as st

st.title("Explications : Chargement et Prétraitement du Dataset")

st.markdown("""
### 1. Importation des bibliothèques
On importe les bibliothèques nécessaires pour :
- Manipulation de fichiers et chemins (`os`, `Path`)
- Traitement d'images (`cv2`, `PIL`)
- Calculs numériques (`numpy`)
- Visualisation (`matplotlib`)
- Prétraitement des labels (`LabelEncoder`)
- Data augmentation (`ImageDataGenerator`)
- Gestion des ensembles d'entraînement et test (`train_test_split`, `to_categorical`)
""")

st.markdown("""
### 2. Chargement des images
- On définit le dossier contenant les images.
- On vérifie que chaque fichier est bien une image (`.jpg`, `.jpeg`, `.png`, `.bmp`).
- On utilise `try-except` pour gérer les fichiers corrompus ou non lisibles.
- On redimensionne toutes les images à 224x224 pour la compatibilité avec le CNN.
""")

st.markdown("""
### 3. Mélange et structuration des données
- Les images et labels sont mélangés ensemble pour éviter tout biais.
- On crée des tableaux NumPy pour que les données soient exploitables par le modèle.
""")

st.markdown("""
### 4. Exploration des données
- Visualisation du nombre d'images par classe.
- Affichage d'un échantillon d'image par classe.
- Vérification de l'équilibre des classes.
""")

st.markdown("""
### 5. Prétraitement des labels
- Conversion des labels textuels en numériques avec `LabelEncoder`.
- Transformation en encodage one-hot avec `to_categorical`.
""")

st.markdown("""
### 6. Data augmentation et rééquilibrage
- Utilisation de `ImageDataGenerator` pour générer de nouvelles images et équilibrer les classes.
- Normalisation des pixels dans la plage `[0,1]`.
""")

st.markdown("""
### 7. Division en ensembles d'entraînement et de test
- On utilise `train_test_split` pour séparer les données.
- Les fichiers finaux sont sauvegardés avec `np.save` dans le dossier `data/processed/`.
""")

st.markdown("""
> ⚠️ Note : Chaque étape ci-dessus correspond aux blocs de code que tu as utilisés dans ton script principal.  
> Les visualisations (`matplotlib`) permettent de vérifier graphiquement que le dataset est correctement chargé et équilibré.
""")
