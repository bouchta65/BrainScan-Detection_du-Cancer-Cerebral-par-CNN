import streamlit as st

st.title("Guide Complet : Projet BrainScan - CNN pour détection du cancer cérébral")


st.header("1️⃣ Chargement et Prétraitement du Dataset")

st.subheader("1.1 Importation des bibliothèques")
st.markdown("""
Pour démarrer, nous importons toutes les bibliothèques nécessaires :  
1. `os` et `Path` pour gérer les fichiers et les dossiers.  
2. `cv2` et `PIL` pour le traitement des images.  
3. `numpy` pour les calculs numériques.  
4. `matplotlib` pour visualiser les images et statistiques.  
5. `LabelEncoder` et `to_categorical` pour préparer les labels.  
6. `ImageDataGenerator` pour l’augmentation de données.  
7. `train_test_split` pour séparer les ensembles d’entraînement et de test.
""")

st.subheader("1.2 Chargement et redimensionnement des images")
st.markdown("""
1. Définir le dossier contenant les images.  
2. Vérifier l’extension de chaque fichier (`jpg`, `jpeg`, `png`, `bmp`).  
3. Utiliser `try-except` pour gérer les fichiers corrompus.  
4. Redimensionner chaque image à 224x224 pour la compatibilité avec le CNN.
""")

st.subheader("1.3 Mélange et structuration des données")
st.markdown("""
1. Mélanger les images et leurs labels pour éviter tout biais.  
2. Transformer les listes en tableaux NumPy pour que le modèle puisse les exploiter.
""")

st.subheader("1.4 Exploration des données")
st.markdown("""
1. Afficher le nombre d’images par classe pour comprendre le dataset.  
2. Montrer un exemple d’image pour chaque classe.  
3. Vérifier l’équilibre entre les classes.
""")

st.subheader("1.5 Prétraitement des labels")
st.markdown("""
1. Convertir les labels textuels en numériques avec `LabelEncoder`.  
2. Transformer les labels en encodage one-hot avec `to_categorical`.
""")

st.subheader("1.6 Data augmentation et rééquilibrage")
st.markdown("""
1. Utiliser `ImageDataGenerator` pour générer de nouvelles images artificiellement (rotation, zoom, décalage).  
2. Rééquilibrer les classes si nécessaire pour éviter que le modèle favorise certaines classes.  
3. Normaliser les pixels dans la plage `[0,1]`.
""")

st.subheader("1.7 Division en ensembles d'entraînement et de test")
st.markdown("""
1. Séparer les données en training set et test set avec `train_test_split`.  
2. Sauvegarder les tableaux finaux (`x_train.npy`, `x_test.npy`, `y_train.npy`, `y_test.npy`) dans `data/processed/`.
""")


st.header("2️⃣ Conception du Modèle CNN")

st.subheader("2.1 Définition de l'architecture")
st.markdown("""
1. Ajouter des couches `Conv2D` pour extraire les motifs des images.  
2. Ajouter des couches `MaxPooling` pour réduire les dimensions progressivement.  
3. Ajouter des couches `Dropout` pour éviter l’overfitting.  
4. Ajouter des couches `Dense` pour la classification finale.
""")

st.subheader("2.2 Choix des fonctions d’activation")
st.markdown("""
1. `relu` pour toutes les couches cachées.  
2. `softmax` pour la couche de sortie afin d’obtenir des probabilités par classe.
""")

st.subheader("2.3 Compilation et vérification")
st.markdown("""
1. Compiler le modèle avec l’optimiseur Adam et la perte `categorical_crossentropy`.  
2. Vérifier la structure du modèle avec `model.summary()` et éventuellement `plot_model()`.
""")

st.subheader("2.4 Hyperparamètres et suivi")
st.markdown("""
1. Définir le taux d’apprentissage, le nombre d’époques et la taille du batch.  
2. Mesurer le temps d’entraînement avec la bibliothèque `time`.
""")


st.header("3️⃣ Entraînement et Évaluation du Modèle")

st.subheader("3.1 Entraînement")
st.markdown("""
1. Lancer l’entraînement avec `model.fit()`.  
2. Sauvegarder le meilleur modèle avec `ModelCheckpoint`.
""")

st.subheader("3.2 Évaluation")
st.markdown("""
1. Évaluer le modèle sur l’ensemble de test.  
2. Visualiser les courbes `loss` et `accuracy`.  
3. Générer une matrice de confusion et un rapport de classification.  
4. Afficher quelques prédictions correctes et incorrectes pour analyse.
""")


st.header("4️⃣ Déploiement et Utilisation")

st.subheader("4.1 Test d'une image")
st.markdown("""
Créer une fonction `predict_image(path)` qui prend une image et retourne la prédiction.
""")

st.subheader("4.2 Sauvegarde du modèle")
st.markdown("""
Sauvegarder le modèle entraîné au format `.h5` ou `.pt`.
""")

st.subheader("4.3 Interface utilisateur")
st.markdown("""
Développer une interface avec Streamlit pour permettre à l’utilisateur de télécharger une image et obtenir la prédiction directement.
""")

st.markdown("""
> 💡 Astuce : cette documentation est rédigée pour **expliquer le projet étape par étape à quelqu’un**.  
> Chaque section est structurée, hiérarchisée et claire pour faciliter la compréhension même pour un débutant.
""")
