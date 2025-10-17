# BrainScan AI - Détection du Cancer Cérébral par IA

**BrainScan AI** est une startup marocaine innovante spécialisée dans les technologies d’imagerie médicale assistée par l’intelligence artificielle.  
Elle vise à améliorer le diagnostic précoce des tumeurs cérébrales grâce à une approche automatisée et précise.

## Objectifs du projet
L’objectif principal est de développer une application intelligente capable de :  

- Analyser et classer les images IRM du cerveau pour détecter la présence de cancer.  
- Assister les médecins dans l’interprétation rapide et fiable des résultats.  
- Optimiser le temps de diagnostic tout en réduisant les erreurs humaines.  

Cette initiative s’inscrit dans une démarche combinant **santé, innovation et IA** pour renforcer la qualité et la rapidité des soins médicaux au Maroc.

---

## Chargement et Prétraitement du Dataset

1. Importer les bibliothèques nécessaires.  
2. Charger les images du dataset et vérifier leurs extensions (`jpeg`, `jpg`, `bmp`, `png`). Supprimer celles qui ne correspondent pas.  
3. Utiliser un bloc `try-except` pour gérer les erreurs lors du chargement.  
4. Explorer les classes du dataset (les noms des dossiers représentent les classes).  
5. Mélanger les images de toutes les classes dans deux listes :  
   - Une pour les **images**  
   - Une pour les **étiquettes**  
   > Chaque image et son label doivent avoir le même indice dans les deux listes.  
6. Redimensionner les images à une taille fixe (ex. 224×224) avec OpenCV.  
7. Convertir les listes en **tableaux NumPy** pour les rendre exploitables par le modèle CNN.  
8. Afficher graphiquement le nombre d’images dans chaque classe.  
9. Montrer un échantillon d’images pour chaque classe.  
10. Vérifier l’équilibre entre les classes et appliquer un rééquilibrage si nécessaire.  
11. Transformer les étiquettes textuelles en format numérique (`LabelEncoder` ou `to_categorical`).  
12. Diviser les données en ensembles **d’entraînement** et **de test**.  
13. Normaliser les valeurs des pixels dans la plage `[0, 1]`.

---

## Conception du Modèle CNN

1. Définir l’architecture CNN : `Conv2D + MaxPooling + Dropout + Dense`.  
2. Choisir les fonctions d’activation optimales pour les couches cachées et la couche de sortie.  
3. Compiler le modèle :  
   - Optimiseur : `Adam`  
   - Fonction de perte : `categorical_crossentropy`  
4. Vérifier l’architecture avec `model.summary()` et `plot_model()`.  
5. Déterminer les hyperparamètres :  
   - Taux d’apprentissage  
   - Nombre d’époques  
   - Taille de batch  
6. Utiliser la bibliothèque `time` pour mesurer la durée de l’entraînement.

---

## Entraînement et Évaluation du Modèle

1. Lancer l’entraînement avec `model.fit()`.  
2. Sauvegarder le **meilleur modèle** avec `ModelCheckpoint`.  
3. Évaluer les performances sur l’ensemble de test.  
4. Visualiser les **courbes d’apprentissage** (accuracy / loss).  
5. Générer une **matrice de confusion** et un **rapport de classification**.  
6. Afficher des exemples de **prédictions correctes et incorrectes**.

---

## Technologies utilisées

- Python 3.11  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Pillow (PIL)  
- Streamlit (interface web)  
- Matplotlib / Seaborn (visualisation)

---

