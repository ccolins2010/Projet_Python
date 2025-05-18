# Application web de Machine Leraning de détection de fraude à la carte bancaire
Ce projet est une application de Machine Learning de détection de fraude à la carte bancaire. Il s'appuie sur un algorithme de classification (Random Forest, SVM, Logistic Regression) et propose une interface interactive via Streamlit et notre Modele de machine Learning sera faite avec Scikit-learn.

## Objectiive
L’objectif de cette application est de détecter automatiquement les fraudes à la carte bancaire à partir d’un jeu de données transactionnelles anonymisé (creditcard.csv) en utilisant des modèles supervisés de classification binaire et aussi de permettre l'utilisateur de verifier manuellement si une transaction est Frauduleuse ou Authentique.

## Scren-shot
PORJET_APP_ML\screen_shot\screen-1.png
PORJET_APP_ML\screen_shot\screen-2.png

## Live demo:
https://drive.google.com/file/d/1yp9n195ajQE898ed8Sr8qs6aeJR43cvK/view?usp=drive_link

## how it works
-  Pour notre application nous avons une page principale et une sidebar
-   Dans le menu latérale gauche nous avons 2 checkbox:

     1- Une 1ere checkbox « Afficher les données brutes »
    lorsque l’utilisateur coche la checkbox un jeux de données brutes s’affichent sur notre page principale et inversement.
    
     2- Une 2 eme checkbox «Faire une prédiction personnalisée »
    Ici lorsque l’utilisateur coche cette case  alors un formulaire s’ouvre sur la page principale de notre application:
    
        . Il rentre les données (heure, le montant par exemple)
        . Choisit le modèle
        . Puis clique sur « Prédire » , l application affichera comme résultat Transaction Frauduleuse ou Authentique
- Etant donné que c’est une tâche de Classification ; l’utilisateur pourra réaliser une prédiction automatique sur les nouvelles données , en faisant des prédictions sur le jeu de test (x_test).
- Pour cela on donne le  choix à l’utilisateur de pouvoir:
  
        ->  choisir  l algorithme qu’il souhaite (Random Forest, Support Vector Machine, Logistic Regression)
        ->  Régler les hyperparamètres
        ->  Choisir le modèle de graphique  pour visualiser les courbes de performances (Matrice de confusion, ROC curve et Precision-recall)
        ->  Clique sur le bouton « Exécution »
    i)   Alors le modèle est entraîné sur le jeu x_train et y_train.
  
    ii)  Il prédît les classes x_test (y_pred = model.predict(x_text).
  
    iii) Il affiche les métriques (accuracy, precision et recall) pour evaluer la performance du modèle choisit pour détecter les fraudes.
  
    iv)  Il trace les graphiques de performances(Confusion Matrix, ROC, Precision-recall)
  
## dataset creditcardfraud:
   -   Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   -   Il comporte les variables suivantes:
     
        . Time, v1...v28, Amount
       
        . Et Class qui sera notre variable cible car elle est binaire; 0 indique que la transaction n’est pas frauduleuse(authentique) et 1 indiquera que la transaction est frauduleuse

       

## Technologies:
Framework : Streamlit.

Librairies : scikit-learn, pandas, numpy
## Installation
### Clone the project
git clone https://github.com/ccolins2010/Projet_Python.git


1 Environement virtuel :

bash

python –m venv .venv

.venv\Scripts\activate

2 installation des dependences

Pip install –r requirement.txt

3 lancer l’appication 

streamlit run app.py


  





  
 




    

