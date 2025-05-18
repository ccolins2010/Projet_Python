import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import  precision_recall_fscore_support, precision_score, recall_score


#  definissons la fonction principale qui va contenire notre application
## -> Tout notre code sera à l'interrieur de cette fonction main()

def main():
    # Titre de notre app
    st.title("Application de machine learning pour la detection de fraude à la carte bancaire")
    
    st.subheader("Auteur: Colins")
    
    st.text("Ceci est une application qui permet la detection de fraude par des algorithmes de Machine Learning ")
    
    ## Importation des données
    
    #Pour garder  les resulats de nos données presedamment utilisées dans toute fois les recharger à chaque calcul
    @st.cache_data 
    # Definissons notre fonction pour lire et afficher nos données csv au nav
    def load_data():
        # Pour lire les jeux de données on utilises pd == pandas
        data = pd.read_csv('creditcard.csv')
        # comme toute fonction on doit retourner un resultat
        return data
    
    # affichage de la table de données
    # on defini la variable pour appler la fonction load_data qui nous permettra d'afficher notre DF au nav
    df = load_data()
    # Example de data frame
    df_sample = df.sample(100)
    
    #Pour le voir au nav => on utilise st.writre()
    #st.write(df)
    
    # Creer notre checkbox au nemu lateral pour que lorsqu'on clic dessus on affiche nore Df 
    # quand il est false c'est rien ne s'affiche mais quand il est true alors on l'affiche
    if st.sidebar.checkbox("Afficher les données brutes", False):
        # Donnons un sous titre qui s'affichera quand on cochera la checkbox
        st.subheader("Jeu de données 'creditcard': Echantillon de 100 observations")
        # Affichage au nav notre exemple de data frame
        st.write(df_sample)
    
    ## Jeux d'entrainement et de Test
    
    seed = 123
    
    
    # definition une fonction split()
    def split(df):
        
        #
        y = df['Class']
        x = df.drop('Class', axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,
            stratify = y,
            random_state = seed)
        
        return x_train, x_test, y_train, y_test
        
        
    # Creons notre model du jeux d'entrainement et de test  (Rq --> df est la variable qui appelle notre fonction load_data()
    x_train, x_test, y_train, y_test = split(df)
    
    
        # Prédiction personnalisée 
    st.sidebar.subheader("Prédiction personnalisée")

    if st.sidebar.checkbox("Faire une prédiction personnalisée"):
        st.subheader("Entrer les valeurs d'une transaction")

        # Création d'un dictionnaire pour les entrées utilisateur
        input_data = {}
        for col in df.drop('Class', axis=1).columns:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

        # Conversion en DataFrame pour prédiction
        input_df = pd.DataFrame([input_data])

        # Choix du modèle à utiliser pour la prédiction personnalisée
        model_type = st.selectbox("Choisir le modèle pour la prédiction", ("Random Forest", "SVM", "Logistic Regression"))

        if st.button("Prédire"):
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
            elif model_type == "SVM":
                model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
            elif model_type == "Logistic Regression":
                model = LogisticRegression(C=1.0, max_iter=100, random_state=seed)

            # Entraînement du modèle sur tout le dataset
            x = df.drop('Class', axis=1)
            y = df['Class']
            model.fit(x, y)

            # Prédiction
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            # Affichage des résultats
            st.markdown(f"### Résultat : {'🚨 **Transaction frauduleuse**' if prediction == 1 else '✅ **Transaction authentique**'}")
            if proba is not None:
                st.write(f"**Probabilité de fraude :** {round(proba, 3)}")
    
    
    ### On veut Expliquer le Label de la marix de confusion afin l'utilisateur comprenne que 0 = operation normal et 1 = operation forduleuse.
    
    # On cree une liste qu on l'appelle class_name avec T. Authentique == transaction 
    class_names = ['T. Authentique', 'T. Frauduleuse']
    
    classifier = st.sidebar.selectbox(
        "Classificateur",
        ("Random Forest", "SVM", "Logistic Regression")
    )
    
    ### Creer notre fonction des graphiques des performances qu'on appelera par  la suite plus bas pour nos graphique 
    
    # Fonction d Analyse de la performances des modèles qui permettra de tracer les graphiques
    
    def plot_perf(graphes, model, x_test, y_test):
        # Condition pour la courbe de la confusion
        if 'Confusion matrix' in graphes:
            st.subheader("Matrice de confusion")
            # Dessine la matrix de confusion
            # plot_confusion_matrix(model, x_test, y_test) on a rajouté display_labels = class_names qui est notre pour expliquer à l'utilisateur que 0 == Transaction authentique et 1 == Transaction frauduleuse
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            
            # Affichage du graphique au navigateur avec la fonction pyplot() 
            st.pyplot(fig)
            
        # Condition pour la courbe ROC
        if 'ROC Curve' in graphes:
            st.subheader("Courbe ROC")
            # Dessine la matrix de confusion
            # plot_roc_curve(model, x_test, y_test)  
            
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            
            # Affichage du graphique au navigateur avec la fonction pyplot() 
            st.pyplot(fig)
            
             # Affichage du graphique au navigateur avec la fonction pyplot() 
            # st.pyplot()  
            
          # Condition pour precision recall
        if 'Precision-Recall curve' in graphes:
            st.subheader("Courbe Precision-Recall")
            # Dessine la matrix de confusion
            # plot_precision_recall_curve(model, x_test, y_test)
            fig, ax = plt.subplots()
            
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            
            # Affichage du graphique au navigateur avec la fonction pyplot() 
            st.pyplot(fig) 
            
            # Affichage du graphique au navigateur avec la fonction pyplot() 
            # st.pyplot()      
            
            
     ### SVM Support Vector Machine (Machine à vecteur de support) 
     
    if classifier == "SVM":
        st.sidebar.subheader('Hyperparamètres du modèles')
        
        # 1er hyperparametre de regularisation iniverse (model = LogisticRegression(C=1.0)) on le fait varier de O,o1 à 10
        hyp_c = st.sidebar.number_input("Choisir la valeur du parametre de régularisation", 0.01, 10.0)
        
        # 2eme Hyperparametre 
        kernel = st.sidebar.radio("Choisir le kernel", ["rbf", "linear"])
        
        #3eme Hyperparametre
        gamma = st.sidebar.radio("Choisier une valeur de Gamme", ["scale", "auto"])
        
        
          # On cree un menu deroulant pour faire le choix de grapique de perfomance ML que l'utilisateur choisira
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du model ML",
                                              ["Confusion matrix", "ROC Curve", "Precision-Recall curve"])
        # Entrainons l'algorithme
        # Ici on veut un bouton de tel sorte que si l'utilisateur a fini de rentrer les données alors il clic et le tout l'algo s'eexécute
        if st.sidebar.button("Execution", key = "classify"):
            st.subheader("Support Vector Machine result")
            # Initialisation d'un modele objet SVC (Support Vector Classifier)
            model = SVC( C= hyp_c, kernel = kernel, gamma=gamma)
                 #Entrainement de l'algorithme avec un model
            model.fit(x_train, y_train)
            
            #Predictions 
            y_pred = model.predict(x_test)
            
            # Metrique de performance   (les round(2) == on arondit à 2 chiffres après la virgule)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            #Afficher les metriques dans l'application
            
            #Affichage de la variable accuracy (les round(accuracy, 2) == on arondit à 2 chiffres après la virgule)
            # Affichage de la variable accuracy
            st.write("Accuracy:", round(accuracy, 3))
            # Affichage de la variable precision
            st.write("Precision:", round(precision, 3))
            # Affichage de la variable recall
            st.write("Recall:",round(recall, 3))
            
            # Affichons les graphiques de performances en appellant la fonction plot_perf()
            
            #Affichage de 
            plot_perf(graphes_perf, model, x_test, y_test)
            
            
            
    ##REGRESSION LOGISTIQUE c'est model de ML (Permet de predire une variable categorielle exmple oui pour 0 et Non pour 1)
    
    if classifier == "Logistic Regression":
          # faisons le menu pour les hyperparamettre
        st.sidebar.subheader('Hyperparamètres du modèles') 
        st.sidebar.text("C(Paramètre de régularisation)") 
        # 1er hyperparametre de regularisation iniverse (model = LogisticRegression(C=1.0)) on le fait varier de O,o1 à 10
        hyp_c = st.sidebar.number_input("Choisir la valeur du parametre de régularisation", 0.01, 10.0)
        
        # nombre maximal d'iteration qu on appelra n_max_iter avec val min 100 val max = 1000 et evolue de 10 en 10 d'ou step = 10
        n_max_iter = st.sidebar.number_input("Choisir le nombre maximum iteration", 100, 1000, step=10)
        
          # On cree un menu deroulant pour faire le choix de grapique de perfomance ML que l'utilisateur choisira
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du model ML",
                                              ["Confusion matrix", "ROC Curve", "Precision-Recall curve"])
        
        # Entrainons l'algorithme
        # Ici on veut un bouton de tel sorte que si l'utilisateur a fini de rentrer les données alors il clic et le tout l'algo s'eexécute
        if st.sidebar.button("Execution", key = "classify"):
            st.subheader("Logistic Regression Result")
            # Initialisation d'un modele objet regression logistique classifier
            model = LogisticRegression(C = hyp_c, max_iter= n_max_iter, random_state= seed)
            
            #Entrainement de l'algorithme avec un model
            model.fit(x_train, y_train)
            
            #Predictions 
            y_pred = model.predict(x_test)
            
            # Metrique de performance   (les round(2) == on arondit à 2 chiffres après la virgule)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            #Afficher les metriques dans l'application
            
            #Affichage de la variable accuracy (les round(accuracy, 2) == on arondit à 2 chiffres après la virgule)
            # Affichage de la variable accuracy
            st.write("Accuracy:", round(accuracy, 3))
            # Affichage de la variable precision
            st.write("Precision:", round(precision, 3))
            # Affichage de la variable recall
            st.write("Recall:",round(recall, 3))
            
            # Affichons les graphiques de performances en appellant la fonction plot_perf()
            
            #Affichage de 
            plot_perf(graphes_perf, model, x_test, y_test)
            
            
            
        
    
    # RANDOM FOREST c'est un model de ML (foret aléatoire et à l'intérieur on a des arbres qui est le plus important)
    
    # 
    if classifier == "Random Forest":
        # faisons le menu pour les hyperparamettre == qui sera nos arbres dans la foret
        st.sidebar.subheader('Hyperparamètres du modèles')
        # Le 1er Hyperparametre est n_arbres
        n_arbres = st.sidebar.number_input("Coisir le nombre d'arbres dans la forêt", 100,
                                               1000, step=10)
        
        # 2eme Hyperparametres est la Profondeurs d'un arbre dans la forêt
        #1 == valeur min; 20 == Val max et Step = 1 veut dire qu'on evolue de 1 à 1 c a d on ait 1 puis 2, 3 ainsi de suite
        profondeur_arbre = st.sidebar.number_input("Choisir la profondeur maximale d'un arbre", 1, 20, step=1)
        
        # Le 3eme Hyperparametres est le bootsrap et on aura de valeur en parametre True et false
        # ce 3 eme parametres est facultatif d ou notre ?
        bootsrap = st.sidebar.radio("Echantillon bootsrap lors de la création d'arbres ?", ("True", "False"))
        
        
        # Graphique on l'a mis apres la création du code du bouton car il fallait tester les jeux de données fonctionnaient; du coup comme cela fonctionne alors on le met avant le if du button
        
        # On cree un menu deroulant pour faire le choix de grapique de perfomance ML que l'utilisateur choisira
        graphes_perf = st.sidebar.multiselect("Choisir un graphique de performance du model ML",
                                              ["Confusion matrix", "ROC Curve", "Precision-Recall curve"])
        
        
        
        
        
        # Entrainons l'algorithme
        # Ici on veut un bouton de tel sorte que si l'utilisateur a fini de rentrer les données alors il clic et le tout l'algo s'eexécute
        if st.sidebar.button("Execution", key = "classify"):
            st.subheader("Random Forest Results")
            # initialisation d'un objet randomforectclassifier
            model = RandomForestClassifier(n_estimators = n_arbres, max_depth = profondeur_arbre, bootstrap=True, random_state= seed)
            
            #Entrainement de l'algorithme avec un model
            model.fit(x_train, y_train)
            
            #Predictions 
            y_pred = model.predict(x_test)
            
            # Metrique de performance   (les round(2) == on arondit à 2 chiffres après la virgule)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            #Afficher les metriques dans l'application
            
            #Affichage de la variable accuracy (les round(accuracy, 2) == on arondit à 2 chiffres après la virgule)
            st.write("Accuracy:", round(accuracy, 3))
             #Affichage de la variable precision
            st.write("Precision:", round(precision, 3))
             #Affichage de la variable recall
            st.write("Recall", round(recall, 3))
            
            
            # Affichons les graphiques de performances en appellant la fonction plot_perf()
            
            #Affichage de 
            plot_perf(graphes_perf, model, x_test, y_test)
               
        
    
    
    ## cette condition assure la fonction main() est valide lorsque le script s'excute directement 
if __name__ == '__main__':
        main()
