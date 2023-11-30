import streamlit as st 
import pandas as pd ##Exploration de données

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score

from sklearn.metrics import precision_score, f1_score


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#st.set_page_config(layout="wide")
def main(): 
    
    #### Partie A : Exploration de notre Dataset
    customers=pd.read_csv('healthcare-dataset-stroke-data.csv')

    st.title('Projet Framework Streamlit pour le machine learning')
    st.write("-----")
    taches=['1. Exploration de données', '2. Représentations graphiques', '3. Modèles de machine learning']

    choix=st.sidebar.selectbox('Selectionner une activité:', taches)
    st.subheader(choix)
    st.write("-----")
    if choix=='1. Exploration de données':
        # st.subheader('''
        # On veut faire de l'exploration de données''')

        affichages =['Afficher le dataset', 'Afficher les colonnes', 'Afficher Les variables', 'Le type des variables', 'Faire un summarise des données']
        affichage_select = st.sidebar.selectbox('Selectionner un affichage:', affichages)

        ##Show dataset
        if affichage_select=='Afficher le dataset':

            slider_ui= st.slider('''choisir l'intervalle de valeurs à afficher''', 1, customers.shape[0], (1000,3000))
            st.dataframe(customers.iloc[list(slider_ui)[0]: list(slider_ui)[1]])
        ##columns names
        if affichage_select=='Afficher les colonnes':
            column_select = st.sidebar.multiselect('Selectionner une colonne:', customers.columns)
            if column_select:
                customers[column_select]
            else:
                st.warning("Veuillez sélectionner au moins une colonne.")
        
        ##Shapes of dataset
        if affichage_select=='Afficher Les variables':
            description_columns = dict()
            description_columns['id'] = "dentifiant unique"
            description_columns['gender'] = " « Homme », « Femme » ou « Autre »"
            description_columns['age'] = "âge du patient"
            description_columns['hypertension'] = "0 si le patient ne souffre pas d'hypertension, 1 si le patient souffre d'hypertension"
            description_columns['heart_disease'] = "0 si le patient n'a aucune maladie cardiaque, 1 si le patient a une maladie cardiaque"
            description_columns['ever_married'] = "Non ou Oui"
            description_columns['work_type'] = "children, Govt_jov, Never_worked, privé ou Indépendant"
            description_columns['Residence_type'] = "Rural or Urban"
            description_columns['avg_glucose_level'] = "taux de glucose moyen dans le sang"
            description_columns['bmi'] = "indice de masse corporelle"
            description_columns['smoking_status'] = "anciennement fumé, jamais fumé, fume ou Inconnu"
            description_columns['stroke'] = "1 si le patient a eu un accident vasculaire cérébral ou 0 sinon"
            index = ['Description']
            description_columns_df = pd.DataFrame(description_columns, index = index)
            description_columns_df.T
        # ##Values Counts
        if affichage_select == 'Le type des variables':
            dtype_columns = dict()
            for colonne in customers.columns:
                dtype_columns[colonne] = customers[colonne].dtype
            index = ['Dtype']
            dtypes_columns_df = pd.DataFrame(dtype_columns, index = index)
            dtypes_columns_df.T

        ##Describle
        if affichage_select == 'Faire un summarise des données':
            st.write(customers.describe())

    if choix=='2. Représentations graphiques':
        plots =['Matrice de corrélations des variables quantitatives', 'Distribution plot']
        plot_select = st.sidebar.selectbox('Selectionner un plot:', plots)
        st.subheader(plot_select)
        if plot_select == 'Matrice de corrélations des variables quantitatives':
            df_numerique = customers.select_dtypes(include='number')
            df_numerique.drop('id', axis=1, inplace=True)
            matrice_corr = df_numerique.corr()

            # Utiliser Plotly Express pour créer un heatmap de la matrice de corrélations
            fig = px.imshow(matrice_corr, labels=dict(x="Variables", y="Variables", color="Corrélation"), x=matrice_corr.columns, y=matrice_corr.columns, color_continuous_scale="Viridis")

            # Afficher le graphique
            st.plotly_chart(fig)

           
        if plot_select == 'Distribution plot':
            pie_or_count =['Pie plot', 'Count plot']
            pie_or_count_select = st.sidebar.selectbox('Pie or count plot:', pie_or_count)
            col_selects = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
            if pie_or_count_select == 'Pie plot':
                columns_choose = st.sidebar.selectbox('Choisissez les colonnes a affichées:', col_selects)
                
                pie_data = customers[columns_choose]
                pie_plot_data = pie_data.value_counts()
                fig = px.pie(names=pie_plot_data.index, values=pie_plot_data.values, title=f'Distribution de {columns_choose}')
                st.plotly_chart(fig)

            else :
                col_selects = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
                columns_choose = st.sidebar.selectbox('Choisissez les colonnes a affichées:', col_selects)

                fig = px.histogram(customers, x=columns_choose, title=f'Distribution de {columns_choose}')
                st.plotly_chart(fig)
    
    if choix=='3. Modèles de machine learning':
        customers_predict = customers
        customers_predict.drop(['gender'], axis=1, inplace=True)
        customers_predict.drop(['smoking_status'], axis=1, inplace=True)
        customers_predict.drop(['ever_married'], axis=1, inplace=True)
        customers_predict.drop(['work_type'], axis=1, inplace=True)
        customers_predict.drop(['Residence_type'], axis=1, inplace=True)
        df_cleaned = customers_predict.dropna()
        stk_0 = df_cleaned[df_cleaned['stroke']==0]
        stk_1 = df_cleaned[df_cleaned['stroke']==1]

        stk_sample = stk_0.sample(n=249)

        df_echantillonnage = pd.concat([stk_sample, stk_1],ignore_index=True)
       
        X = df_echantillonnage.drop('stroke', axis=1)
        #st.write(X)
        y = df_echantillonnage['stroke']
        #st.write(y)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
        modeles = ["SVM", "KNN", "RandomForest", "Régression logistique"]
        model_select = st.sidebar.selectbox('Choisir un modéle:', modeles)
        st.subheader(model_select)
        if model_select == "SVM": 
            C = st.sidebar.slider("Choisissez le paramètre C pour SVM :", 0.1, 10.0, step=0.1, value=1.0)
            clf=SVC(C=C)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
       
            acc = accuracy_score(y_test, y_predict)
            score = precision_score(y_test, y_predict)
            scoref1 = f1_score(y_test, y_predict)

            st.write("Accuracy: ", round(acc, 2))
            st.write("Précision: ", round(score, 2))
            st.write("F1-score: ", round(scoref1, 2))
        
        if model_select == "RandomForest": 
            n_estimators = st.sidebar.slider("Choisissez le nombre d'estimateurs :", 1, 100, step=1, value=10)
            max_depth = st.sidebar.slider("Choisissez la profondeur maximale :", 1, 20, step=1, value=5)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

           
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Précision: ", round(precision, 2))
            st.write("F1-score: ", round(f1, 2))

        if model_select == "Régression logistique": 
            Crl = st.sidebar.slider("Choisissez le paramètre de régularisation C :", 0.1, 10.0, step=0.1, value=1.0)


            model = LogisticRegression(C=Crl, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculer les métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Afficher les métriques
            st.write(f"Accuracy : {accuracy:.2f}")
            st.write(f"Précision : {precision:.2f}")
            st.write(f"F1-score : {f1:.2f}")

        if model_select == "KNN": 
            n_neighbors = st.sidebar.slider("Choisissez le nombre de voisins (k) :", 1, 20, step=1, value=5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            st.write(f"Accuracy : {accuracy:.2f}")
            st.write(f"Précision : {precision:.2f}")
            st.write(f"F1-score : {f1:.2f}")


            


            
            
        





    
    
    # else:
    #     st.subheader('Représentations graphiques')

        








if __name__=='__main__':
    main()
