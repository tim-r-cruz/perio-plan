import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Periodontal Planning App 

This app predicts periodontal prognosis based on the McGuire and Nunn Classification System 

Data is randomly generated and used only for demonstration purposes.
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe

def user_input_features():
    furcation = st.sidebar.selectbox('FURCATION',('Grade I','Grade II','Grade III', 'Grade IV'))
    mobility = st.sidebar.selectbox('MOBILITY',('Grade 0','Grade 1', 'Grade 2', 'Grade 3'))
    probingDepth = st.sidebar.slider('Probing Depth (mm)', 0, 10, 5)
    recession = st.sidebar.slider('Recession (mm)', 0, 10, 5)
    clinicalAttachmentLoss = st.sidebar.slider('Clinical Attachment Loss (mm)', 0, 10, 5)
    bleeding = st.sidebar.slider('Bleeding (ml)', 0, 10, 5)
    data = {'furcation': furcation,
            'mobility': mobility,
            'probingDepth': probingDepth,
            'recession': recession,
            'clinicalAttachmentLoss': clinicalAttachmentLoss,
            'bleeding': bleeding}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
pronosis_raw = pd.read_csv('data/synthetic_prognosis.csv')
prognosis = pronosis_raw.drop(columns=['prognosis'])
df = pd.concat([input_df,prognosis],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['furcation','mobility']
for col in encode:
    dummy = pd.get_dummies(df[col])
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
# st.subheader('User Input features')
# st.write('Currently using example input parameters (shown below).')
# st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('models/prognosis_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

prognosis_array = np.array(['Good','Fair','Poor', 'Questionable', 'Hopeless'])
prognosis_prediction = str(prognosis_array[prediction])
st.subheader(f'Prognosis Prediction: {prognosis_prediction}')

st.subheader('Prognosis Prediction Probabilities')
df_probs = pd.DataFrame(prediction_proba, columns=prognosis_array)
st.write(df_probs)