import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 

prognosis = pd.read_csv("data/synthetic_prognosis.csv")

df = prognosis.copy() 
target = 'prognosis'
encode = ['furcation', 'mobility']

for col in encode: 
	dummy = pd.get_dummies(df[col], prefix=col)
	df = pd.concat([df, dummy], axis=1)
	del df[col]

target_mapper = {'Good':0, 'Fair':1, 'Poor':2, 'Questionable':3, 'Hopeless':4}
def target_encode(val):
    return target_mapper[val]

df['prognosis'] = df['prognosis'].apply(target_encode)

# Separating X and y
X = df.drop('prognosis', axis=1)
Y = df['prognosis']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('models/prognosis_clf.pkl', 'wb'))

print("Model created!")