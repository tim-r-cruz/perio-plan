import numpy as np 
import pandas as pd 


furcation_array = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']
mobility_array = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']
prognosis_array = ['Good', 'Fair', 'Poor', 'Questionable', 'Hopeless']

df = pd.DataFrame(
	np.random.randint(0.0, 10,size=(1000, 4)), 
	columns=['probingDepth',
			 'recession',
			 'clinicalAttachmentLoss',
			 'bleeding'
])

df['furcation'] = np.random.choice(furcation_array, 1000)
df['mobility'] = np.random.choice(mobility_array, 1000)
df['prognosis'] = np.random.choice(prognosis_array, 1000)

df.to_csv('data/synthetic_prognosis.csv', index=False)

print("Synthetic data generated!")