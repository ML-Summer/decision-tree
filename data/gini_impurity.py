import pandas as pd
import numpy as np

file = '/Users/annamariabugaj/PycharmProjects/decision-tree/data/mice.csv'
df = pd.read_csv(file)
print(df.head())
df['obese'] = (df.Index >= 4).astype('int')
df.drop('Index', axis = 1, inplace = True)

sample_df = df.head(16)
print(sample_df)

def Gini_impurity(feature: pd.Series):

    '''
    :param feature: pd.Series of feature values
    :return: Gini impurity for the given feature
    '''
    if isinstance(feature, pd.Series): # checking if object is a pd.Series
        proportion = feature.value_counts()/feature.shape[0]
        gini = 1-np.sum(proportion**2)
        return gini
    # if object is not a pd. Series raise Error-message
    else:
        raise ('Object must be a Pandas Series.')


for column in df.columns:
    gini_value = Gini_impurity(df[column])
    print(f'Gini impurity for {column}: {gini_value}')

