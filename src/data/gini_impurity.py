import pandas as pd
import numpy as np

def convert_file_to_pd_df(filepath):
    """
    reads the csv file and converts it to Pandas dataframe
    :return: pd.DataFrame
    """
    df = pd.read_csv(filepath)
    return df


filepath = '/Users/annamariabugaj/PycharmProjects/decision-tree/src/data/mice.csv'
df = convert_file_to_pd_df(filepath)

df['target'] = (df.Index >= 4).astype('int')
df.drop('Index', axis=1, inplace=True)

sample_df = df.head(16)
sample_df.to_csv('/Users/annamariabugaj/PycharmProjects/decision-tree/src/data/sample_data.csv')

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
def show_Gini_for_each_feature(df):
    for column in df.columns[:-1]:
        gini_value = Gini_impurity(df[column])
        print(f'Gini impurity for {column}: {gini_value}')

print(show_Gini_for_each_feature(df))