from data.gini_impurity import *

class TestGiniImpurity:
    def test__convert_file_to_pd_df(self):
        csv_file = '/Users/annamariabugaj/PycharmProjects/decision-tree/src/data/sample_data.csv'
        df = convert_file_to_pd_df(csv_file)
        assert isinstance(df, pd.DataFrame), "Returned object is not a DataFrame."

   # def test_Gini_impurity(self):
