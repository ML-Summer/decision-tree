import pytest
from pytest import approx

from data.gini_impurity import *


class TestGiniImpurity:
    def test_convert_file_to_pd_df(self):
        csv_file = '/Users/annamariabugaj/PycharmProjects/decision-tree/src/data/sample_data.csv'
        df = convert_file_to_pd_df(csv_file)
        assert isinstance(df, pd.DataFrame), "Returned object is not a DataFrame."

    def test_Gini_impurity_invalid_input(self):
        # Test Gini impurity with an invalid input (not a pandas Series)
        with pytest.raises(TypeError):
            Gini_impurity([1, 2, 3, 4, 5])

    def test_Gini_impurity_valid_input(self):
        csv_path = '/Users/annamariabugaj/PycharmProjects/decision-tree/src/data/sample_data.csv'
        df = pd.read_csv(csv_path)
        feature = df['Height']
        gini = Gini_impurity(feature)
        assert gini == approx(0.9140625)


pytest.main()