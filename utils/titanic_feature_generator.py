import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class Titanic_Feature_Generator(BaseEstimator, TransformerMixin):
    """Non-optimized feature generator
    Can be used as sklearn transformer
    Can filter columns (cols_to_drop or _keep parameters)"""
    def __init__(self, cols_to_drop=None, cols_to_keep=None):
        self.cols_to_drop = cols_to_drop
        self.cols_to_keep = cols_to_keep

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        self.create_features_from_name(X, fit=True)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        X = X.copy()
        X = self.create_features_from_name(X, fit=False)
        X = self.create_features_from_ticket(X)
        X = self.create_additional_feats(X)
        X = X.drop(columns=['name', 'ticket', 'cabin', 'sex'])
    
        if self.cols_to_drop:
            X = X.loc[:, ~X.columns.isin(self.cols_to_drop)]
        if self.cols_to_keep:
            X = X.loc[:, X.columns.isin(self.cols_to_keep)]
        return X

    def create_features_from_name(self, df, fit=False):
        df['name_l'] = df['name'].str.lower()
        
        # Clear punctuation signs
        df['name_clear'] = df['name_l'].str.strip()
        
        punct_signs = ['.', ',', '(', ')', '', "'", '"']
        for char in punct_signs:
            df['name_clear'] = df['name_clear'].str.replace(char, '', regex=False)\
                                                            .str.replace(' +', ' ', regex=True)
        
        # find titles
        df_titles = df['name_l'].str.extract(' ([a-z]+)\.', expand=False).value_counts()
        titles = (df_titles.index + ' ').tolist()
        
        # clear titles
        df['name_clear_no_t'] = df['name_clear'].str.replace('|'.join(titles), '', regex=True)\
                                                    .str.replace(' +', ' ', regex=True)

        if fit:
            self.common_titles = (df_titles[df_titles > 3].index + ' ').tolist()
            self.rare_titles = (df_titles[df_titles <= 3].index + ' ').tolist()
            self.top_10_names = list(df['name_clear_no_t'].str.split(expand=True).stack()\
                                                                .value_counts()[:10].index)
            # sorry for this
            return
             
        # create dataframes with single name part as values 
        name_expanded_no_m = df['name_clear_no_t'].str.split(expand=True)

        # lengths of individual name parts (titles were removed)
        name_lengths = name_expanded_no_m.apply(lambda x: x.str.len())
        
        # generate features
        df['name_size'] = name_expanded_no_m.shape[1] - name_expanded_no_m.isna().sum(axis=1)
        df['name_length'] = df['name_clear_no_t'].str.len()
        df['name_longest'] = name_lengths.max(axis=1)
        df['name_length_mean'] = name_lengths.mean(axis=1)
        df['popular_name'] = df['name_clear_no_t'].str.contains('|'.join(self.top_10_names), regex=True)\
                                                                            .astype(int)
        
        # has title name part
        for _title in self.common_titles:
            df[f'has_{_title}'] = df['name_clear'].str.contains(_title).astype(int)
            
        df['has_rare_title'] = df['name_clear'].str.contains('|'.join(self.rare_titles)).astype(int)
        
        df = df.drop(columns=['name_l', 'name_clear', 'name_clear_no_t'])

        return df

    @staticmethod
    def create_features_from_ticket(df): 
        df['ticket_num'] = df['ticket'].str.extract('(\d+)')
        df['ticket_num_len'] = df['ticket_num'].str.len()

        df['ticket_has_text'] = (df['ticket'].str.replace('\d+', '', regex=True).str.strip()!='').astype(int)
        
        df['n_cabins'] = df['cabin'].str.split(' ', expand=True).notna().sum(axis=1)
        df['has_cabin'] = (df['n_cabins'] != 0).astype(int)
        # only first letter if several
        df['cabin_letter'] = df['cabin'].str.replace('\d+', '', regex=True).str.split(' ', expand=True)[0]
        
        df = df.drop(columns=['ticket_num'])

        return df

    @staticmethod
    def create_additional_feats(df): 
        df['n_relatives'] = df['sibsp'].fillna(0) + df['parch'].fillna(0)
        df['alone'] = (df['n_relatives'] == 0).astype(int)
        df['sex_male'] = (df['sex'].fillna('male') == 'male').astype(int)
        return df