from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WoE_Transformer(BaseEstimator, TransformerMixin):
    """
    Sklearn API WOE transformer
    During fit method iterates over dataframe columns.
        1) Groups variable (into intervals and categories) (for now by quantiles)
        2) Calculates WoE values for each group
        3) Optional: uses external filter to reduce (combine) groups in variable
    As a result creates dict with key=columns value=pd.series(interval/category:WoE)

    During transform method method maps intervals/categories to their WoE calculated
    during fitting.

    Can take andvantage of provided filtering algorithm (with required API)
    to combine groups.

    Some specifics.
    Numeric (continuous variables):
        Takes advantage of pd.qcut function.
        Adds additional category for missing values if encountered any.

    Categorical (any not numeric):
        If variable dtype is not numeric, considers it as categorical variable and
        puts each unique value in separate group (which can be combined later by
        filtering algo)

    Unknown values (that weren't met during fit) can be set to none or the mean WOE value of groups of this variable

    Notes:
    For hand changed WOE tables, intervals should go first, then specials/categories.

    Parameters:
    num_intervals: int
        n bins for continuous variable (pd.qcut(q))
    max_groups: int
        max groups to keep. ONLY WORKS WHEN filter_groups_algo provided
    min_unique_cut: int
        if variable has less than this unique values, consider it categorical
    categorical_cols: list ("in" clause)
        variables to consider as categories unconditionally
    cols_to_transform: list, default = None (all columns)
        variables that will be affected by transofremer (others will be passed unchanged)
    open_edge_bins: bool, defualt = True
        If edge bins wil be set to (+- inf). Ex. values lower than min value during fit
        stage will be set to the lowest group. If param = False, this vals will be set to unknown
    missing_vals: str, default = 'separate'
        How to handle missing values. As separate category or as worst category
            'separate' - special category
            'ignore_to_worst' - empty values will have same woe as worst group
    unknown_vals: str, default = 'nan'
        Method for transformation unknown values (that weren't met during fit stage)
            'woe_mean': mean WOE of this variable
            'nan': sets to np.nan
    specials: dict, {variable_name: list}
        values of continuous variables that will be put as separate category
        but may be combined by filter_groups_algo (depends on algorithm functionality)
    filter_groups_algo: class instance (default=None)
        Algorithm for reducing amount of bins/groups in variable
        Must have .reduce_groups(freq_table, max_groups=self.max_groups) method with max_groups param.
    custom_var_params: dict of dicts,
            {variable_name: {param_name: value,
                            ...},
            ...}
        During transformation of each variable, specific to this variable params could be used.
        If param don't specified here, default value will be used.
        Variable specific params are:
            'num_intervals', 'max_groups', 'min_unique_cut',
            'open_edge_bins', 'filter_groups_algo', 'missing_vals', 'unknown_vals'
    """

    def __init__(
        self,
        num_intervals=7,
        max_groups=5,
        min_unique_cut=3,
        categorical_cols=[],
        cols_to_transform=None,
        open_edge_bins=True,
        missing_vals='separate',  # 'ignore_to_worst'
        unknown_vals='nan',  # 'woe_mean'
        specials={},
        filter_groups_algo=None,
        custom_var_params={},
    ):
        setattr(self, 'num_intervals', num_intervals)
        setattr(self, 'max_groups', max_groups)
        setattr(self, 'min_unique_cut', min_unique_cut)
        setattr(self, 'categorical_cols', categorical_cols)
        setattr(self, 'cols_to_transform', cols_to_transform)
        setattr(self, 'open_edge_bins', open_edge_bins)
        setattr(self, 'missing_vals', missing_vals)
        setattr(self, 'unknown_vals', unknown_vals)
        setattr(self, 'specials', specials)
        setattr(self, 'filter_groups_algo', filter_groups_algo)

        # Custom params for each variable
        setattr(self, 'custom_var_params', custom_var_params)

        # Creating params that refer to a single variable transforamtion
        self.attributes_for_var_fit = [
            'num_intervals',
            'max_groups',
            'min_unique_cut',
            'open_edge_bins',
            'filter_groups_algo',
            'missing_vals',
        ]
        self.attributes_for_var_trans = ['missing_vals', 'unknown_vals']

        # default variable transformation params (could be replaced by self.custom_var_params)
        self.default_params_fit = {}
        self.default_params_trans = {}

        for attrname in self.attributes_for_var_fit:
            self.default_params_fit[attrname] = getattr(self, attrname)
        for attrname in self.attributes_for_var_trans:
            self.default_params_trans[attrname] = getattr(self, attrname)

        assert self.missing_vals in ['separate', 'ignore_to_worst']
        assert self.unknown_vals in ['nan', 'woe_mean']

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits each column separately and creates dict with
            key=columns value=pd.DataFrame that contains groups/intervals with corresponding woe values

        After fit can be accessed by class_instance.woe_tables method
        """
        # Preparing variables (TODO - consider not pandas input)
        y = pd.Series(y, name='target', index=X.index).astype(int)
        df = pd.concat([X, y], axis=1)
        self.targ_name = y.name
        self.vars_to_transform = X.columns if self.cols_to_transform is None else self.cols_to_transform
        del X, y

        self.woe_tables = {}
        self.vars_stats = {}

        #### Performing calculations separately for each column ####
        for col in self.vars_to_transform:
            self.var_params = self.__get_var_params(self.custom_var_params.get(col), self.default_params_fit)
            self.woe_tables[col], self.vars_stats[col] = self._form_woe_table_and_stats(df[[col, self.targ_name]], col)
        # stats to DF
        self.vars_stats = pd.DataFrame(self.vars_stats.values(), index=self.vars_stats.keys())

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transforms each column to WOE values.
        Uses class_instance.woe_tables that were obtained during fit method (by can be set manually)
        """
        X = X.copy()
        for col in self.vars_to_transform:
            var_params = self.__get_var_params(self.custom_var_params.get(col), self.default_params_trans)

            X[col] = self._map_column_to_woe(X[col], self.woe_tables[col]['woe'], var_params)
        return X.astype(float)

    def _form_woe_table_and_stats(self, df, col_name):
        var_stats = {}
        # make freq table (using self.num_intervals param)
        freq_table = self.__make_freq_table(df, col_name)
        # calc WOE, and also IV
        freq_table['woe'], var_stats['IV'] = self._calc_freq_table_woe(freq_table)

        # filter groups (using max_groups param)
        if self.var_params['filter_groups_algo'] is not None:
            freq_table = self.var_params['filter_groups_algo'].reduce_groups(
                freq_table, max_groups=self.var_params['max_groups']
            )

        return freq_table, var_stats

    def _calc_freq_table_woe(self, table):
        freq_table = table.copy()

        # correcting special cases (zeros in groups)
        freq_table.loc[freq_table[1] == 0, 1] = 0.5
        freq_table.loc[freq_table[0] == 0, 0] = 0.5  # freq_table.loc[freq_table[0] == 0, 1] -

        # sizes of groups
        size_0, size_1 = freq_table[[0, 1]].sum()

        freq_table['woe'] = freq_table.apply(lambda x: np.log((x[0] / size_0) / (x[1] / size_1)), axis=1)
        iv = freq_table.apply(lambda x: (x[0] / size_0 - x[1] / size_1) * x['woe'], axis=1).sum()

        return freq_table['woe'], iv

    def __make_freq_table(self, df, col_name):
        # binning and grouping
        df[col_name] = self._find_bins(df[col_name])
        freq_table = df.groupby([col_name, self.targ_name]).size().unstack(level=1, fill_value=0)

        return freq_table

    @staticmethod
    def _map_column_to_woe(variable, woe_table, var_params):
        # Create dummy series for future filling
        X_new = pd.Series(np.full(len(variable), 'unknown_v'), index=variable.index, name=variable.name)

        # score missing values as min WOE
        if var_params['missing_vals'] == 'ignore_to_worst':
            woe_table = woe_table.copy()
            woe_table.loc['missing'] = woe_table.min()

        # Main mapping from WOE table to new data
        for group, wi_val in zip(woe_table.index, woe_table):
            if (not isinstance(group, Iterable)) or isinstance(group, str):
                group = [group]  # if group has single value/interval
            for subgroup in group:
                if isinstance(subgroup, pd._libs.interval.Interval):
                    # Filling values in the interval and writing min/max intervals
                    X_new.loc[(subgroup.left < variable) & (variable <= subgroup.right)] = wi_val
                elif subgroup == 'missing':
                    X_new.loc[variable.isna()] = wi_val
                else:
                    X_new.loc[variable == subgroup] = wi_val

        # check unknowns
        p_unknows = (X_new == 'unknown_v').sum() / len(X_new)
        if p_unknows > 0.3:
            print(f"Variable {X_new.name} contains too many unknown values {round(p_unknows, 4)}")

        # Replacing unknown values with chosen strategy
        if var_params['unknown_vals'] == 'nan':
            X_new.loc[X_new == 'unknown_v'] = np.nan
        elif var_params['unknown_vals'] == 'woe_mean':
            X_new.loc[X_new == 'unknown_v'] = woe_table.mean()

        return X_new

    def _find_bins(self, variable):
        """
        Returns transformed variable, where each value is the group of the original value
        """
        n_uniques = variable.nunique()
        specials = self.specials.get(variable.name)

        # if not numeric or nuniques is small or in category list return as is
        if (
            (not pd.api.types.is_numeric_dtype(variable))
            or (n_uniques < self.var_params['min_unique_cut'])
            or variable.name in self.categorical_cols
        ):
            variable = variable.astype('category')
            if variable.isna().any():
                variable = variable.cat.add_categories('missing').fillna('missing')
            return variable

        # for simple pd.cut remove and replace specials with nans
        specials = [] if specials is None else specials
        var_clean = variable.copy()
        var_clean.loc[var_clean.isin(specials)] = np.nan

        # PD CUT (replace values with intervals)
        var_bin = pd.qcut(var_clean, q=self.var_params['num_intervals'], precision=0, duplicates='drop')

        # setting most left/right edges to +- inf
        if self.var_params['open_edge_bins']:
            cats_new = list(var_bin.cat.categories)
            cats_new[0] = pd.Interval(-np.inf, cats_new[0].right)
            cats_new[-1] = pd.Interval(cats_new[-1].left, np.inf)
            var_bin = var_bin.cat.set_categories(cats_new, rename=True, ordered=True)

        # returning special values to it's place
        for special_ in specials:
            var_bin = var_bin.cat.add_categories(special_)
            var_bin.loc[variable == special_] = special_

        if self.var_params['missing_vals'] == 'separate':
            var_bin = var_bin.cat.add_categories('missing').fillna('missing')

        var_bin = var_bin.cat.remove_unused_categories()
        return var_bin

    def __get_var_params(self, custom_params, default_params):
        """Taking default or custom params for variable transformation"""
        var_params = default_params.copy()
        if custom_params is not None:
            for key, val in custom_params.items():
                var_params[key] = val

        self._assert_params(var_params)

        return var_params

    def _assert_params(self, var_params):
        # TODO, add asserts
        assert var_params.get('missing_vals', 'separate') in ['separate', 'ignore_to_worst']
        assert var_params.get('unknown_vals', 'nan') in ['nan', 'woe_mean']


class WoE_groups_filter:
    """
    Filter for WOE tables
    This is subproject of WoE_Transformer class

    Takes pd.Dataframe with WOE table and recursively combines groups
    Recommended method - WOE_DISTANCE, that combines groups with close WOE values

    Intervals can be connected only with neighboors. Categories can be combined in any fashion.
    """

    def __init__(self, method='WOE_DISTANCE', size_alpha=0.5):
        """Class for filtering existing WOE tables

        Args:
            method (str, optional): Method to combine groups. Defaults to 'WOE_DISTANCE'.
            size_alpha (float, optional): Punishing small size groups. Higher value - more tendency
            towards combining small groups (min value 0). Defaults to 0.5.
        """

        assert method in ['IV', 'WOE_DISTANCE']
        self.method = method
        self.size_alpha = size_alpha

    def reduce_groups(self, df, max_groups):
        """
        df: freq table with index=group, columns 0 and 1 has number of examples with target 0 or 1)
        """
        if len(df) <= max_groups:
            return df

        df = df.copy()

        # reset index for simpler location
        index_name = df.index.name if df.index.name is not None else 'index'
        df = df.reset_index().rename(columns={index_name: 'group'})

        # switch to object (from category)
        df['group'] = df['group'].astype('object')

        # Reducing N groups
        while len(df) > max_groups and len(df) > 2:
            # split into intervals and categorical groups
            df_int, df_cat = self.divide_and_calc_metr(df)
            df = self.combine_worst_groups(df_int, df_cat)

        # recalculate woe for the result
        df['woe'] = self._recalculate_woe(df)

        # lists to frozensets for hashability (indexing)
        df['group'] = df['group'].map(
            lambda x: frozenset(x) if (isinstance(x, Iterable) and not isinstance(x, str)) else x
        )
        # returning to categorical index
        df['group'] = df['group'].astype('category')
        df = df.set_index('group').drop(columns=['metric'])
        return df

    def divide_and_calc_metr(self, df):
        """Separate categories from intervals, and then calculate metric for group.
        Metric depends on method, but generally - low metric groups are combined.
        """
        df['woe'] = self._recalculate_woe(df)

        # Divide intervals and categories into different dataframes
        interval_mask = df['group'].map(lambda x: isinstance(x, pd._libs.interval.Interval)).astype('bool')
        df_int = df[interval_mask]
        df_cat = df[~interval_mask]

        if self.method == 'IV':
            df_int, df_cat = self._calc_iv_metric(df_int, df_cat)

        if self.method == 'WOE_DISTANCE':
            df_int, df_cat = self._calc_woe_dist_metric(df_int, df_cat)

        # if min_group_size is not None:
        #    self.combine_min_sized_groups(df_int, d_cat)

        # for categories just sort by metric
        df_cat = df_cat.sort_values('metric', ascending=True)

        # For simpler category combinations put in lists
        df_cat['group'] = df_cat['group'].map(
            lambda x: list(x) if (isinstance(x, Iterable) and not isinstance(x, str)) else [x]
        )

        df_cat = df_cat.reset_index(drop=True)
        df_int = df_int.reset_index(drop=True)

        return df_int, df_cat

    def _calc_iv_metric(self, df_int, df_cat):
        df_int['metric'] = self._calc_group_iv(df_int, self.size_alpha)
        df_cat['metric'] = self._calc_group_iv(df_cat, self.size_alpha)

        # for intervals rolling of 3 close intervals
        df_int['metric'] = df_int['metric'].rolling(3, center=True, min_periods=2).mean()

        return df_int, df_cat

    @staticmethod
    def _calc_group_iv(df, size_alpha):
        if len(df) > 1:
            freq_table = df.copy()
            # correcting special cases (zeros in groups)
            freq_table.loc[freq_table[1] == 0, 1] = 0.5
            freq_table.loc[freq_table[0] == 0, 0] = freq_table.loc[freq_table[0] == 0, 1] - 0.5

            size_0, size_1 = freq_table[[0, 1]].sum()

            # calc IV
            metric = freq_table.apply(lambda x: (x[0] / size_0 - x[1] / size_1) * x['woe'], axis=1)
            # weight on group size (small groups have lower metric)
            metric = metric * ((freq_table[0] + freq_table[1]) / (size_0 + size_1)) ** size_alpha
        else:
            metric = None

        return metric

    def _calc_woe_dist_metric(self, df_int, df_cat):
        # For intervals calc abs woe diff beetween nighboors
        woe_down = abs(df_int['woe'] - df_int['woe'].shift(-1))
        woe_up = abs(df_int['woe'] - df_int['woe'].shift())
        df_int['metric'] = pd.concat([woe_down, woe_up], axis=1).min(axis=1)

        # for categories just sort by WOE and calc metric
        df_cat = df_cat.sort_values('woe', ascending=True)
        df_cat['metric'] = abs(df_cat['woe'] - df_cat['woe'].shift()).fillna(method='bfill')

        size_0, size_1 = df_int[[0, 1]].sum()
        df_int['metric'] = df_int['metric'] * ((df_int[0] + df_int[1]) / (size_0 + size_1)) ** self.size_alpha
        size_0, size_1 = df_cat[[0, 1]].sum()
        df_cat['metric'] = df_cat['metric'] * ((df_cat[0] + df_cat[1]) / (size_0 + size_1)) ** self.size_alpha

        return df_int, df_cat

    @staticmethod
    def _recalculate_woe(df):
        freq_table = df.copy()
        # correcting special cases (zeros in groups)
        freq_table.loc[freq_table[1] == 0, 1] = 0.5
        freq_table.loc[freq_table[0] == 0, 0] = 0.5  # freq_table.loc[freq_table[0] == 0, 1] - 0.5

        # sizes of groups
        size_0, size_1 = freq_table[[0, 1]].sum()

        woe = freq_table.apply(lambda x: np.log((x[0] / size_0) / (x[1] / size_1)), axis=1)

        return woe

    def combine_worst_groups(self, df_int, df_cat):
        """Find wheather we have worst group in in intervals or categories.
        Then perform proper combinations"""
        if len(df_int) > 1 and len(df_cat) > 1:
            min_metr_int = df_int['metric'].min()
            min_metr_cat = df_cat['metric'].min()
            if min_metr_cat > min_metr_int:
                df_int = self.combine_worst_intervals(df_int)
            else:
                df_cat = self.combine_worst_cat(df_cat)

        elif len(df_int) > 1:
            df_int = self.combine_worst_intervals(df_int)
        elif len(df_cat) > 1:
            df_cat = self.combine_worst_cat(df_cat)

        df = df_int.append(df_cat)
        return df

    @staticmethod
    def combine_worst_cat(df):
        # combine worst groups, so first two rows (since ordered by metric)
        df.loc[1, [0, 1]] = df.loc[0, [0, 1]] + df.loc[1, [0, 1]]

        # combine categories into list
        new_groups = df['group'].to_list()
        new_group = new_groups[0]
        new_group.extend(new_groups[1])
        new_groups = [new_group] + new_groups[2:]

        # drop worst group (first row)
        df = df.drop(index=df.iloc[0:1].index)
        df['group'] = new_groups

        return df

    @staticmethod
    def combine_worst_intervals(df):
        # find index (interval) with lowest metric
        worst_row_idx = df['metric'].idxmin()

        # edge cases
        if worst_row_idx == 0:
            worst_neighb_idx = 1
        elif worst_row_idx == len(df) - 1:
            worst_neighb_idx = len(df) - 2
        else:
            # look for lowest metric neighboor
            worst_neighb_idx = df.loc[[worst_row_idx - 1, worst_row_idx + 1], 'metric'].idxmin()

        # combine intervals
        if worst_row_idx > worst_neighb_idx:
            new_interval = pd.Interval(df.loc[worst_neighb_idx, 'group'].left, df.loc[worst_row_idx, 'group'].right)
        else:
            new_interval = pd.Interval(df.loc[worst_row_idx, 'group'].left, df.loc[worst_neighb_idx, 'group'].right)

        # replace value of worst neighboor with sum of it with worst value and drop worst
        df.loc[worst_neighb_idx, [0, 1]] = df.loc[worst_neighb_idx, [0, 1]] + df.loc[worst_row_idx, [0, 1]]
        df = df.drop(index=worst_row_idx)

        df.loc[worst_neighb_idx, 'group'] = new_interval

        return df