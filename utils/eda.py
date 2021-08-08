import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

def make_splits_info(df_train, df_test) -> pd.DataFrame:
    df_info = pd.DataFrame([[len(df_train), df_train['target'].mean()],
                            [len(df_test), df_test['target'].mean()]],
                    columns=['size', 'mean target'], 
                    index=['train', 'test'])
    return df_info


def plot_target_bins(data: pd.DataFrame, column: str, target='target', bins=5):
    """
    Plots binned variable with target percentage line on different axis
    Works with categorical data
    
    data: pd.Dataframe
    column: column to plot
    target: column name of target variable
    bins: int,  max bins without specials and missing
    """
    df = data.copy()

    df[column] = find_column_bins(df[column], bins)
    result = df.groupby([column, 'sex']).agg(mean_target= (target, 'mean'), 
                                              bin_size= (column, 'count'))
    result = result.reset_index()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])

    sns.barplot(x=column, y='bin_size', hue='sex', data=result, ax=ax)
    sns.pointplot(x=column, 
                y='mean_target', 
                hue='sex', 
                data=result, 
                markers=["o", "x"], 
                linestyles=["-", "--"],
                color="#bb3f3f",
                ax=ax2)
    plt.title(f"Mean target on {column}")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

def find_column_bins(ser, bins, specials=None, norm_limits=None, bin_method='q'):
    """Looking for bins (can handle specials and ignores categorical cols)
    bins: if int then q for pd.qcut, or if list then bins for pd.cut
    specials: list whith special values for separate bins
    norm_limits: tuple (min, max), values out of range will be considered as specials (carefull, may create too many bins)
    """
    n_uniques = ser.nunique()
    if isinstance(bins, int):
        l_bins = bins
    else:
        l_bins = len(bins)
    # if not numeric or nuniques is small return as is 
    if (not pd.api.types.is_numeric_dtype(ser)) or (n_uniques < l_bins):
        return ser
    if specials is None:
        specials = []
    # if value should have some min, max value and others append to specials
    if norm_limits is not None:
        specials_out = ser[(ser < norm_limits[0]) | (ser > norm_limits[1])].unique()
        specials.extend(specials_out)

    if len(specials) > 10:
        print(f"WARNING, FOUND TOO MANY SPECIALS: {len(specials)}")

    # for simple pd.cut remove replace specials with nans
    ser_clean = ser.copy()
    ser_clean.loc[ser_clean.isin(specials)] = np.nan

    ser_bin = pd.Series(np.nan, index=ser.index)

    # cutting, q for int, or simple cut for specified bins
    if isinstance(bins, int) and bin_method == 'q':
        ser_bin = pd.qcut(ser_clean, bins, precision=0, duplicates='drop')
    else:
        ser_bin = pd.cut(ser_clean, bins)

    # returning special values to it's place
    for i, special in enumerate(specials):
        ser_bin = ser_bin.cat.add_categories(special)
        ser_bin.loc[ser == special] = special

    if ser.isna().any():
        ser_bin = ser_bin.cat.add_categories('missing').fillna('missing')
    return ser_bin