import pandas as pd

def exclude_outliers(df, categories=['CN','CN*','MCI','AD'], category_col='category_criteria_1', value_col='wm_gm_diff'):
    """ Exclude outliers category by category."""
    for cat in categories:
        q1 = df.loc[df[category_col]==cat, value_col].quantile(0.25)
        q3 = df.loc[df[category_col]==cat, value_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = df.loc[(df[category_col] == cat) &
            ((df[value_col] < lower_bound) | (df[value_col] > upper_bound))].index
        df = df.drop(outlier_indices)
        print(f"Removed {len(outlier_indices)} outliers from {cat}.")
    return df


def exclude_outliers_paired(df, categories=['CN','CN*','MCI','AD'], category_col='category_criteria_1', value_col='wm_gm_diff'):
    """ If we want to exlude outliers after data matching is done, 
    we must use this function to make sure that the matched pairs will not be broken after the outlier removal, 
    at the cost of removing more data points.
    """
    
    match_id_remove = []
    
    for i, cat in enumerate(categories):
        q1 = df.loc[df[category_col]==cat, value_col].quantile(0.25)
        q3 = df.loc[df[category_col]==cat, value_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.loc[(df[category_col] == cat) &
            ((df[value_col] < lower_bound) | (df[value_col] > upper_bound)), ]
        match_id_remove.extend(outliers['match_id'].tolist())
    
    df = df.loc[~df['match_id'].isin(match_id_remove), ]

    return df