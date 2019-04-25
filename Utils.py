import numpy as np
import pandas as pd


def clean_data(data, feat_info):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data

    INPUT: Demographics DataFrame, Feature info DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """

    # convert missing value codes into NaNs
    ints = ['-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9'] + list('0123456789')
    for i in feat_info.index:
        missing_labels = feat_info.iloc[i]['missing_or_unknown']
        missing_labels = missing_labels.translate(str.maketrans('' ,'' ,'[]')).split(',')
        missing_labels = list(map(lambda x: int(x) if x in ints else x, missing_labels))

        varname = feat_info.iloc[i]['attribute']
        data[varname] = data[varname].map(lambda x: np.nan if x in missing_labels else x)

    # remove columns with more than 20% of values missing
    # list of vars is ['TITEL_KZ', 'AGER_TYP', 'KK_KUNDENTYP', 'KBA05_BAUMAX', 'GEBURTSJAHR', 'ALTER_HH']
    mv = data.isna().sum()
    mv_outliers = mv[mv /len(data.index) > 0.2]
    mv_outliers = mv[['TITEL_KZ', 'AGER_TYP', 'KK_KUNDENTYP', 'KBA05_BAUMAX', 'GEBURTSJAHR', 'ALTER_HH']]
    data = data.drop(columns=mv_outliers.index)

    # drop rows that are missing values for more than 25 of the original variables
    mv_count = data.isna().sum(axis=1)
    mv_idx = mv_count[mv_count > 25].index
    data = data.drop(index=mv_idx)

    # get list of categorical values to change into dummies
    cat_vars = feat_info[feat_info['type' ]=='categorical']['attribute']
    cat_vars = cat_vars[~cat_vars.isin(mv_outliers.index)] # already dropped these columns
    cat_vars = cat_vars[~cat_vars.isin
        (['FINANZTYP', 'LP_FAMILIE_FEIN', 'GFK_URLAUBERTYP', 'LP_STATUS_FEIN', 'CAMEO_DEU_2015'])] # dropping these
    # cat_vars = cat_vars[~cat_vars.isin(['GREEN_AVANTGARDE_2', 'SOHO_KZ'])] # already indicators

    # Re-encode categorical variables into dummy variables
    for var in cat_vars:
        nan_idx = pd.isnull(data[var]).nonzero()[0]
        df = pd.get_dummies(data[var], prefix=var, drop_first=True)
        df.iloc[nan_idx ,:] = np.nan
        data = data.drop(columns=var)
        data = data.join(df)

    # drop categorical variables that don't offer much additional information (to reduce dimensionality)
    data = data.drop(columns=['FINANZTYP', 'LP_FAMILIE_FEIN', 'GFK_URLAUBERTYP', 'LP_STATUS_FEIN', 'CAMEO_DEU_2015'])

    #  Create "generation" variable
    data['generation_50'] = data['PRAEGENDE_JUGENDJAHRE'].isin([3., 4.]).astype(int) # 50s
    data['generation_60'] = data['PRAEGENDE_JUGENDJAHRE'].isin([5., 6., 7.]).astype(int) # 60s
    data['generation_70'] = data['PRAEGENDE_JUGENDJAHRE'].isin([8., 9.]).astype(int) # 70s
    data['generation_80'] = data['PRAEGENDE_JUGENDJAHRE'].isin([10., 11., 12.]).astype(int) # 80s
    data['generation_90'] = data['PRAEGENDE_JUGENDJAHRE'].isin([13., 14.]).astype(int) # 90s

    # Create "mainstream or counterculture" variable
    data['avantgarde_youth'] = data['PRAEGENDE_JUGENDJAHRE'].isin([2., 4., 6., 7., 9., 11., 13., 15.]).astype \
        (int) # avantgarde

    # fix any values that are actually missing but became False during dummy creation
    nan_idx = pd.isnull(data['PRAEGENDE_JUGENDJAHRE']).nonzero()[0]
    for var in ['generation_50', 'generation_60', 'generation_70', 'generation_80', 'generation_90', 'avantgarde_youth']:
        data.iloc[nan_idx, data.columns.get_loc(var)] = np.nan

    # Create SES variable
    data['SES'] = data['CAMEO_INTL_2015'].replace(['11', '12', '13', '14', '15',
                                                   '21', '22', '23', '24', '25',
                                                   '31', '32', '33', '34', '35',
                                                   '41', '42', '43', '44', '45',
                                                   '51', '52', '53', '54', '55'],
                                                  [4, 4, 4, 4, 4,
                                                   3, 3, 3, 3, 3,
                                                   2, 2, 2, 2, 2,
                                                   1, 1, 1, 1, 1,
                                                   0, 0, 0, 0, 0])

    # Create "life stage" variable
    data['life_stage_type'] = data['CAMEO_INTL_2015'].replace(['11', '12', '13', '14', '15',
                                                               '21', '22', '23', '24', '25',
                                                               '31', '32', '33', '34', '35',
                                                               '41', '42', '43', '44', '45',
                                                               '51', '52', '53', '54', '55'],
                                                              [0, 1, 2, 3, 4,
                                                               0, 1, 2, 3, 4,
                                                               0, 1, 2, 3, 4,
                                                               0, 1, 2, 3, 4,
                                                               0, 1, 2, 3, 4])

    # drop unused columns
    data = data.drop(columns='PRAEGENDE_JUGENDJAHRE') # transformed into generation and avantgarde variables
    data = data.drop(columns='CAMEO_INTL_2015') # transformed into SES and life stage variables
    data = data.drop(columns='LP_LEBENSPHASE_FEIN') # drop LP_LEBENSPHASE_FEIN in favor of rougher variable
    data = data.drop(columns='PLZ8_BAUMAX') # drop PLZ8_BAUMAX to reduce dimensionality--seems relatively unimportant

    # turning these mixed vars into dummy variables
    # life stage, neighborhood quality or rural flag
    mixed_to_dummies = ['LP_LEBENSPHASE_GROB', 'WOHNLAGE']

    # Re-encode mixed-categorical variable(s) to be kept in the analysis.
    for var in mixed_to_dummies:
        nan_idx = pd.isnull(data[var]).nonzero()[0]
        df = pd.get_dummies(data[var], prefix=var, drop_first=True)
        df.iloc[nan_idx ,:] = np.nan
        data = data.drop(columns=var)
        data = data.join(df)

    # convert to pandas dataframe
    data = pd.DataFrame(data)

    # Return the cleaned dataframe.
    return data
