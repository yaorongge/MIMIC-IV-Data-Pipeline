import csv
import numpy as np
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

########################## GENERAL ##########################
def dataframe_from_csv(path, compression='gzip', header=0, index_col=0, chunksize=None):
    return pd.read_csv(path, compression=compression, header=header, index_col=index_col, chunksize=None)

def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'core/admissions.csv.gz'))
    admits=admits.reset_index()
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'ethnicity']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits


def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'core/patients.csv.gz'))
    pats = pats.reset_index()
    pats = pats[['subject_id', 'gender','dod','anchor_age','anchor_year', 'anchor_year_group']]
    pats['yob']= pats['anchor_year'] - pats['anchor_age']
    #pats.dob = pd.to_datetime(pats.dob)
    pats.dod = pd.to_datetime(pats.dod)
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(mimic4_path):
    diag = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/diagnoses_icd.csv.gz'))
    diag.reset_index(inplace=True)
    return diag


def read_d_icd_diagnoses_table(mimic4_path):
    d_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_diagnoses.csv.gz'))
    d_icd.reset_index(inplace=True)
    return d_icd[['icd_code', 'long_title']]


def read_diagnoses(mimic4_path):
    return read_diagnoses_icd_table(mimic4_path).merge(
        read_d_icd_diagnoses_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


def standardize_icd(mapping, df, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

    def icd_9to10(icd):
        # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except:
            print("Error on code", icd)
            return np.nan

    # Create new column with original codes as default
    col_name = 'icd10_convert'
    if root: col_name = 'root_' + col_name
    df[col_name] = df['icd_code'].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    for code, group in df.loc[df.icd_version == 9].groupby(by='icd_code'):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code


########################## LABS ##########################
def read_labevents_table(mimic4_path):
    labevents = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/labevents.csv.gz'), chunksize=1000)
    labevents.reset_index(inplace=True)
    return labevents[['subject_id', 'itemid', 'hadm_id', 'charttime', 'storetime', 'value', 'valueuom', 'flag']]


def read_d_labitems_table(mimic4_path):
    labitems = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_labitems.csv.gz'), chunksize=1000)
    labitems.reset_index(inplace=True)
    return labitems[['itemid', 'label', 'category', 'lonic_code']]


def read_labs(mimic4_path):
    labevents = read_labevents_table(mimic4_path)
    labitems =  read_d_labitems_table(mimic4_path)
    return labevents(mimic4_path).merge(
        labitems, how='inner', left_on=['itemid'], right_on=['itemid']
    )


########################## PROCEDURES ##########################
def read_procedures_icd_table(mimic4_path):
    proc = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/procedures_icd.csv.gz'))
    proc.reset_index(inplace=True)
    return proc


def read_d_icd_procedures_table(mimic4_path):
    p_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_procedures.csv.gz'))
    p_icd.reset_index(inplace=True)
    return p_icd[['icd_code', 'long_title']]


def read_procedures(mimic4_path):
    return read_procedures_icd_table(mimic4_path).merge(
        read_d_icd_procedures_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


########################## MEDICATIONS ##########################
def read_prescriptions_table(mimic4_path):
    meds = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/prescriptions.csv.gz'))
    meds = meds.reset_index()
    return meds[['subject_id', 'hadm_id', 'starttime', 'stoptime', 'ndc', 'gsn', 'drug', 'drug_type']]

def get_generic_drugs(mapping, df):
    """Takes NDC product table and prescriptions dataframe; adds column with NDC table's corresponding generic name"""

    def brand_to_generic(ndc):
        # We only want the first 2 sections of the NDC code: xxxx-xxxx-xx
        matches = list(re.finditer(r"-", ndc))
        if len(matches) > 1:
            ndc = ndc[:matches[1].start()]
        try:
            return mapping.loc[mapping.PRODUCTNDC == ndc].NONPROPRIETARYNAME.iloc[0]
        except:
            print("Error: ", ndc)
            return np.nan

    df['generic_drug_name'] = df['ndc'].apply(brand_to_generic)


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    mapping = pd.read_csv(map_path, header=0, delimiter='\t')
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def read_ndc_mapping(map_path):
    ndc_map = pd.read_csv(map_path, header=0, delimiter='\t')
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.fillna("")
    ndc_map.NONPROPRIETARYNAME = ndc_map.NONPROPRIETARYNAME.apply(str.lower)
    ndc_map.columns = list(map(str.lower, ndc_map.columns))
    return ndc_map


########################## PREPROCESSING ##########################
def get_range(df: pd.DataFrame, time_col:str, anchor_col:str, measure='days') -> pd.Series:
    """Uses array arithmetic to find the ranges an observation time could be in based on the patient's anchor info"""

    def get_year_timedelta(group: tuple) -> str:
        """Applied to a Series of tuples in the form (start year, end year) that represent the range of years a labevent could occur according to a patient's anchor info"""
        # Recall that we only want info in a 3 year time range anywhere between 2008 and 2019.
        # All possible ranges are:  ['2008 - 2010', '2009 - 2011', '2010 - 2012', '2011 - 2013',
        # '2012 - 2014', '2013 - 2015','2014 - 2016',  '2015 - 2017', '2016 - 2018',  '2017 - 2019']
        if group[0] >= 2008 and group[0] <= 2017 and group[1] >= 2010 and group[1] <= 2019:
            return group[0] - 2008
        else:
            return np.nan

    if measure == 'years':
        # Array arithmetic to obtain a tuple of (start year, end year) for each labevent
        # After getting list of tuples, apply get_timedelta
        shift = (df[time_col] - df[anchor_col])
        return pd.Series(list(zip(df.min_year_group + shift, df.max_year_group + shift))).apply(get_year_timedelta)
    elif measure =='days':
        # Convert base_year into a datetime and find the difference in days between the time_col and the beginning of the base_year
        base_dt = df[anchor_col].apply(lambda x: dt.datetime(year=x, month=1, day=1))
        return (df[time_col] - base_dt).dt.days
    else:
        raise Exception('\'measure\' argument must be either \'years\' or \'days\'.')


def timestamp_cohort_data(dataset_path: str, cohort_path:str, time_col:str, anchor_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""

    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()

        # Only consider values in our cohort
        cohort = pd.read_pickle(cohort_path, compression='gzip')

        # Get outer ranges of a subject's anchor_year_group, then extract year from each lab's charttime
        cohort['min_year_group'] = cohort.anchor_year_group.str.slice(start=0, stop=4).astype(int)
        cohort['max_year_group'] = cohort.anchor_year_group.str.slice(start=-4).astype(int)
        module['admit_year'] = module[time_col].dt.year

        # merge module and cohort
        return module.merge(cohort[['subject_id', 'label', anchor_col, 'max_year_group', 'min_year_group', 'anchor_year']], how='inner', left_on='subject_id', right_on='subject_id')

    df_cohort = merge_module_cohort()
    df_cohort['timedelta_days'] = get_range(df_cohort, time_col, anchor_col, measure='days')
    df_cohort['timedelta_years'] = get_range(df_cohort, 'admit_year', 'anchor_year', measure='years')

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort.loc[(df_cohort.timedelta_years <= 6) & (~df_cohort.timedelta_years.isna())].sort_values(by='subject_id')


def preproc_icd_module(module_path:str, adm_cohort_path:str, icd_map_path=None, map_code_colname=None, only_icd10=True) -> pd.DataFrame:
    """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path"""    
    
    def get_module_cohort(module_path:str, cohort_path:str):
        module = pd.read_csv(module_path, compression='gzip', header=0)
        adm_cohort = pd.read_csv(adm_cohort_path, compression='gzip', header=0)
        #print(module.head())
        #print(adm_cohort.head())
        
        #adm_cohort = adm_cohort.loc[(adm_cohort.timedelta_years <= 6) & (~adm_cohort.timedelta_years.isna())]
        return module.merge(adm_cohort[['hadm_id', 'label']], how='inner', left_on='hadm_id', right_on='hadm_id')

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a modulenosis dataframe; adds column with converted ICD10 column"""
        
        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
            except:
                #print("Error on code", icd)
                return np.nan

        # Create new column with original codes as default
        col_name = 'icd10_convert'
        if root: col_name = 'root_' + col_name
        df[col_name] = df['icd_code'].values

        # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
        for code, group in tqdm(df.loc[df.icd_version == 9].groupby(by='icd_code')):
            new_code = icd_9to10(code)
            for idx in group.index.values:
                # Modify values of original df at the indexes in the groups
                df.at[idx, col_name] = new_code

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df['root'] = df[col_name].apply(lambda x: x[:3] if type(x) is str else np.nan)

    module = get_module_cohort(module_path, adm_cohort_path)
    #print(module.shape)
    #print(module['icd_code'].nunique())

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        #print(icd_map)
        standardize_icd(icd_map, module, root=True)
        print("# unique ICD-9 codes",module[module['icd_version']==9]['icd_code'].nunique())
        print("# unique ICD-10 codes",module[module['icd_version']==10]['icd_code'].nunique())
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10)",module['root_icd10_convert'].nunique())
        print("# unique ICD-10 codes (After clinical gruping ICD-10 codes)",module['root'].nunique())
    return module


def pivot_cohort(df: pd.DataFrame, prefix: str, target_col:str, values='values', use_mlb=False, ohe=True, max_features=None):
    """Pivots long_format data into a multiindex array:
                                            || feature 1 || ... || feature n ||
        || subject_id || label || timedelta ||
    """
    aggfunc = np.mean
    pivot_df = df.dropna(subset=[target_col])

    if use_mlb:
        mlb = MultiLabelBinarizer()
        output = mlb.fit_transform(pivot_df[target_col].apply(ast.literal_eval))
        output = pd.DataFrame(output, columns=mlb.classes_)
        if max_features:
            top_features = output.sum().sort_values(ascending=False).index[:max_features]
            output = output[top_features]
        pivot_df = pd.concat([pivot_df[['subject_id', 'label', 'timedelta']].reset_index(drop=True), output], axis=1)
        pivot_df = pd.pivot_table(pivot_df, index=['subject_id', 'label', 'timedelta'], values=pivot_df.columns[3:], aggfunc=np.max)
    else:
        if max_features:
            top_features = pd.Series(pivot_df[['subject_id', target_col]].drop_duplicates()[target_col].value_counts().index[:max_features], name=target_col)
            pivot_df = pivot_df.merge(top_features, how='inner', left_on=target_col, right_on=target_col)
        if ohe:
            pivot_df = pd.concat([pivot_df.reset_index(drop=True), pd.Series(np.ones(pivot_df.shape[0], dtype=int), name='values')], axis=1)
            aggfunc = np.max
        pivot_df = pivot_df.pivot_table(index=['subject_id', 'label', 'timedelta'], columns=target_col, values=values, aggfunc=aggfunc)

    pivot_df.columns = [prefix + str(i) for i in pivot_df.columns]
    return pivot_df