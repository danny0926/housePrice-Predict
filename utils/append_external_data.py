from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
from utils.coordinateTransform import lonlat_to_97
from tqdm import tqdm

def mining_hospital(df, hospital_df, neighbors, mining_function):
    ### only used for mining "醫療機構基本資料.csv"
    
    ### 1. use 型態別
    type_map = {t: i for i, t in enumerate(hospital_df['型態別'].unique())}
    inv_type_map = {i: t for i, t in enumerate(hospital_df['型態別'].unique())}
    sum_list = [0] * len(hospital_df['型態別'].unique())
    for n in neighbors:
        hospital = hospital_df.iloc[n]
        hospital_type = hospital['型態別']
        sum_list[type_map[hospital_type]] += 1
        
    for i, s in enumerate(sum_list):
        mining_function[inv_type_map[i]].append(s)
    return mining_function


def concat_externaldata(df, file_path, new_col_name, radius=300.0, mining_name=None):
    new_df = pd.read_csv(file_path, sep=",")
    new_df[['橫坐標', '縱坐標']] = new_df[['lng', 'lat']].apply(lambda row: pd.Series(lonlat_to_97(row['lng'], row['lat'])), axis=1)
    new_df['橫坐標'] = new_df['橫坐標'].round(2)
    new_df['縱坐標'] = new_df['縱坐標'].round(2)
    
    df[new_col_name] = 0
    life_function = []
    if mining_name == 'hospital':
        mining_function = {}
        for t in new_df['型態別'].unique():
            mining_function[t] = [];

    b_coordinates = np.array(new_df[['橫坐標', '縱坐標']])
    b_tree = cKDTree(b_coordinates)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        a_coordinates = np.array([row['橫坐標'], row['縱坐標']])
        neighbors = b_tree.query_ball_point(a_coordinates, radius)
        count = len(neighbors)
        life_function.append(count)
        
        if mining_name == 'hospital':
            mining_function = mining_hospital(df, new_df, neighbors, mining_function)
    if mining_name is not None:       
        for key in mining_function:
            df[key] = mining_function[key]
    df[new_col_name] = life_function

    return df

def concat_externaldata_concentric(df, file_path, new_col_name, radius):
    new_df = pd.read_csv(file_path, sep=",")
    new_df[['橫坐標', '縱坐標']] = new_df[['lng', 'lat']].apply(lambda row: pd.Series(lonlat_to_97(row['lng'], row['lat'])), axis=1)
    new_df['橫坐標'] = new_df['橫坐標'].round(2)
    new_df['縱坐標'] = new_df['縱坐標'].round(2)
    
    b_coordinates = np.array(new_df[['橫坐標', '縱坐標']])
    b_tree = cKDTree(b_coordinates)

    for r in radius:
        df[new_col_name + '_' + str(r)] = 0
        life_function = []
        for _, row in df.iterrows():
            a_coordinates = np.array([row['橫坐標'], row['縱坐標']])
            neighbors = b_tree.query_ball_point(a_coordinates, r)
            count = len(neighbors)
            life_function.append(count)
        df[new_col_name + '_' + str(r)] = life_function

    return df