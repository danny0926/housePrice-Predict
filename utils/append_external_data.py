from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
from utils.coordinateTransform import lonlat_to_97


def concat_externaldata(df, file_path, new_col_name, radius=300.0):
    new_df = pd.read_csv(file_path, sep=",")
    new_df[['橫坐標', '縱坐標']] = new_df[['lng', 'lat']].apply(lambda row: pd.Series(lonlat_to_97(row['lng'], row['lat'])), axis=1)
    new_df['橫坐標'] = new_df['橫坐標'].round(2)
    new_df['縱坐標'] = new_df['縱坐標'].round(2)
    
    df[new_col_name] = 0
    life_function = []

    b_coordinates = np.array(new_df[['橫坐標', '縱坐標']])
    b_tree = cKDTree(b_coordinates)

    for _, row in df.iterrows():
        a_coordinates = np.array([row['橫坐標'], row['縱坐標']])
        neighbors = b_tree.query_ball_point(a_coordinates, radius)
        count = len(neighbors)
        life_function.append(count)
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