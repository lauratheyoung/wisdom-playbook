import gspread as gs
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

def split_user_data(df_user_data):
    i_cols = [col for col in df_user_data.columns if col.startswith("I ")]
    df_user_qs = df_user_data[i_cols]               # subset with "I " columns
    df_user_meta = df_user_data.drop(columns=i_cols)  # subset with all other columns
    return df_user_meta, df_user_qs

def split_peer_data(df_peer_data):
    they_cols = [col for col in df_peer_data.columns if col.startswith("They ")]
    df_peer_qs = df_peer_data[they_cols]               # subset with "I " columns
    df_peer_meta = df_peer_data.drop(columns=they_cols)  # subset with all other columns
    return df_peer_meta, df_peer_qs

def prepare_user_data(user_data_source):
    #Replace empty strings with NaNs
    df_user_data = user_data_source.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')

    #Split dfs into Questions and Metadata subsets
    df_user_meta, df_user_qs = split_user_data(df_user_data)

    #Add formatted full name col
    full_name_format_user = lambda row: str(row.iloc[1] + " " + row.iloc[6]).lower()
    df_user_meta['formatted_fullname'] =  df_user_meta.apply(full_name_format_user, axis=1)

    return pd.concat([df_user_meta, df_user_qs], axis=1)

def prepare_peer_data(peer_data_source):
    #Replace empty strings with NaNs
    df_peer_data = peer_data_source.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')

    #Split dfs into Questions and Metadata subsets
    df_peer_meta, df_peer_qs = split_peer_data(df_peer_data)

    #Add formatted full name col
    full_name_format_peer = lambda row: row.iloc[1].lower()
    df_peer_meta['formatted_fullname'] =  df_peer_meta.apply(full_name_format_peer, axis=1)

    return pd.concat([df_peer_meta, df_peer_qs], axis=1)

def get_user_row_by_uuid(user_data, uuid):
    return user_data[user_data['UUID'] == uuid].iloc[0]

def get_peer_data_from_user_row(peer_data, user_row):
    user_formatted_name = str(user_row['formatted_fullname'])
    return peer_data[peer_data['formatted_fullname'] == user_formatted_name]

def avg_peer_scores(peer_data):
    df_peer_qs = split_peer_data(peer_data)[1]
    return list(df_peer_qs.mean())

def get_user_scores_from_row(user_row):
    df_user_row = split_user_data(pd.DataFrame(user_row).T)[1]
    return list(df_user_row.iloc[0].astype(float))