"""
Allocate predictit buys based on portfolio optimization

November 01, 2020
"""

import os
import zipfile

import pandas
import requests


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def _load_df_probabilities_economist() -> pandas.DataFrame:
    df = pandas.read_csv(
        'https://cdn.economistdatateam.com/us-2020-forecast/data/'
        'president/state_averages_and_predictions_topline.csv'
    )

    df = df[[
        'state',
        'projected_win_prob',
    ]]
    df['r_win_prob'] = 1 - df['projected_win_prob']

    df.to_csv(os.path.join(_data_dir(), 'economist_probabilities.csv'), index=False)

    return df


def _load_df_state_models_fivethirtyeight() -> pandas.DataFrame:
    df = pandas.read_csv(
        'https://projects.fivethirtyeight.com/2020-general-data/'
        'presidential_state_toplines_2020.csv'
    )

    df.to_csv(os.path.join(_data_dir(), 'fivethirtyeight_probabilities.csv'), index=False)

    return df


def _load_df_correlation_economist() -> pandas.DataFrame:
    r = requests.get(
        'https://cdn.economistdatateam.com/us-2020-forecast/'
        'data/president/economist_model_output.zip'
    )

    with open(os.path.join(_data_dir(), 'economist.zip'), 'wb') as f:
        f.write(r.content)

    with zipfile.ZipFile(os.path.join(_data_dir(), 'economist.zip'), 'r') as f:
        f.extractall(_data_dir())

    df = pandas.read_csv(
        os.path.join(_data_dir(), 'output', 'site_data', 'state_correlation_matrix.csv')
    )

    return df


if __name__ == '__main__':
    _load_df_state_models_fivethirtyeight()
    _load_df_probabilities_economist()
    _load_df_correlation_economist()
