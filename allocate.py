"""
Allocate predictit buys based on portfolio optimization

November 01, 2020
"""

from functools import lru_cache
import os
import time
import zipfile

import requests

import numpy as np
import pandas
from pypfopt import EfficientFrontier


memoize = lru_cache(None)

df_state_info = pandas.DataFrame([
    ('California', 'CA', 6611),
    ('Arizona', 'AZ', 5596),
    ('North Carolina', 'NC', 5599),
    ('Texas', 'TX', 5798),
    ('Michigan', 'MI', 5545),
    ('Ohio', 'OH', 5600),
    ('Pennsylvania', 'PA', 5543),
    ('Minnesota', 'MN', 5597),
    ('Utah', 'UT', 6585),
    ('Alaska', 'AK', 6591),
    ('Florida', 'FL', 5544),
    ('Wisconsin', 'WI', 5542),
    ('Georgia', 'GA', 5604),
    ('Mississippi', 'MS', 6628),
    ('Montana', 'MT', 6606),
    ('New Hampshire', 'NH', 5598),
    ('South Carolina', 'SC', 6609),
    ('New Mexico', 'NM', 6573),
    ('Virginia', 'VA', 5602),
    ('New York', 'NY', 6612),
    ('Colorado', 'CO', 5605),
    ('New Jersey', 'NJ', 6580),
    ('Illinois', 'IL', 6613),
    ('Iowa', 'IA', 5603),
    ('Oregon', 'OR', 6582),
    ('Nevada', 'NV', 5601),
    ('Hawaii', 'HI', 6631),
    ('Washington', 'WA', 6598),
    ('Delaware', 'DE', 6636),
    ('Maryland', 'MD', 6593),
    ('Massachusetts', 'MA', 6596),
    ('Rhode Island', 'RI', 6629),
    ('Connecticut', 'CT', 6587),
    ('Vermont', 'VT', 6633),
    ('District of Columbia', 'DC', 6644),
    ('Maine', 'ME', 6571),
    ('Missouri', 'MO', 6581),
    ('South Dakota', 'SD', 6638),
    ('Louisiana', 'LA', 6617),
    ('Indiana', 'IN', 6572),
    ('Tennessee', 'TN', 6586),
    ('Alabama', 'AL', 6625),
    ('Kansas', 'KS', 6627),
    ('North Dakota', 'ND', 6637),
    ('Kentucky', 'KY', 6592),
    ('Arkansas', 'AK', 6597),
    ('West Virginia', 'WV', 6615),
    ('Oklahoma', 'OK', 6616),
    ('Nebraska', 'NE', 6624),
    ('Idaho', 'ID', 6623),
    ('Wyoming', 'WY', 6632),
],
    columns=['STATE', 'STATE_INIT', 'PREDICTIT_ID'])
assert len(df_state_info) == 51, len(df_state_info)
df_state_info = df_state_info.head(25)

# Predictit charges 10% on profits
PREDICTIT_PROFIT_COMMISSION = 0.1


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


@memoize
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


@memoize
def _load_df_state_models_fivethirtyeight() -> pandas.DataFrame:
    df = pandas.read_csv(
        'https://projects.fivethirtyeight.com/2020-general-data/'
        'presidential_state_toplines_2020.csv'
    )

    df.to_csv(os.path.join(_data_dir(), 'fivethirtyeight_probabilities.csv'), index=False)

    return df


@memoize
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


@memoize
def _load_df_predictit_prices() -> pandas.DataFrame:
    df_list = []

    for idx, row in df_state_info.sort_values('STATE_INIT').iterrows():
        print('Getting prices for', row['STATE_INIT'])
        market_info = requests.get(
            f'https://www.predictit.org/api/marketdata/markets/{row["PREDICTIT_ID"]}'
        )
        if not market_info.status_code == 200:
            raise ValueError(market_info.content)

        contracts = market_info.json()['contracts']
        d_contract = [
            contract for contract in contracts if contract['name'] == 'Democratic'
        ][0]
        r_contract = [
            contract for contract in contracts if contract['name'] == 'Republican'
        ][0]
        df_list.append({
            'PREDICTIT_ID': row['PREDICTIT_ID'],
            'STATE': row['STATE'],
            'STATE_INIT': row['STATE_INIT'],
            'D_YES': d_contract['bestBuyYesCost'],
            'R_YES': r_contract['bestBuyYesCost'],
        })

        time.sleep(5)

    df_prices = pandas.DataFrame(df_list)

    df_prices.to_csv(
        os.path.join(_data_dir(), 'predictit_prices.csv')
    )

    return df_prices


def _compute_df_mu_sigma_return() -> pandas.DataFrame:
    df_probs = _load_df_state_models_fivethirtyeight()
    df_probs['DATE'] = pandas.to_datetime(df_probs['modeldate'])
    df_probs = df_probs[df_probs['DATE'].eq(df_probs['DATE'].max())][[
        'state',
        'winstate_inc',
        'winstate_chal',
    ]].rename(columns={
        'state': 'STATE',
        'winstate_inc': 'PROB_R',
        'winstate_chal': 'PROB_D',
    })

    df_prices = _load_df_predictit_prices()
    df = df_probs.merge(df_prices)

    df['D_WIN_PAYOUT'] = (1 - df['D_YES']).mul(1 - PREDICTIT_PROFIT_COMMISSION)
    df['R_WIN_PAYOUT'] = (1 - df['R_YES']).mul(1 - PREDICTIT_PROFIT_COMMISSION)

    # Expected value of one share is
    # P(win) * (1 - Price) * (1 - Commission) - (1 - P(win)) * Price
    df['EV_D_YES'] = df['PROB_D'].mul(df['D_WIN_PAYOUT']) \
        .sub((1 - df['PROB_D']).mul(df['D_YES']))
    df['EV_R_YES'] = df['PROB_R'].mul(df['R_WIN_PAYOUT']) \
        .sub((1 - df['PROB_R']).mul(df['R_YES']))

    df['MU_D_YES'] = df['EV_D_YES'].div(df['D_YES'])
    df['MU_R_YES'] = df['EV_R_YES'].div(df['R_YES'])

    df['STD_D_YES'] = (df['PROB_D'].mul((df['D_WIN_PAYOUT'].sub(df['MU_D_YES'])).pow(2))
        .add((1 - df['PROB_D']).mul((df['D_YES'].mul(-1).sub(df['MU_D_YES'])).pow(2)))).pow(1 / 2)
    df['STD_R_YES'] = (df['PROB_R'].mul((df['R_WIN_PAYOUT'].sub(df['MU_R_YES'])).pow(2))
        .add((1 - df['PROB_R']).mul((df['R_YES'].mul(-1).sub(df['MU_R_YES'])).pow(2)))).pow(1 / 2)

    df = df[['STATE', 'STATE_INIT', 'PREDICTIT_ID', 'MU_D_YES', 'STD_D_YES']]
    df['SHARPE_D'] = df['MU_D_YES'].div(df['STD_D_YES'])

    return df


if __name__ == '__main__':
    df_corr = _load_df_correlation_economist()
    df_corr = df_corr.set_index(df_corr['state'])
    df_mu_sigma = _compute_df_mu_sigma_return()
    df_corr = df_corr.loc[df_mu_sigma['STATE_INIT']][df_mu_sigma['STATE_INIT'].values]

    # Covariance = diag(S) * Corr * diag(S)
    df_cov = pandas.DataFrame(np.matmul(
        np.matmul(
            np.diag(df_mu_sigma['STD_D_YES'].values),
            df_corr.values
        ),
        np.diag(df_mu_sigma['STD_D_YES'].values)
    ), index=df_mu_sigma['STATE_INIT'].values, columns=df_mu_sigma['STATE_INIT'].values)
    df_mu_sigma.to_csv(os.path.join(_data_dir(), 'mu_sigma.csv'))
    df_cov.to_csv(os.path.join(_data_dir(), 'vcov_matrix.csv'))

    ef = EfficientFrontier(
        expected_returns=df_mu_sigma.set_index('STATE_INIT')['MU_D_YES'],
        cov_matrix=df_cov)
    raw_weights = ef.max_sharpe()
    df_weights = pandas.DataFrame(
        [(state, w) for state, w in ef.clean_weights().items()], columns=['STATE_INIT', 'WEIGHT'])
    print(df_weights.sort_values('WEIGHT', ascending=False))
    ef.portfolio_performance(verbose=True)

    df_weights.to_csv('portfolio_weights.csv', index=False)
