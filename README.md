# predictit-538
Optimal allocation of predictit choices given 538/Economist models

Data from FiveThirtyEight and The Economist election models

## About
*For entertainment purposes only*

This project uses the FiveThirtyEight win probabilities for each state and The Economist state correlation matrix to 
identify the optimal portfolio of Predictit market buys based on the portfolio with the maximum Sharpe ratio (Expected return / standard deviation).

Given a vector of expected returns by state, and the variance-covariance matrix of state returns, the [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) package then estimates the 
portfolio weights for the portfolio with the optimal Sharpe ratio.

To estimate the optimal portfolio at a given point in time based on the latest 538 and Economist models, as well as the latest prices in the Predictit markets, 
run `allocate.py`
