import numpy as np
import pandas as pd
from arch import arch_model
# from scipy.stats import norm
import scipy.stats as stats
import json
import matplotlib.pyplot as plt

# Load price data
with open('price_data.json', 'r') as f:
    prices = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(prices)

# Calculate log returns
df['log_returns'] = np.log(df['rate_close'] / df['rate_close'].shift(1))

# Drop missing values
df = df.dropna()

# Rescale log returns
df['log_returns'] = df['log_returns'] * 100

# Fit a GARCH(1, 1) model to the log returns
model = arch_model(df['log_returns'], vol='Garch', p=1, q=1, rescale=False)
model_fit = model.fit()

# Calculating Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)
print("AIC: ", model_fit.aic)
print("BIC: ", model_fit.bic)

# Use the GARCH model to forecast the next day's volatility
forecast = model_fit.forecast(start=0)

# Get the standard deviation (square root of variance)
volatility = np.sqrt(forecast.variance.iloc[-1,:])

# Risk level
risk_level = 0.01

# Calculate z-score corresponding to risk level
zScore = -1 * stats.norm.ppf(risk_level)

# Collateral value
collateralValue = 10000

# VaR
daily_VaR = collateralValue * zScore * volatility
yearly_VaR = daily_VaR * np.sqrt(365)

print("Daily VaR USD", daily_VaR.values[0])
print("Yearly VaR USD", yearly_VaR.values[0])

################################################################################
#                     VaR CALC USING t DISTRIBUTION
################################################################################

# Fit t-distribution to data
params = stats.t.fit(df['log_returns'])

# Calculate t-value corresponding to risk level
t_value = -stats.t.ppf(risk_level, params[0])

# VaR under t-distribution
daily_VaR_t = collateralValue * t_value * volatility
yearly_VaR_t = daily_VaR_t * np.sqrt(365)

print("Daily VaR USD (t-distribution, GARCH volatility)", daily_VaR_t.values[0])
print("Yearly VaR USD (t-distribution, GARCH volatility)", yearly_VaR_t.values[0])

################################################################################
#                 VaR CALC USING HISTORICAL SIMULATION
################################################################################

# Calculate Historical VaR
df['VaR_hist'] = -df['log_returns'].rolling(window=30).quantile(risk_level)

# Calculate the mean VaR
mean_VaR_hist = df['VaR_hist'].mean()

print("Mean 1-day VaR USD (Historical Simulation): ", mean_VaR_hist)

################################################################################
#                      ESPECTED SHORTFALL CALCULATION
################################################################################


# Calculate Expected Shortfall (ES) or Conditional VaR using t distribution
df['ES'] = -df['log_returns'][df['log_returns'] <= -daily_VaR_t.iloc[0]].mean()

# Calculate the mean ES
mean_ES_t = df['ES'].mean()

print("Mean 1-day ES USD (t distribution): ", mean_ES_t)

# Calculate Expected Shortfall (ES) or Conditional VaR using historical simulation
df['ES_hist'] = df['log_returns'][df['log_returns'] <= -df['VaR_hist'].iloc[0]].mean()

# Calculate the mean ES
mean_ES_hist = df['ES'].mean()

print("Mean 1-day ES USD (historical simulation): ", mean_ES_hist)

################################################################################
#                                GRAPHS
################################################################################

# Calculate the realized volatility
df['realized_vol'] = np.abs(df['log_returns'])

# Calculate predicted volatility
df['predicted_vol'] = np.sqrt(forecast.variance)

# Plot the predicted and realized volatility
plt.figure(figsize=(10,5))
plt.plot(df['realized_vol'], label='Realized Volatility')
plt.plot(df['predicted_vol'], color='r', label='Predicted Volatility from GARCH(1, 1)')
plt.legend(loc='best')
plt.title('Realized Volatility vs Predicted Volatility for ETH / USD')
plt.show()