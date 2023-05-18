import numpy as np
import pandas as pd
from arch import arch_model
# from scipy.stats import norm
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load price data
with open('price_data.json', 'r') as f:
    prices = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(prices)

# Risk level
risk_level = 0.01

# Collateral value
collateralValue = 10000

# Calculate log returns
df['log_returns'] = np.log(df['rate_close'] / df['rate_close'].shift(1))

# Drop missing values
df = df.dropna()

# Rescale log returns
df['log_returns'] = df['log_returns'] * 100

# Fit a GARCH(1, 1) model to the log returns
model = arch_model(df['log_returns'], vol='Garch', p=1, q=1, rescale=False)
model_fit = model.fit()

# Print the summary
print(model_fit.summary())

################################################################################
#                              RESIDUAL ANALYSIS
################################################################################

# Residuals
residuals = model_fit.resid

# Perform the Jarque-Bera normality test on the residuals
jb_test = stats.jarque_bera(residuals)
print(f'Jarque-Bera test -- statistic: {jb_test[0]}, p-value: {jb_test[1]}')

# Perform the Ljung-Box Q test for autocorrelation in the residuals
lb_test = acorr_ljungbox(residuals, lags=[10])
print(f'Ljung-Box Q test for lag 10 -- statistic: {lb_test["lb_stat"].values[0]}, p-value: {lb_test["lb_pvalue"].values[0]}')

# Perform the ARCH test for heteroskedasticity in the residuals
arch_test = het_arch(residuals)
print(f'ARCH test -- LM statistic: {arch_test[0]}, LM-Test p-value: {arch_test[1]}, F-statistic: {arch_test[2]}, F-Test p-value: {arch_test[3]}')

################################################################################
#                              FORECASTING
################################################################################

# Use the GARCH model to forecast the next day's volatility
forecast = model_fit.forecast(start=0)

# Get the standard deviation (square root of variance)
volatility = np.sqrt(forecast.variance.iloc[-1,:])

################################################################################
#                   VaR CALC USING NORMAL DISTRIBUTION
################################################################################

# Calculate z-score corresponding to risk level
zScore = -1 * stats.norm.ppf(risk_level)

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
# df['realized_vol'] = np.abs(df['log_returns'])
df['realized_vol'] = df['log_returns'].rolling(window=30).std()

# Shift the realized volatility forward by one day
df['realized_vol'] = df['realized_vol'].shift(1)

# Calculate predicted volatility
df['predicted_vol'] = np.sqrt(forecast.variance.iloc[:, 0])

# Plot the predicted and realized volatility
plt.figure(figsize=(10,5))
plt.plot(df['realized_vol'], label='Realized Volatility')
plt.plot(df['predicted_vol'], color='r', label='Predicted Volatility from GARCH(1, 1)')
plt.legend(loc='best')
plt.title('Realized Volatility vs Predicted Volatility for ETH / USD')
plt.show()

# Plot the residuals
plt.figure(figsize=(10,5))
plt.plot(residuals)
plt.title('Residuals of the GARCH(1, 1) Model')
plt.show()

# Plot the ACF of the residuals
plot_acf(residuals, lags=20)
plt.title('ACF of the GARCH(1, 1) Model Residuals')
plt.show()
