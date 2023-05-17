import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Open the JSON file
with open('price_data.json') as f:
    # Load the data
    price_data = json.load(f)

# Prices
df = pd.DataFrame(price_data)

# Calculate log returns
df['log_returns'] = np.log(df['rate_close'] / df['rate_close'].shift(1))

# Drop the NaN values that were created by the shift operation
df = df.dropna()

# Histogram
plt.hist(df['log_returns'], bins=30, alpha=0.5, color='g', edgecolor='black')
plt.title('Histogram of Log Returns')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Q-Q Plot
stats.probplot(df['log_returns'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Shapiro-Wilk Normality test
shapiro_test = stats.shapiro(df['log_returns'])
print(shapiro_test)

################################################################################
# Test to determine if data follows a t-student distribution (distribution with fatter tails)
################################################################################

# Fit t-distribution to data
params = stats.t.fit(df['log_returns'])

# Generate t-distribution values
t_dist = stats.t.rvs(*params, size=len(df['log_returns']))

# Normalize both data sets for better comparison
log_returns_norm = (df['log_returns'] - np.mean(df['log_returns'])) / np.std(df['log_returns'])
t_dist_norm = (t_dist - np.mean(t_dist)) / np.std(t_dist)

# Sort the values
log_returns_sorted = np.sort(log_returns_norm)
t_dist_sorted = np.sort(t_dist_norm)

# Perform Kolmogorov-Smirnov test
D, p_value = stats.ks_2samp(log_returns_sorted, t_dist_sorted)

print('D statistic:', D)
print('p-value:', p_value)
