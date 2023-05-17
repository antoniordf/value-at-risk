import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.stats as stats

# Open the JSON file
with open('price_data.json') as f:
    # Load the data
    price_data = json.load(f)

df = pd.DataFrame(price_data)

# Histogram
plt.hist(df['rate_close'], bins=30, alpha=0.5, color='g', edgecolor='black')
plt.title('Histogram of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Q-Q Plot
stats.probplot(df['rate_close'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Shapiro-Wilk Normality test
shapiro_test = stats.shapiro(df['rate_close'])
print(shapiro_test)
