from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from arch import arch_model
import scipy.stats as stats

app = Flask(__name__)

@app.route('/calculate_var', methods=['POST'])
def calculate_var():
    data = request.get_json()
    prices = data['prices']
    collateralValue = data['collateralValue']
    risk_level = data['risk_level']

    df = pd.DataFrame(prices)
    df['log_returns'] = np.log(df['rate_close'] / df['rate_close'].shift(1))
    df = df.dropna()
    df['log_returns'] = df['log_returns'] * 100

    model = arch_model(df['log_returns'], vol='Garch', p=1, q=1, rescale=False)
    model_fit = model.fit()
    forecast = model_fit.forecast(start=0)
    volatility = np.sqrt(forecast.variance.iloc[-1,:])

    zScore = -1 * stats.norm.ppf(risk_level)
    daily_VaR = collateralValue * zScore * volatility
    yearly_VaR = daily_VaR * np.sqrt(365)

    params = stats.t.fit(df['log_returns'])
    t_value = -stats.t.ppf(risk_level, params[0])
    daily_VaR_t = collateralValue * t_value * volatility
    yearly_VaR_t = daily_VaR_t * np.sqrt(365)

    df['VaR_hist'] = -df['log_returns'].rolling(window=len(prices)).quantile(risk_level)
    mean_VaR_hist = df['VaR_hist'].mean()

    return jsonify({
        'Daily VaR USD': daily_VaR.values[0],
        'Yearly VaR USD': yearly_VaR.values[0],
        'Daily VaR USD (t-distribution, GARCH volatility)': daily_VaR_t.values[0],
        'Yearly VaR USD (t-distribution, GARCH volatility)': yearly_VaR_t.values[0],
        'Mean 1-day VaR USD (Historical Simulation)': mean_VaR_hist
    })

if __name__ == "__main__":
    app.run(debug=True)