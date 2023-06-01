from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from arch import arch_model
import scipy.stats as stats
import os
import requests
from datetime import datetime
import time

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "The API is working!"


@app.route('/calculate_var', methods=['POST'])
def calculate_var():
    try:
        data = request.get_json()
        asset = data['asset']
        end_date = data['end_date']
        collateralValue = data['collateralValue']
        risk_level = data['risk_level']

        # Convert end_date to Unix timestamp (milliseconds)
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        end_date_unix = int(time.mktime(end_date.timetuple()) *
                            1000)  # Convert to milliseconds

        # Fetch prices on Binance API
        url = f"https://api.binance.com/api/v3/klines?symbol={asset}&interval=1d&endTime={end_date_unix}&limit=1000"
        response = requests.get(url)
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data,
                          columns=[
                              'Open time', 'Open', 'High', 'Low', 'Close',
                              'Volume', 'Close time', 'Quote asset volume',
                              'Number of trades',
                              'Taker buy base asset volume',
                              'Taker buy quote asset volume', 'Ignore'
                          ])

        # Extract the closing prices from the DataFrame
        prices = df['Close'].astype(float)  # Ensure the prices are float type

        # Calculate log returns using the prices array
        df['log_returns'] = np.log(prices / prices.shift(1))

        # Drop missing values
        df = df.dropna()

        # Rescale log returns
        df['log_returns'] = df['log_returns'] * 100

        # Fit a GARCH(1, 1) model to the log returns
        model = arch_model(df['log_returns'],
                           vol='Garch',
                           p=1,
                           q=1,
                           rescale=False)
        model_fit = model.fit()

        # Forecasting
        forecast = model_fit.forecast(start=0)
        volatility = np.sqrt(forecast.variance.iloc[-1, :])

        ############################################################################
        #                   VaR CALC USING NORMAL DISTRIBUTION
        ############################################################################

        zScore = -1 * stats.norm.ppf(risk_level)
        daily_VaR = collateralValue * zScore * volatility
        yearly_VaR = daily_VaR * np.sqrt(365)

        ############################################################################
        #                     VaR CALC USING t DISTRIBUTION
        ############################################################################

        params = stats.t.fit(df['log_returns'])
        t_value = -stats.t.ppf(risk_level, params[0])
        daily_VaR_t = collateralValue * t_value * volatility
        yearly_VaR_t = daily_VaR_t * np.sqrt(365)

        df['VaR_hist'] = -df['log_returns'].rolling(
            window=len(prices)).quantile(risk_level)
        mean_VaR_hist = df['VaR_hist'].mean()

        ############################################################################
        #                            RETURN RESULTS
        ############################################################################

        return jsonify({
            'Daily VaR USD':
            daily_VaR.values[0],
            'Yearly VaR USD':
            yearly_VaR.values[0],
            'Daily VaR USD (t-distribution, GARCH volatility)':
            daily_VaR_t.values[0],
            'Yearly VaR USD (t-distribution, GARCH volatility)':
            yearly_VaR_t.values[0],
            'Mean 1-day VaR USD (Historical Simulation)':
            mean_VaR_hist
        })

    except requests.exceptions.HTTPError as errh:
        return jsonify({'error': 'HTTP Error: {}'.format(errh)}), 400
    except requests.exceptions.ConnectionError as errc:
        return jsonify({'error': 'Error Connecting: {}'.format(errc)}), 400
    except requests.exceptions.Timeout as errt:
        return jsonify({'error': 'Timeout Error: {}'.format(errt)}), 400
    except requests.exceptions.RequestException as err:
        return jsonify({'error': 'Something went wrong: {}'.format(err)}), 400
    except Exception as e:
        return jsonify({'error':
                        'An unexpected error occurred: {}'.format(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
