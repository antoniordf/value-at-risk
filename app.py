from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import pandas as pd
from arch import arch_model
import scipy.stats as stats
import requests
from datetime import datetime
import time
import implied_vol

app = FastAPI()


class CalculateVarInput(BaseModel):
    asset: str
    end_date: str
    collateralValue: float
    risk_level: float


@app.get("/")
def home():
    return {"message": "The API is working!"}


@app.post("/calculate_var")
async def calculate_var(data: CalculateVarInput):
    try:
        asset = data.asset
        end_date = data.end_date
        collateralValue = data.collateralValue
        risk_level = data.risk_level

        # Fetch implied vol data from Deribit
        deribit_vol = await implied_vol.download_vol(data.asset)

        # Convert end_date to Unix timestamp (milliseconds)
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        end_date_unix = int(time.mktime(end_date.timetuple()) * 1000)

        # Fetch prices on Binance API
        url = f"https://api.binance.com/api/v3/klines?symbol={asset}&interval=1d&endTime={end_date_unix}&limit=1000"
        response = requests.get(url)
        binance_data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(binance_data,
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
        daily_VaR_GARCH = collateralValue * zScore * volatility
        yearly_VaR_GARCH = daily_VaR_GARCH * np.sqrt(365)
        daily_VaR_implied = collateralValue * zScore * (
            (deribit_vol / 100) / np.sqrt(365))
        yearly_VaR_implied = collateralValue * zScore * (deribit_vol / 100)

        ############################################################################
        #                     VaR CALC USING t DISTRIBUTION
        ############################################################################

        params = stats.t.fit(df['log_returns'])
        t_value = -stats.t.ppf(risk_level, params[0])
        daily_VaR_t_GARCH = collateralValue * t_value * volatility
        yearly_VaR_t_GARCH = daily_VaR_t_GARCH * np.sqrt(365)
        daily_VaR_t_implied = collateralValue * t_value * (
            (deribit_vol / 100) / np.sqrt(365))
        yearly_VaR_t_implied = collateralValue * t_value * (deribit_vol / 100)

        df['VaR_hist'] = -df['log_returns'].rolling(
            window=120).quantile(risk_level)
        mean_VaR_hist = df['VaR_hist'].mean()

        ############################################################################
        #                            RETURN RESULTS
        ############################################################################

        return {
            'Daily VaR GARCH USD':
            daily_VaR_GARCH.values[0],
            'Yearly VaR GARCH USD':
            yearly_VaR_GARCH.values[0],
            'Daily VaR Implied Vol USD':
            daily_VaR_implied,
            'Yearly VaR Implied Vol':
            yearly_VaR_implied,
            'Daily VaR USD (t-distribution, GARCH volatility)':
            daily_VaR_t_GARCH.values[0],
            'Yearly VaR USD (t-distribution, GARCH volatility)':
            yearly_VaR_t_GARCH.values[0],
            'Daily VaR USD (t-distribution, implied volatility)':
            daily_VaR_t_implied,
            'Yearly VaR USD (t-distribution, implied volatility)':
            yearly_VaR_t_implied,
            'Mean 1-day VaR USD (Historical Simulation)':
            mean_VaR_hist
        }

    except requests.exceptions.HTTPError as errh:
        raise HTTPException(status_code=400,
                            detail='HTTP Error: {}'.format(errh))
    except requests.exceptions.ConnectionError as errc:
        raise HTTPException(status_code=400,
                            detail='Error Connecting: {}'.format(errc))
    except requests.exceptions.Timeout as errt:
        raise HTTPException(status_code=400,
                            detail='Timeout Error: {}'.format(errt))
    except requests.exceptions.RequestException as err:
        raise HTTPException(status_code=400,
                            detail='Something went wrong: {}'.format(err))
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Exception: {e}")
        raise HTTPException(status_code=400,
                            detail=f'An unexpected error occurred: {e}')
