import asyncio
import websockets
import json


async def download_vol(asset):
    try:
        # Extract the asset symbol without additional characters
        asset_symbol = asset.split('USDT')[0]

        msg = {
            "jsonrpc": "2.0",
            "id": 8387,
            "method": "public/get_historical_volatility",
            "params": {
                "currency": asset_symbol
            }
        }

        async with websockets.connect(
                'wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(json.dumps(msg))
            while websocket.open:
                response = await websocket.recv()
                response_json = json.loads(response)
                if 'result' in response_json:
                    result = response_json['result']
                    if len(result) > 0:
                        return result[-1][1]

        raise ValueError("No result found in the response")

    except (websockets.exceptions.ConnectionClosedError,
            asyncio.TimeoutError) as err:
        raise ConnectionError(f"Connection error: {err}")

    except Exception as err:
        raise ValueError(f"An error occurred: {err}")


# async def main():
#     await download_vol("ETHUSDT")

# asyncio.run(main())
