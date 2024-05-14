from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.stream import TradingStream
import alpaca_config
import pandas as pd



async def trade_status(data):
     print(data)
     

def Connect():
    client = TradingClient(alpaca_config.API_KEY, alpaca_config.SECRET_KEY, paper=True)
    return client


def placeOrder(symbol, vol, buy=True):
    client = Connect()
    side = OrderSide.BUY if buy else OrderSide.SELL
    order_details = MarketOrderRequest(
        symbol= symbol,
        qty = vol,
        side = side,
        time_in_force = TimeInForce.DAY
        )
    _ = client.submit_order(order_data= order_details)


def showPortfolio():
    client = Connect()
    assets = [asset for asset in client.get_all_positions()]
    positions = [(asset.symbol, asset.qty, asset.current_price) for asset in assets]
    df = pd.DataFrame()
    df['symbol'] = [x[0] for x in positions]
    df['volume'] = [x[1] for x in positions]
    df['value'] = [x[2] for x in positions]
    return df


def AccountBalance():
    client = Connect()
    account = client.get_account()
    return float(account.buying_power)  


def SymbolPrice(symbol):
    client = Connect()
    position = client.get_open_position(symbol)
    price = position.current_price
    return float(price)


def AccountPerformance(initial_investment):
    client = Connect()
    account = client.get_account()
    print(f'Account proffit/loss: {float(account.equity)-initial_investment:.3f} USD')
    
def SellAll():
    client = Connect()
    assets = [asset for asset in client.get_all_positions()]
    for asset in assets:
        placeOrder(asset.symbol, asset.qty, False)