import time
import math
import pandas as pd
from binance.client import Client
from binance.enums import *
from colorama import Fore, Style, init
import ta
from tabulate import tabulate
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import config  # Importa el archivo config.py
from tqdm import tqdm

# Inicializa colorama
init(autoreset=True)

# Configura tus claves de API desde el archivo config.py
api_key = config.api_key
api_secret = config.api_secret

# Inicializa el cliente de Binance
client = Client(api_key, api_secret)

# Parámetros del bot
initial_usd_amount = 1  # Cantidad inicial en USD
stop_loss_percentage = 0.10  # 10% de stop-loss
take_profit_percentage = 0.05  # 5% de take-profit
min_usdt_balance = 5  # Saldo mínimo en USDT después de cada transacción
trailing_stop_loss_percentage = 0.03  # 3% de trailing stop-loss

# Función para obtener datos históricos
def get_historical_data(symbol, interval, lookback):
    try:
        print(f"{Fore.BLUE}Obteniendo datos históricos...{Style.RESET_ALL}")
        frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + ' min ago UTC'))
        frame = frame.iloc[:, :6]
        frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        frame = frame.set_index('Time')
        frame.index = pd.to_datetime(frame.index, unit='ms')
        frame = frame.astype(float)
        print(f"{Fore.GREEN}Datos históricos obtenidos.{Style.RESET_ALL}")
        return frame
    except Exception as e:
        print(f"{Fore.RED}Error al obtener datos históricos: {e}{Style.RESET_ALL}")
        return None

# Función para calcular indicadores técnicos
def calculate_indicators(data):
    print(f"{Fore.BLUE}Calculando indicadores técnicos...{Style.RESET_ALL}")
    data['MA_short'] = ta.trend.sma_indicator(data['Close'], window=10)
    data['MA_long'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
    data = data.dropna()  # Eliminar filas con NaN
    print(f"{Fore.GREEN}Indicadores técnicos calculados.{Style.RESET_ALL}")
    return data

# Función para ejecutar una orden de compra
def buy_order(symbol, quantity):
    try:
        print(f"{Fore.BLUE}Ejecutando orden de compra...{Style.RESET_ALL}")
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity)
        print(f"{Fore.GREEN}Orden de compra ejecutada: {order}{Style.RESET_ALL}")
        return order
    except Exception as e:
        print(f"{Fore.RED}Error al ejecutar orden de compra: {e}{Style.RESET_ALL}")
        return None

# Función para ejecutar una orden de venta
def sell_order(symbol, quantity, buy_price=None):
    try:
        print(f"{Fore.BLUE}Ejecutando orden de venta...{Style.RESET_ALL}")
        
        # Verificar saldo disponible antes de redondear
        available_asset = check_balance(symbol.replace('USDT', ''))
        quantity = min(quantity, available_asset)
        
        # Redondear la cantidad según las especificaciones de Binance
        quantity = round_quantity(symbol, quantity)
        
        # Verificar si la cantidad redondeada es válida
        if quantity <= 0:
            print(f"{Fore.RED}La cantidad a vender es inválida después de redondear.{Style.RESET_ALL}")
            return None
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity)
        print(f"{Fore.GREEN}Orden de venta ejecutada: {order}{Style.RESET_ALL}")
        if buy_price:
            calculate_profit(buy_price, float(order['fills'][0]['price']), quantity)
        return order
    except Exception as e:
        print(f"{Fore.RED}Error al ejecutar orden de venta: {e}{Style.RESET_ALL}")
        return None

# Función para calcular el stop-loss
def calculate_stop_loss(buy_price, stop_loss_percentage):
    print(f"{Fore.BLUE}Calculando stop-loss...{Style.RESET_ALL}")
    stop_loss_price = buy_price * (1 - stop_loss_percentage)
    print(f"{Fore.GREEN}Stop-loss calculado: {stop_loss_price:.8f}{Style.RESET_ALL}")
    return stop_loss_price

# Función para calcular el take-profit
def calculate_take_profit(buy_price, take_profit_percentage):
    print(f"{Fore.BLUE}Calculando take-profit...{Style.RESET_ALL}")
    take_profit_price = buy_price * (1 + take_profit_percentage)
    print(f"{Fore.GREEN}Take-profit calculado: {take_profit_price:.8f}{Style.RESET_ALL}")
    return take_profit_price

# Función para calcular las ganancias
def calculate_profit(buy_price, sell_price, quantity):
    profit_usd = (sell_price - buy_price) * quantity
    profit_percentage = ((sell_price - buy_price) / buy_price) * 100
    print(f"{Fore.GREEN}Ganancia en USD: {profit_usd:.8f}, Ganancia en %: {profit_percentage:.2f}%{Style.RESET_ALL}")

# Función para entrenar el modelo de machine learning
def train_model(data):
    print(f"{Fore.BLUE}Entrenando modelo de machine learning...{Style.RESET_ALL}")
    # Crear etiquetas de compra/venta
    data['Signal'] = 0
    data.loc[data['Close'].shift(-1) > data['Close'], 'Signal'] = 1  # Compra
    data.loc[data['Close'].shift(-1) < data['Close'], 'Signal'] = -1  # Venta

    # Eliminar filas con NaN
    data = data.dropna()

    # Dividir datos en entrenamiento y prueba
    X = data[['MA_short', 'MA_long', 'RSI', 'MACD', 'Stochastic', 'ATR', 'ADX', 'Bollinger_High', 'Bollinger_Low']]
    y = data['Signal']

    # Escalar datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo de Random Forest con barra de progreso
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    for i in tqdm(range(100), desc="Entrenando modelo", unit="iter"):
        model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{Fore.GREEN}Modelo entrenado con precisión: {accuracy}{Style.RESET_ALL}")

    return model, scaler

# Función para verificar el saldo disponible
def check_balance(asset):
    try:
        balance = client.get_asset_balance(asset=asset)
        available_balance = float(balance['free'])
        print(f"{Fore.CYAN}Saldo disponible en {asset}: {available_balance}{Style.RESET_ALL}")
        return available_balance
    except Exception as e:
        print(f"{Fore.RED}Error al verificar el saldo: {e}{Style.RESET_ALL}")
        return 0

# Función para redondear la cantidad según las especificaciones de Binance
def round_quantity(symbol, quantity):
    info = client.get_symbol_info(symbol)
    step_size = float(next(filter(lambda x: x['filterType'] == 'LOT_SIZE', info['filters']))['stepSize'])
    precision = int(round(-math.log(step_size, 10), 0))
    # Usar floor para redondear hacia abajo
    quantity = math.floor(quantity * (10 ** precision)) / (10 ** precision)
    print(f"Step size: {step_size}, Precisión: {precision}, Cantidad original: {quantity}")
    return quantity

# Función para verificar la conexión a la API de Binance
def check_connection():
    try:
        client.ping()
        return True
    except Exception as e:
        print(f"{Fore.RED}Error de conexión: {e}{Style.RESET_ALL}")
        return False

# Función para obtener órdenes abiertas
def get_open_orders(symbol):
    try:
        open_orders = client.get_open_orders(symbol=symbol)
        print(f"{Fore.CYAN}Órdenes abiertas: {open_orders}{Style.RESET_ALL}")
        return open_orders
    except Exception as e:
        print(f"{Fore.RED}Error al obtener órdenes abiertas: {e}{Style.RESET_ALL}")
        return []

# Función para obtener el estado actual del mercado
def get_market_status(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        print(f"{Fore.CYAN}Estado actual del mercado: {ticker}{Style.RESET_ALL}")
        return ticker
    except Exception as e:
        print(f"{Fore.RED}Error al obtener el estado del mercado: {e}{Style.RESET_ALL}")
        return None

# Función para obtener las 10 mejores monedas por volumen y otros indicadores
def get_top_symbols():
    try:
        tickers = client.get_ticker()
        tickers_sorted = sorted(tickers, key=lambda x: float(x['volume']), reverse=True)
        top_symbols = [ticker['symbol'] for ticker in tickers_sorted[:10]]
        print(f"{Fore.CYAN}Monedas seleccionadas:{Style.RESET_ALL}")
        for i, symbol in enumerate(top_symbols, 1):
            print(f"{i}. {symbol}")
        return top_symbols
    except Exception as e:
        print(f"{Fore.RED}Error al obtener las mejores monedas: {e}{Style.RESET_ALL}")
        return []

# Función para calcular el trailing stop-loss dinámico
def dynamic_trailing_stop_loss(buy_price, current_price, trailing_stop_loss_percentage):
    trailing_stop_loss_price = buy_price * (1 - trailing_stop_loss_percentage)
    if current_price > buy_price:
        trailing_stop_loss_price = max(trailing_stop_loss_price, current_price * (1 - trailing_stop_loss_percentage))
    return trailing_stop_loss_price

# Función principal del bot
def trading_bot():
    short_window = 10
    long_window = 50
    lookback = '1000'
    interval = '1m'
    buy_price = None
    stop_loss_price = None
    take_profit_price = None
    last_check_time = time.time()

    # Obtener las 10 mejores monedas por volumen
    top_symbols = get_top_symbols()
    if not top_symbols:
        return

    # Permitir al usuario seleccionar 3 monedas
    selected_symbols = []
    while len(selected_symbols) < 3:
        try:
            choice = int(input(f"Seleccione una moneda (número, {3 - len(selected_symbols)} restantes): "))
            if 1 <= choice <= len(top_symbols) and top_symbols[choice - 1] not in selected_symbols:
                selected_symbols.append(top_symbols[choice - 1])
            else:
                print(f"{Fore.RED}Selección inválida. Intente nuevamente.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Entrada inválida. Intente nuevamente.{Style.RESET_ALL}")

    # Obtener datos históricos y entrenar modelo para cada moneda
    models = {}
    scalers = {}
    for symbol in selected_symbols:
        data = get_historical_data(symbol, interval, lookback)
        if data is None:
            continue
        data = calculate_indicators(data)
        model, scaler = train_model(data)
        models[symbol] = model
        scalers[symbol] = scaler

    try:
        while True:
            print(f"{Fore.BLUE}Iniciando iteración del bot...{Style.RESET_ALL}")
            if not check_connection():
                time.sleep(60)
                continue

            for symbol in selected_symbols:
                data = get_historical_data(symbol, interval, lookback)
                if data is None:
                    continue

                data = calculate_indicators(data)
                X = data[['MA_short', 'MA_long', 'RSI', 'MACD', 'Stochastic', 'ATR', 'ADX', 'Bollinger_High', 'Bollinger_Low']].iloc[-1:]
                X_scaled = scalers[symbol].transform(X)
                signal = models[symbol].predict(X_scaled)[0]

                # Verificar saldo disponible en la moneda seleccionada y USDT
                available_asset = check_balance(symbol.replace('USDT', ''))
                available_usdt = check_balance('USDT')

                if signal == 1 and available_usdt < initial_usd_amount:
                    # Vender otra moneda para obtener USDT
                    print(f"{Fore.YELLOW}Saldo USDT insuficiente para comprar {symbol}. Necesitas recargar USDT.{Style.RESET_ALL}")
                elif signal == 1 and available_usdt >= initial_usd_amount + min_usdt_balance:
                    # Comprar la moneda seleccionada
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    quantity = (available_usdt - min_usdt_balance) / current_price
                    quantity = round_quantity(symbol, quantity)
                    if quantity * current_price >= initial_usd_amount:
                        print(f"{Fore.GREEN}Señal de compra detectada. Comprando {quantity} {symbol.replace('USDT', '')} a {current_price:.8f}...{Style.RESET_ALL}")
                        order = buy_order(symbol, quantity)
                        if order:
                            buy_price = float(order['fills'][0]['price'])
                            stop_loss_price = calculate_stop_loss(buy_price, stop_loss_percentage)
                            take_profit_price = calculate_take_profit(buy_price, take_profit_percentage)
                            print(f"{Fore.GREEN}Precio de compra: {buy_price:.8f}, Stop-loss: {stop_loss_price:.8f}, Take-profit: {take_profit_price:.8f}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}El valor de la orden es menor que el mínimo requerido.{Style.RESET_ALL}")
                elif signal == -1 and available_asset > 0:
                    # Vender la moneda seleccionada
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    quantity = min(round_quantity(symbol, available_asset), available_asset)
                    if quantity > 0:
                        print(f"{Fore.YELLOW}Señal de venta detectada. Vendiendo {quantity} {symbol.replace('USDT', '')} a {current_price:.8f}...{Style.RESET_ALL}")
                        sell_order(symbol, quantity, buy_price)
                        buy_price = None
                        stop_loss_price = None
                        take_profit_price = None
                        print(f"{Fore.YELLOW}Venta completada.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}La cantidad de {symbol.replace('USDT', '')} a vender es inválida.{Style.RESET_ALL}")
                else:
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    print(f"{Fore.LIGHTGREEN_EX}Mercado desfavorable... Precio actual: {current_price:.8f}{Style.RESET_ALL}")

                # Verificar stop-loss y trailing stop-loss
                if buy_price is not None:
                    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    if current_price <= stop_loss_price:
                        print(f"{Fore.RED}Stop-loss alcanzado. Vendiendo a {current_price:.8f}...{Style.RESET_ALL}")
                        sell_order(symbol, round_quantity(symbol, initial_usd_amount / buy_price), buy_price)
                        buy_price = None
                        stop_loss_price = None
                        take_profit_price = None
                        print(f"{Fore.RED}Venta por stop-loss completada.{Style.RESET_ALL}")

                    # Verificar take-profit en cada iteración
                    elif current_price >= take_profit_price:
                        print(f"{Fore.GREEN}Take-profit alcanzado. Vendiendo la mitad de la posición a {current_price:.8f}...{Style.RESET_ALL}")
                        half_quantity = round_quantity(symbol, (initial_usd_amount / buy_price) / 2)
                        sell_order(symbol, half_quantity, buy_price)
                        print(f"{Fore.GREEN}Venta por take-profit completada.{Style.RESET_ALL}")

                    # Verificar trailing stop-loss
                    trailing_stop_loss_price = dynamic_trailing_stop_loss(buy_price, current_price, trailing_stop_loss_percentage)
                    if current_price <= trailing_stop_loss_price:
                        print(f"{Fore.RED}Trailing stop-loss dinámico alcanzado. Vendiendo la otra mitad de la posición a {current_price:.8f}...{Style.RESET_ALL}")
                        half_quantity = round_quantity(symbol, (initial_usd_amount / buy_price) / 2)
                        sell_order(symbol, half_quantity, buy_price)
                        buy_price = None
                        stop_loss_price = None
                        take_profit_price = None
                        print(f"{Fore.RED}Venta por trailing stop-loss dinámico completada.{Style.RESET_ALL}")

                # Mostrar datos formateados
                print(f"{Fore.CYAN}Datos actuales para {symbol}:{Style.RESET_ALL}")
                print(tabulate(data.tail(), headers='keys', tablefmt='pretty', showindex=False))

                # Obtener y mostrar información de órdenes abiertas
                open_orders = get_open_orders(symbol)
                print(f"{Fore.CYAN}Número de órdenes abiertas para {symbol}: {len(open_orders)}{Style.RESET_ALL}")

                # Obtener y mostrar el estado actual del mercado
                market_status = get_market_status(symbol)
                print(f"{Fore.CYAN}Precio actual para {symbol}: {market_status['price']}{Style.RESET_ALL}")

            time.sleep(30)  # Esperar 30 segundos antes de la siguiente iteración

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}Bot detenido por el usuario.{Style.RESET_ALL}")

if __name__ == "__main__":
    trading_bot()