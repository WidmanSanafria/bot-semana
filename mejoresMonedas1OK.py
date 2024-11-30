import asyncio
import os
import aiohttp
import pandas as pd
import talib
import argparse


# Si estamos en Windows, aseguramos el uso de un bucle compatible
if os.name == 'nt':  # Si estamos en Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# URL base de la API de Binance
BASE_URL = "https://api.binance.com/api/v3"

# Función para configurar los parámetros
def cargar_configuracion():
    # Definimos los parámetros de configuración como variables dinámicas
    # Estos valores pueden ser modificados por el usuario o pasados como argumentos
    return {
        "UMBRAL_VOLATILIDAD_COMPRA": 0.04,  # Umbral de volatilidad para compra
        "UMBRAL_SMA_COMPRA": 1.05,  # 5% por encima de la SMA_50 para compra
        "UMBRAL_RSI_COMPRA_MIN": 30,  # RSI mínimo para compra
        "UMBRAL_RSI_COMPRA_MAX": 70,  # RSI máximo para compra

        "UMBRAL_VOLATILIDAD_VENTA": 0.02,  # Umbral de volatilidad para venta
        "UMBRAL_SMA_VENTA": 0.95,  # 5% por debajo de la SMA_50 para venta
        "UMBRAL_RSI_VENTA": 60,  # RSI para venta

        "intervalo": "1h",  # Intervalo de las velas (1m, 5m, 1h, 1d)
        "limite": 1680  # Limite de datos (por defecto, 168 horas = 7 días)
    }

# Función para manejar argumentos de la línea de comandos
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluar monedas en Binance")
    parser.add_argument('--intervalo', type=str, default='1h', help='Intervalo de tiempo (1m, 5m, 1h, 1d)')
    parser.add_argument('--limite', type=int, default=168, help='Número de datos a obtener')
    return parser.parse_args()

# Obtener parámetros de configuración desde la línea de comandos
args = parse_args()
config = cargar_configuracion()
config["intervalo"] = args.intervalo  # Actualizar el intervalo con el argumento
config["limite"] = args.limite  # Actualizar el límite con el argumento


# Función para obtener los pares de trading
async def obtener_pares_trading(session):
    try:
        url = f"{BASE_URL}/exchangeInfo"
        async with session.get(url) as response:
            exchange_info = await response.json()
            pares = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USDT']
            return pares
    except Exception as e:
        print(f"Error al obtener pares de trading: {e}")
        return []


# Función asincrónica para obtener datos históricos de una moneda
async def obtener_datos_asincrono(session, moneda, intervalo='1h', limite=168):  # 168 horas = 7 días
    try:
        url = f"{BASE_URL}/klines"
        params = {
            'symbol': moneda,
            'interval': intervalo,
            'limit': limite
        }
        async with session.get(url, params=params) as response:
            klines = await response.json()
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data['close'] = pd.to_numeric(data['close'])
            data['volume'] = pd.to_numeric(data['volume'])
            return data
    except Exception as e:
        print(f"Error al obtener datos para {moneda}: {e}")
        return None


# Función para calcular indicadores técnicos (Volatilidad, SMA, RSI, MACD, Volumen)
def calcular_indicadores(df):
    df['volatilidad'] = df['close'].pct_change().rolling(window=14).std()  # Volatilidad de 14 períodos
    df['SMA_50'] = df['close'].rolling(window=50).mean()  # SMA de 50 períodos
    df['volumen_promedio'] = df['volume'].rolling(window=14).mean()  # Volumen promedio de 14 períodos
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  # RSI de 14 períodos
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  # MACD
    return df


# Función para evaluar las mejores monedas basadas en criterios de compra
async def evaluar_monedas_compra(session, config):
    pares_trading = await obtener_pares_trading(session)
    mejores_monedas = []
    tareas = []

    for moneda in pares_trading:
        tareas.append(obtener_datos_asincrono(session, moneda, config["intervalo"], config["limite"]))

    resultados = await asyncio.gather(*tareas)

    for i, df in enumerate(resultados):
        if df is not None:
            df = calcular_indicadores(df)
            volatilidad = df['volatilidad'].iloc[-1]
            volumen = df['volume'].iloc[-1]
            volumen_promedio = df['volumen_promedio'].iloc[-1]
            sma_actual = df['SMA_50'].iloc[-1]
            precio_actual = df['close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_hist = df['MACD_hist'].iloc[-1]

            # Condiciones de filtrado (ajustadas para compra)
            if volatilidad > config["UMBRAL_VOLATILIDAD_COMPRA"] and volumen > volumen_promedio and precio_actual > sma_actual * config["UMBRAL_SMA_COMPRA"] and config["UMBRAL_RSI_COMPRA_MIN"] < rsi < config["UMBRAL_RSI_COMPRA_MAX"] and macd > macd_signal:
                mejores_monedas.append({
                    'moneda': pares_trading[i],
                    'volatilidad': volatilidad,
                    'volumen': volumen,
                    'volumen_promedio': volumen_promedio,
                    'precio_actual': precio_actual,
                    'SMA_50': sma_actual,
                    'RSI': rsi,
                    'MACD': macd,
                    'MACD_signal': macd_signal,
                    'MACD_hist': macd_hist
                })

    mejores_monedas = sorted(mejores_monedas, key=lambda x: x['volatilidad'], reverse=True)
    return mejores_monedas


# Función para evaluar las mejores monedas basadas en criterios de venta
async def evaluar_monedas_venta(session, config):
    pares_trading = await obtener_pares_trading(session)
    mejores_monedas = []
    tareas = []

    for moneda in pares_trading:
        tareas.append(obtener_datos_asincrono(session, moneda, config["intervalo"], config["limite"]))

    resultados = await asyncio.gather(*tareas)

    for i, df in enumerate(resultados):
        if df is not None:
            df = calcular_indicadores(df)
            volatilidad = df['volatilidad'].iloc[-1]
            volumen = df['volume'].iloc[-1]
            volumen_promedio = df['volumen_promedio'].iloc[-1]
            sma_actual = df['SMA_50'].iloc[-1]
            precio_actual = df['close'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            macd_hist = df['MACD_hist'].iloc[-1]

            # Condiciones de filtrado (ajustadas para venta)
            cumple_condiciones = True

            if volatilidad <= config["UMBRAL_VOLATILIDAD_VENTA"]:
                cumple_condiciones = False
            if volumen <= volumen_promedio:
                cumple_condiciones = False
            if precio_actual >= sma_actual * config["UMBRAL_SMA_VENTA"]:
                cumple_condiciones = False
            if rsi <= config["UMBRAL_RSI_VENTA"]:
                cumple_condiciones = False
            if macd >= macd_signal:
                cumple_condiciones = False

            if cumple_condiciones:
                mejores_monedas.append({
                    'moneda': pares_trading[i],
                    'volatilidad': volatilidad,
                    'volumen': volumen,
                    'volumen_promedio': volumen_promedio,
                    'precio_actual': precio_actual,
                    'SMA_50': sma_actual,
                    'RSI': rsi,
                    'MACD': macd,
                    'MACD_signal': macd_signal,
                    'MACD_hist': macd_hist
                })

    mejores_monedas = sorted(mejores_monedas, key=lambda x: x['volatilidad'], reverse=True)
    return mejores_monedas


# Función principal para ejecutar el análisis
async def main():
    async with aiohttp.ClientSession() as session:
        # Evaluamos las mejores monedas para compra
        mejores_monedas_compra = await evaluar_monedas_compra(session, config)
        print("\nMejores monedas para comprar:")
        for moneda in mejores_monedas_compra:
            print(moneda)

        # Evaluamos las mejores monedas para venta
        mejores_monedas_venta = await evaluar_monedas_venta(session, config)
        print("\nMejores monedas para vender:")
        for moneda in mejores_monedas_venta:
            print(moneda)


# Ejecutamos la función principal
if __name__ == "__main__":
    asyncio.run(main())
