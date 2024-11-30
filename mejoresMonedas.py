import asyncio
import os
import aiohttp
import pandas as pd

# Si estamos en Windows, aseguramos el uso de un bucle compatible
if os.name == 'nt':  # Si estamos en Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# URL base de la API de Binance
BASE_URL = "https://api.binance.com/api/v3"

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

# Función para calcular indicadores técnicos (Volatilidad, SMA, Volumen)
def calcular_indicadores(df):
    df['volatilidad'] = df['close'].pct_change().rolling(window=14).std()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['volumen_promedio'] = df['volume'].rolling(window=14).mean()
    return df

# Función para evaluar las mejores monedas basadas en criterios
async def evaluar_monedas(session, intervalo='1h', limite=168):
    pares_trading = await obtener_pares_trading(session)
    mejores_monedas = []
    tareas = []

    for moneda in pares_trading:
        tareas.append(obtener_datos_asincrono(session, moneda, intervalo, limite))

    resultados = await asyncio.gather(*tareas)

    for i, df in enumerate(resultados):
        if df is not None:
            df = calcular_indicadores(df)
            volatilidad = df['volatilidad'].iloc[-1]
            volumen = df['volume'].iloc[-1]
            volumen_promedio = df['volumen_promedio'].iloc[-1]
            sma_actual = df['SMA_50'].iloc[-1]
            precio_actual = df['close'].iloc[-1]

            if volatilidad > 0.02 and volumen > volumen_promedio and precio_actual > sma_actual:
                mejores_monedas.append({
                    'moneda': pares_trading[i],
                    'volatilidad': volatilidad,
                    'volumen': volumen,
                    'volumen_promedio': volumen_promedio,
                    'precio_actual': precio_actual,
                    'SMA_50': sma_actual
                })

    mejores_monedas = sorted(mejores_monedas, key=lambda x: x['volatilidad'], reverse=True)
    return mejores_monedas

# Ejecutamos la función asincrónica
async def main():
    async with aiohttp.ClientSession() as session:
        resultados = await evaluar_monedas(session)
        for resultado in resultados:
            print(f"Moneda: {resultado['moneda']}, Volatilidad: {resultado['volatilidad']:.4f}, Volumen: {resultado['volumen']}, SMA_50: {resultado['SMA_50']:.2f}, Precio Actual: {resultado['precio_actual']:.2f}")

# Ejecutar el bucle principal
asyncio.run(main())
