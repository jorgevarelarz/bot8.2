import asyncio
import ccxt.pro as ccxt
import aiosqlite
import numpy as np
import talib
import logging
import aiohttp
import ssl
import certifi
from datetime import datetime, timedelta, timezone
import random
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Utiliza ccxt.pro con WebSockets y filtra dinámicamente los símbolos válidos.

# =====================================
# Configuración y Parámetros Globales
# =====================================
CONFIG = {
    "API_KEY": "q41oOMmogN7lqIREmzHWtLv5ozggFojEhrA1faNZNydWhoRGSM2bwR2xkvXsheYV",
    "API_SECRET": "p6cMNjoDQ27B1GgCO5E1t03COB0iJZskcpxjipAxuZqUiJtNlQWFr7rkGhI7oO4y",
    "TELEGRAM_TOKEN": "8270934653:AAGyyNG5EjwKnVoTM1e3wxD2C8tJo7sZIus",
    "TELEGRAM_CHAT_ID": "8178071066",
    "COMISION_BINANCE": 0.001,
    "CAPITAL_POR_OPERACION": 0.5,   # Porcentaje del saldo asignado a cada operación
    "STOP_LOSS_PORCENTAJE": 0.02,
    "TAKE_PROFIT_PORCENTAJE": 0.05,
    "TRAILING_STOP_PORCENTAJE": 0.03,
    "UMBRAL_COMPRA": 70,           # Umbral mínimo para operaciones en vivo
    "UMBRAL_VENTA": 30,            # Umbral máximo para operaciones en vivo
    # Parámetros para backtesting (ajustados según los resultados)
    "UMBRAL_COMPRA_BACKTEST": 30,  # Reducido para aumentar señales de entrada
    "UMBRAL_VENTA_BACKTEST": 25,   # Ajustado para salir con mayor frecuencia
    "SYMBOL": "BTC/USDT",
    "ATR_MULTIPLIER_STOP": 1.5,
    "ATR_MULTIPLIER_TP": 3.0,
    "EMA_TREND_PERIOD": 50,
    "BBANDS_PERIOD": 20,
    "BBANDS_STD": 2,
    "BALANCE_UPDATE_INTERVAL": 300  # 5 minutos
}

# Configurar contexto SSL usando certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Configurar logging para consola y archivo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", mode="a")
    ]
)

logger = logging.getLogger(__name__)


async def notify_telegram(msg: str):
    if not (CONFIG.get("TELEGRAM_TOKEN") and CONFIG.get("TELEGRAM_CHAT_ID")):
        return
    url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
    async with aiohttp.ClientSession() as s:
        r = await s.post(url, data={'chat_id': CONFIG['TELEGRAM_CHAT_ID'], 'text': msg})
        text = await r.text()
        if r.status != 200:
            logger.error(f"Telegram error {r.status}: {text}")


async def filter_valid_symbols(exchange, symbols):
    markets = await exchange.load_markets()
    valid = [s for s in symbols if s in markets]
    invalid = set(symbols) - set(valid)
    if invalid:
        logger.warning(f"Símbolos inválidos descartados: {invalid}")
    return valid

# ========================================================
# MÓDULO DE MACHINE LEARNING CON FEATURES ENRIQUECIDAS
# ========================================================
def entrenar_y_optimizar_parametros(db_path, current_umbral_compra, current_umbral_venta, current_indicators):
    """
    Entrena un modelo utilizando trades históricos enriquecidos con indicadores técnicos y
    optimiza los umbrales de compra y venta basándose en la relación riesgo/recompensa.
    """
    conn = sqlite3.connect(db_path)
    # Verificar si las columnas existen, si no, agregarlas
    cursor = conn.execute("PRAGMA table_info(trades)")
    columns = [row[1] for row in cursor.fetchall()]
    for col in ['rsi', 'ema', 'macd', 'atr']:
        if col not in columns:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} REAL")
    conn.commit()

    df = pd.read_sql_query("SELECT precio, stop_loss, take_profit, rsi, ema, macd, atr FROM trades", conn)
    conn.close()

    if df.empty:
        return current_umbral_compra, current_umbral_venta

    df['profit_ratio'] = (df['take_profit'] - df['precio']) / (df['precio'] - df['stop_loss'] + 1e-6)
    df['umbral_compra'] = current_umbral_compra
    df['umbral_venta'] = current_umbral_venta

    X = df[['umbral_compra', 'umbral_venta', 'rsi', 'ema', 'macd', 'atr']]
    y = df['profit_ratio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Preparar los features actuales a partir de los indicadores actuales
    base_features = np.array([
        current_indicators['rsi'],
        current_indicators['ema'],
        current_indicators['macd'],
        current_indicators['atr']
    ])

    # Nombres de columnas para el DataFrame
    feature_names = ['umbral_compra', 'umbral_venta', 'rsi', 'ema', 'macd', 'atr']

    mejor_score = -np.inf
    mejor_umbral_compra = current_umbral_compra
    mejor_umbral_venta = current_umbral_venta

    # Usamos un DataFrame para pasar los features al modelo
    for uc in range(max(0, current_umbral_compra - 10), min(100, current_umbral_compra + 10) + 1):
        for uv in range(max(0, current_umbral_venta - 10), min(100, current_umbral_venta + 10) + 1):
            features = pd.DataFrame(
                [np.concatenate((np.array([uc, uv]), base_features))],
                columns=feature_names
            )
            score = model.predict(features)[0]
            if score > mejor_score:
                mejor_score = score
                mejor_umbral_compra = uc
                mejor_umbral_venta = uv

    return mejor_umbral_compra, mejor_umbral_venta

# ========================================================
# CLASE ASYNCTRADINGBOT CON MEJORAS PARA PRODUCCIÓN
# ========================================================
class AsyncTradingBot:
    def __init__(self, config):
        self.config = config
        self.symbol = config.get("SYMBOL", "BTC/USDT")
        self.exchange = ccxt.binance({
            'apiKey': config["API_KEY"],
            'secret': config["API_SECRET"],
            'options': {'adjustForTimeDifference': True},
        })
        self.db = None
        self.order_open = False
        self.entry_price = 0.0
        self.trailing_stop = None
        self.last_update_id = 0
        # Inicializamos last_balance_sent como offset-aware (por ejemplo, epoch en UTC)
        self.last_balance_sent = datetime.fromtimestamp(0, timezone.utc)
        self.last_hour_balance = None
        self.lock = asyncio.Lock()
        self.session = None
        # Variables de caché para el tipo de cambio EUR
        self._cached_eur_rate = None
        self._eur_cache_time = None

    async def setup(self):
        await self.setup_database()
        symbols = await filter_valid_symbols(self.exchange, [self.symbol])
        if not symbols:
            logger.error("No hay símbolos válidos.")
            raise RuntimeError("Símbolo inválido")
        self.symbol = symbols[0]
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))

    async def setup_database(self):
        self.db = await aiosqlite.connect("trading_log.db")
        await self.db.execute('''CREATE TABLE IF NOT EXISTS trades (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    fecha TEXT,
                                    tipo TEXT,
                                    par TEXT,
                                    cantidad REAL,
                                    precio REAL,
                                    stop_loss REAL,
                                    take_profit REAL,
                                    rsi REAL,
                                    ema REAL,
                                    macd REAL,
                                    atr REAL)''')
        await self.db.commit()

    async def fetch_with_retry(self, func, *args, retries=3, initial_delay=1, backoff_factor=2, **kwargs):
        delay = initial_delay
        for attempt in range(retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                await asyncio.sleep(delay)
                delay *= backoff_factor

    async def obtener_tipo_cambio_eur(self):
        # Usamos caché durante 10 segundos para evitar múltiples solicitudes
        if self._cached_eur_rate and (datetime.now(timezone.utc) - self._eur_cache_time).total_seconds() < 10:
            return self._cached_eur_rate
        try:
            ticker = await self.fetch_with_retry(self.exchange.watch_ticker, "EUR/USDT")
            rate = ticker['last']
            self._cached_eur_rate = rate
            self._eur_cache_time = datetime.now(timezone.utc)
            return rate
        except Exception as e:
            logging.error(f"Error al obtener tipo de cambio EUR/USDT: {e}")
            return 1

    async def obtener_detalle_balance(self):
        try:
            balance = await self.fetch_with_retry(self.exchange.fetch_balance)
        except Exception as e:
            logging.error(f"Error al obtener balance: {e}")
            return "No se pudo obtener el balance."
        conversion_rate = await self.obtener_tipo_cambio_eur()
        total_usdt = 0.0
        message_lines = []
        emojis = {
            "USDT": "💵",
            "BTC": "₿",
            "ETH": "⚡",
            "BNB": "🧊",
            "ADA": "🅰️",
            "XRP": "🌊",
            "DOGE": "🐶",
        }
        for asset, amount in balance.get('total', {}).items():
            if amount > 0:
                if asset == "USDT":
                    value_usdt = amount
                else:
                    pair = f"{asset}/USDT"
                    try:
                        ticker = await self.fetch_with_retry(self.exchange.watch_ticker, pair)
                        value_usdt = amount * ticker['last']
                    except Exception as e:
                        logging.warning(f"No se pudo obtener ticker para {pair}: {e}")
                        value_usdt = 0
                total_usdt += value_usdt
                asset_emoji = emojis.get(asset, "🔸")
                value_eur = value_usdt / conversion_rate if conversion_rate != 0 else 0
                message_lines.append(f"{asset_emoji} *{asset}*: {amount:.4f} (≈ {value_eur:.2f}€)")
        total_eur = total_usdt / conversion_rate if conversion_rate != 0 else total_usdt
        message = f"💰 *Balance Total: {total_eur:.2f}€*\n" + "\n".join(message_lines)
        return message

    async def enviar_mensaje_telegram(self, mensaje):
        url = f"https://api.telegram.org/bot{self.config['TELEGRAM_TOKEN']}/sendMessage"
        data = {"chat_id": self.config['TELEGRAM_CHAT_ID'], "text": mensaje, "parse_mode": "Markdown"}
        try:
            async with self.session.post(url, data=data) as resp:
                if resp.status != 200:
                    logging.error(f"Error al enviar mensaje a Telegram. Estado: {resp.status}")
        except Exception as e:
            logging.error(f"Excepción al enviar mensaje a Telegram: {e}")

    async def enviar_saldo_telegram(self):
        ahora = datetime.now(timezone.utc)
        if (ahora - self.last_balance_sent).total_seconds() >= self.config["BALANCE_UPDATE_INTERVAL"]:
            mensaje = await self.obtener_detalle_balance()
            await self.enviar_mensaje_telegram(mensaje)
            self.last_balance_sent = ahora

    async def enviar_pnl_horario(self):
        conversion_rate = await self.obtener_tipo_cambio_eur()
        total_usdt = await self.obtener_saldo_total()
        total_eur = total_usdt / conversion_rate if conversion_rate != 0 else total_usdt

        if self.last_hour_balance is None:
            self.last_hour_balance = total_eur
            mensaje = f"⏱ *Inicio del seguimiento horario*\nBalance inicial: {total_eur:.2f}€"
            await self.enviar_mensaje_telegram(mensaje)
        else:
            diff = total_eur - self.last_hour_balance
            perc = (diff / self.last_hour_balance * 100) if self.last_hour_balance != 0 else 0
            emoji = "📈" if diff >= 0 else "📉"
            signo = "+" if diff >= 0 else ""
            mensaje = (f"⏱ *Resumen Horario*\n"
                       f"Balance actual: {total_eur:.2f}€\n"
                       f"{emoji} Ganancia/Pérdida: {signo}{diff:.2f}€ ({signo}{perc:.2f}%)")
            await self.enviar_mensaje_telegram(mensaje)
            self.last_hour_balance = total_eur

    async def obtener_saldo_total(self):
        try:
            balance = await self.fetch_with_retry(self.exchange.fetch_balance)
        except Exception as e:
            logging.error(f"Error al obtener balance: {e}")
            return 0.0
        total_usdt = 0.0
        for asset, amount in balance.get('total', {}).items():
            if amount > 0:
                if asset == "USDT":
                    total_usdt += amount
                else:
                    pair = f"{asset}/USDT"
                    try:
                        ticker = await self.fetch_with_retry(self.exchange.watch_ticker, pair)
                        total_usdt += amount * ticker['last']
                    except Exception as e:
                        logging.warning(f"No se pudo obtener ticker para {pair}: {e}")
                        continue
        return round(total_usdt, 2)

    async def obtener_indicadores(self, par=None):
        if par is None:
            par = self.symbol
        try:
            # Calcular since para 100 velas de 5m (500 minutos atrás)
            since = int((datetime.now(timezone.utc) - timedelta(minutes=500)).timestamp() * 1000)
            data = await self.fetch_with_retry(self.exchange.watch_ohlcv, par, '5m', since, 100)
        except Exception as e:
            logging.error(f"Error al obtener OHLCV: {e}")
            return {"rsi": 0, "ema": 0, "macd": 0, "atr": 0,
                    "last_price": 0, "upperband": 0, "middleband": 0, "lowerband": 0}
        
        data = np.array(data)
        close_prices = data[:, 4]
        high_prices = data[:, 2]
        low_prices = data[:, 3]
        last_price = close_prices[-1]

        rsi = talib.RSI(close_prices, timeperiod=14)[-1]
        ema = talib.EMA(close_prices, timeperiod=20)[-1]
        macd, _, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        upperband, middleband, lowerband = talib.BBANDS(
            close_prices, 
            timeperiod=self.config["BBANDS_PERIOD"],
            nbdevup=self.config["BBANDS_STD"],
            nbdevdn=self.config["BBANDS_STD"],
            matype=0
        )
        return {
            "rsi": round(rsi, 2),
            "ema": round(ema, 2),
            "macd": round(macd[-1], 2),
            "atr": round(atr, 2),
            "last_price": round(last_price, 2),
            "upperband": round(upperband[-1], 2),
            "middleband": round(middleband[-1], 2),
            "lowerband": round(lowerband[-1], 2)
        }

    async def obtener_tendencia(self):
        try:
            # Calcular since para 50 velas de 1h (50 horas atrás)
            since = int((datetime.now(timezone.utc) - timedelta(hours=50)).timestamp() * 1000)
            data = await self.fetch_with_retry(self.exchange.watch_ohlcv, self.symbol, '1h', since, 50)
        except Exception as e:
            logging.error(f"Error al obtener OHLCV de 1h: {e}")
            return None
        data = np.array(data)
        close_prices = data[:, 4]
        ema_trend = talib.EMA(close_prices, timeperiod=self.config["EMA_TREND_PERIOD"])[-1]
        current_price = close_prices[-1]
        tendencia = current_price > ema_trend
        logging.info(f"Tendencia 1h: Precio actual {current_price:.2f} vs EMA {ema_trend:.2f} -> {'Bullish' if tendencia else 'Bearish'}")
        return tendencia

    async def calcular_probabilidades(self, par=None):
        if par is None:
            par = self.symbol
        indicadores = await self.obtener_indicadores(par)
        rsi = indicadores['rsi']
        ema = indicadores['ema']
        macd = indicadores['macd']
        last_price = indicadores['last_price']
        atr = indicadores['atr']
        upperband = indicadores['upperband']
        lowerband = indicadores['lowerband']
        
        compra = 0
        venta = 0

        if rsi < 30:
            compra += 50
        elif rsi > 70:
            venta += 50

        if last_price > ema:
            compra += 20
        else:
            venta += 20

        if macd > 0:
            compra += 20
        else:
            venta += 20
        
        if last_price < lowerband:
            compra += 10
        elif last_price > upperband:
            venta += 10

        return compra, venta, atr

    async def obtener_precio(self, par=None):
        if par is None:
            par = self.symbol
        try:
            ticker = await self.fetch_with_retry(self.exchange.watch_ticker, par)
            return ticker['last']
        except Exception as e:
            logging.error(f"Error al obtener precio para {par}: {e}")
            return 0.0

    async def evaluar_orden(self):
        async with self.lock:
            if self.order_open:
                precio_actual = await self.obtener_precio(self.symbol)
                _, _, atr = await self.calcular_probabilidades(self.symbol)
                if atr == 0:
                    atr = precio_actual * self.config["ATR_MULTIPLIER_STOP"]

                stop_loss = self.entry_price - (atr * self.config["ATR_MULTIPLIER_STOP"])
                take_profit = self.entry_price + (atr * self.config["ATR_MULTIPLIER_TP"])

                if self.trailing_stop is None:
                    self.trailing_stop = self.entry_price
                
                if precio_actual > self.entry_price:
                    nuevo_trailing = precio_actual - (atr * self.config["ATR_MULTIPLIER_STOP"])
                    if nuevo_trailing > self.trailing_stop:
                        self.trailing_stop = nuevo_trailing
                        logging.info(f"Nuevo nivel de trailing stop: {self.trailing_stop:.2f}")

                logging.info(f"Precio actual: {precio_actual:.2f}, Stop Loss: {stop_loss:.2f}, "
                             f"Take Profit: {take_profit:.2f}, Trailing Stop: {self.trailing_stop:.2f}")

                if precio_actual <= stop_loss:
                    await self.cerrar_orden('Stop Loss', stop_loss, take_profit)
                elif precio_actual >= take_profit:
                    await self.cerrar_orden('Take Profit', stop_loss, take_profit)
                elif precio_actual <= self.trailing_stop:
                    await self.cerrar_orden('Trailing Stop', stop_loss, take_profit)

    async def cerrar_orden(self, razon, stop_loss, take_profit):
        async with self.lock:
            if self.order_open:
                mensaje = f"❌ Orden cerrada por *{razon}*. Precio de entrada: {self.entry_price:.2f}"
                await self.enviar_mensaje_telegram(mensaje)
                await notify_telegram(mensaje)
                saldo = await self.obtener_saldo_total()
                _, _, atr = await self.calcular_probabilidades(self.symbol)
                distancia_sl = atr * self.config["ATR_MULTIPLIER_STOP"]
                posicion_sugerida = (saldo * self.config["CAPITAL_POR_OPERACION"]) / distancia_sl if distancia_sl > 0 else 0
                indicadores = await self.obtener_indicadores(self.symbol)
                await self.log_trade("venta", self.entry_price, stop_loss, take_profit, posicion_sugerida, indicadores)
                self.order_open = False
                self.entry_price = 0.0
                self.trailing_stop = None

    async def abrir_orden(self, tipo_orden):
        async with self.lock:
            if not self.order_open:
                self.order_open = True
                self.entry_price = await self.obtener_precio(self.symbol)
                _, _, atr = await self.calcular_probabilidades(self.symbol)
                self.trailing_stop = self.entry_price

                saldo = await self.obtener_saldo_total()
                riesgo_operacion = saldo * self.config["CAPITAL_POR_OPERACION"]
                distancia_sl = atr * self.config["ATR_MULTIPLIER_STOP"]
                posicion_sugerida = riesgo_operacion / distancia_sl if distancia_sl > 0 else 0

                mensaje = (f"✅ Orden de *{tipo_orden}* abierta.\n"
                           f"Precio de entrada: {self.entry_price:.2f}\n"
                           f"Tamaño de posición sugerido: {posicion_sugerida:.4f} unidades")
                await self.enviar_mensaje_telegram(mensaje)
                await notify_telegram(mensaje)

    async def evaluar_mercado(self):
        async with self.lock:
            compra_prob, venta_prob, _ = await self.calcular_probabilidades(self.symbol)
            tendencia = await self.obtener_tendencia()
            logging.info(f"Probabilidades - Compra: {compra_prob}%, Venta: {venta_prob}%")
            if compra_prob >= self.config["UMBRAL_COMPRA"] and not self.order_open:
                if tendencia is None or tendencia:
                    await self.abrir_orden("compra")
                else:
                    logging.info("Señal de compra, pero la tendencia es bajista. No se abre la orden.")
            elif venta_prob >= self.config["UMBRAL_VENTA"] and self.order_open:
                if tendencia is None or not tendencia:
                    await self.cerrar_orden("venta", 0, 0)
                else:
                    logging.info("Señal de venta, pero la tendencia es alcista. Se mantiene la posición.")

    async def enviar_mercado_telegram(self):
        precio_actual = await self.obtener_precio(self.symbol)
        compra_prob, venta_prob, _ = await self.calcular_probabilidades(self.symbol)
        mensaje = (f"📈 *Estado del Mercado* 📉\n\n"
                   f"💲 *Precio {self.symbol}:* {precio_actual:.2f} USDT\n"
                   f"🟢 *Prob. de Comprar:* {compra_prob}%\n"
                   f"🔴 *Prob. de Vender:* {venta_prob}%")
        await self.enviar_mensaje_telegram(mensaje)

    async def manejar_comandos_telegram(self):
        url = f"https://api.telegram.org/bot{self.config['TELEGRAM_TOKEN']}/getUpdates?offset={self.last_update_id + 1}"
        try:
            async with self.session.get(url) as response:
                data = await response.json()
        except Exception as e:
            logging.error(f"Error al obtener comandos de Telegram: {e}")
            return
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if update_id:
                self.last_update_id = update_id
            message = update.get("message", {}).get("text", "")
            if message == "/mercado":
                await self.enviar_mercado_telegram()
            elif message == "/backtest":
                await self.backtest_strategy()

    # -------------------- MÓDULO DE MACHINE LEARNING --------------------
    async def actualizar_parametros_ml(self):
        current_umbral_compra = self.config["UMBRAL_COMPRA"]
        current_umbral_venta = self.config["UMBRAL_VENTA"]
        current_indicators = await self.obtener_indicadores(self.symbol)

        loop = asyncio.get_event_loop()
        nuevo_umbral_compra, nuevo_umbral_venta = await loop.run_in_executor(
            None, entrenar_y_optimizar_parametros, "trading_log.db", current_umbral_compra, current_umbral_venta, current_indicators
        )
        self.config["UMBRAL_COMPRA"] = nuevo_umbral_compra
        self.config["UMBRAL_VENTA"] = nuevo_umbral_venta
        logging.info(
            f"Optimización ML: Umbral Compra actualizado a {nuevo_umbral_compra}, "
            f"Umbral Venta actualizado a {nuevo_umbral_venta}"
        )

    async def task_actualizacion_ml(self):
        while True:
            try:
                await self.actualizar_parametros_ml()
            except Exception as e:
                logging.error(f"Error en la actualización ML: {e}")
            await asyncio.sleep(3600)

    # -------------------- BACKTESTING MEJORADO --------------------
    async def backtest_strategy(self):
        """
        Realiza una simulación mejorada de la estrategia usando datos históricos.
        Se generan señales de entrada y salida basadas en indicadores (RSI, EMA, MACD, Bollinger Bands)
        y se utiliza la tendencia (EMA de 50 períodos) para confirmar las señales.
        """
        try:
            # Se obtienen 300 velas de 1h para contar con suficientes datos (incluida la EMA de tendencia)
            since = int((datetime.now(timezone.utc) - timedelta(hours=300)).timestamp() * 1000)
            ohlcv = await self.fetch_with_retry(self.exchange.watch_ohlcv, self.symbol, '1h', since, 300)
        except Exception as e:
            logging.error("Error en backtesting: " + str(e))
            await self.enviar_mensaje_telegram("Error en backtesting.")
            return

        ohlcv = np.array(ohlcv)
        open_prices = ohlcv[:, 1]
        high_prices = ohlcv[:, 2]
        low_prices = ohlcv[:, 3]
        close_prices = ohlcv[:, 4]

        # Cálculo de indicadores globales
        rsi_array = talib.RSI(close_prices, timeperiod=14)
        ema_array = talib.EMA(close_prices, timeperiod=20)
        macd_array, _, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        atr_array = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        ema_trend = talib.EMA(close_prices, timeperiod=self.config["EMA_TREND_PERIOD"])

        profit_total = 0.0
        trades = 0
        position = False
        entry_price = 0.0

        # Umbrales para backtesting (ajustados según lo que considero adecuado)
        entry_threshold = self.config.get("UMBRAL_COMPRA_BACKTEST", 35)
        exit_threshold = self.config.get("UMBRAL_VENTA_BACKTEST", 25)

        # Iteramos desde el índice 50 para tener datos suficientes (evitar NaN)
        start_index = 50
        for i in range(start_index, len(ohlcv) - 1):
            current_close = close_prices[i]
            current_rsi = rsi_array[i]
            current_ema = ema_array[i]
            current_macd = macd_array[i]
            current_atr = atr_array[i] if not np.isnan(atr_array[i]) else 0

            # Calculamos Bollinger Bands para una ventana de 20 velas
            if i >= self.config["BBANDS_PERIOD"]:
                window = close_prices[i - self.config["BBANDS_PERIOD"] + 1: i + 1]
                upperband, middleband, lowerband = talib.BBANDS(window,
                                                               timeperiod=self.config["BBANDS_PERIOD"],
                                                               nbdevup=self.config["BBANDS_STD"],
                                                               nbdevdn=self.config["BBANDS_STD"],
                                                               matype=0)
                current_upperband = upperband[-1]
                current_lowerband = lowerband[-1]
            else:
                current_upperband = current_close
                current_lowerband = current_close

            # Se calculan las probabilidades de compra y venta
            compra = 0
            venta = 0
            if current_rsi < 30:
                compra += 50
            elif current_rsi > 70:
                venta += 50

            if current_close > current_ema:
                compra += 20
            else:
                venta += 20

            if current_macd > 0:
                compra += 20
            else:
                venta += 20

            if current_close < current_lowerband:
                compra += 10
            elif current_close > current_upperband:
                venta += 10

            # La tendencia se determina comparando el precio con la EMA de tendencia
            trend = current_close > ema_trend[i]

            # Señal de entrada: abrir posición si no hay abierta, prob. de compra supera entry_threshold y tendencia alcista
            if not position and compra >= entry_threshold and trend:
                entry_price = open_prices[i + 1]  # Simula la entrada al inicio de la siguiente vela
                position = True
                trades += 1
                logging.info(f"Abrir trade en vela {i+1} a precio {entry_price:.2f}")
            # Señal de salida: cerrar posición si prob. de venta supera exit_threshold o se revierte la tendencia
            elif position and (venta >= exit_threshold or not trend):
                exit_price = open_prices[i + 1]  # Simula la salida al inicio de la siguiente vela
                profit = exit_price - entry_price
                profit_total += profit
                logging.info(f"Cerrar trade en vela {i+1} a precio {exit_price:.2f}, beneficio: {profit:.2f}")
                position = False

        # Si hay una posición abierta al final, se cierra al precio de cierre de la última vela
        if position:
            exit_price = close_prices[-1]
            profit = exit_price - entry_price
            profit_total += profit
            trades += 1
            logging.info(f"Cerrar trade final a precio {exit_price:.2f}, beneficio: {profit:.2f}")
            position = False

        msg = f"Backtesting: {trades} trades simulados. Beneficio neto: {profit_total:.2f} USDT."
        await self.enviar_mensaje_telegram(msg)

    # -------------------- TAREAS PROGRAMADAS --------------------
    async def task_mercado(self):
        while True:
            try:
                await self.evaluar_mercado()
            except Exception as e:
                logging.error(f"Error en evaluación de mercado: {e}")
            await asyncio.sleep(60)

    async def task_orden(self):
        while True:
            try:
                await self.evaluar_orden()
            except Exception as e:
                logging.error(f"Error en evaluación de orden: {e}")
            await asyncio.sleep(60)

    async def task_comandos(self):
        while True:
            try:
                await self.manejar_comandos_telegram()
            except Exception as e:
                logging.error(f"Error en manejo de comandos: {e}")
            await asyncio.sleep(60)

    async def task_saldo(self):
        while True:
            try:
                await self.enviar_saldo_telegram()
            except Exception as e:
                logging.error(f"Error en envío de saldo: {e}")
            await asyncio.sleep(10)

    async def task_pnl(self):
        while True:
            await asyncio.sleep(3600)
            try:
                await self.enviar_pnl_horario()
            except Exception as e:
                logging.error(f"Error en envío de PnL horario: {e}")

    async def iniciar_bot(self):
        await self.setup()
        tasks = [
            asyncio.create_task(self.task_mercado()),
            asyncio.create_task(self.task_orden()),
            asyncio.create_task(self.task_comandos()),
            asyncio.create_task(self.task_saldo()),
            asyncio.create_task(self.task_pnl()),
            asyncio.create_task(self.task_actualizacion_ml())
        ]
        await asyncio.gather(*tasks)

    async def cerrar(self):
        if self.db:
            await self.db.close()
        await self.exchange.close()
        if self.session:
            await self.session.close()

    async def log_trade(self, tipo, precio, stop_loss, take_profit, posicion, indicadores):
        # Función para registrar una operación en la base de datos
        fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        async with self.db.execute(
            "INSERT INTO trades (fecha, tipo, par, cantidad, precio, stop_loss, take_profit, rsi, ema, macd, atr) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fecha, tipo, self.symbol, posicion, precio, stop_loss, take_profit, indicadores.get('rsi', 0), indicadores.get('ema', 0), indicadores.get('macd', 0), indicadores.get('atr', 0))
        ) as cursor:
            await self.db.commit()

# -------------------- INTEGRACIÓN CON UVLOOP --------------------
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logging.info("uvloop activado.")
except ImportError:
    logging.info("uvloop no disponible, se usará el loop estándar.")

async def main():
    bot = AsyncTradingBot(CONFIG)
    try:
        await bot.iniciar_bot()
    finally:
        await bot.cerrar()


async def test_order(symbol, side, amount):
    exchange = ccxt.binance({
        'apiKey': CONFIG['API_KEY'],
        'secret': CONFIG['API_SECRET'],
    })
    try:
        await exchange.load_markets()
        await exchange.watch_ticker(symbol)
        order = await exchange.create_market_order(symbol, side, amount)
        print(order)
        await notify_telegram(f"Test order {side} {amount} {symbol} ejecutada")
    except Exception as e:
        print(f"Error test order: {e}")
        await notify_telegram(f"Error test order: {e}")
    finally:
        await exchange.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        sym, side, amt = sys.argv[1], sys.argv[2].lower(), float(sys.argv[3])
        asyncio.run(test_order(sym, side, amt))
    else:
        asyncio.run(main())

