from fastapi import FastAPI, HTTPException, status
from pybit.unified_trading import HTTP
import time

def get_bybit_candles(symbol: str, timeframe: int, candles_num: int):
    """
    Получить свечи для указанного символа.

    - **symbol**: Торговый символ (например: BTCUSDT)
    - **timeframe**: Таймфрейм (в минутах). Допустимы вот эти 1, 3, 5, 15, 30, 60, 120, 240, 360, 720
    - **candles_num**: Количество запрашиваемых свечей
    """
    # Валидация параметров (бизнес-логика)

    if timeframe not in [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Некорректный запрос таймфрейма",
                "details": "Допустимы следующие значения в минутах: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720"
            }
        )

    if candles_num <= 0 or candles_num > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Некорректный запрос числа свечей",
                "details": "Допустимы значения в интервале от 1 до 1000"
            }
        )

    try:
        session = HTTP(testnet=False)

        bybit_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            start=round(time.time() * 1000) - timeframe * candles_num * 60 * 1000,
            end=round(time.time() * 1000),
        )
        bybit_data['candles_num'] = candles_num
        bybit_data['timeframe'] = timeframe

        return bybit_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Не удалось получить данные с биржи Bybit",
                "details": e.message if hasattr(e, 'message') else e
            }
        )