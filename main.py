import os
import uvicorn
import json
from pybit.unified_trading import HTTP
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_csv, bybit_candles_to_df
from services.bybit_symbols_list import BybitSymbolsList
from predictor import CryptoModelPredictor
import time

"""
Эта балалайка запрашивает с биржи Bybit данные о торговле конкретной криптовалютной парой с текущего момента времени и до момента в прошлом, определяемого
Как это работает:   
    http://127.0.0.1:8000/prognosis/?symbol=BTCUSDT&timeframe=3&candles_num=200
    Единственный эндпоинт /prognosis/ принимает:
        symbol (строка) - торговый символ
        timeframe - таймфрейм (1, 5, 15 минут и т.д.)
        candles_num (целое число) - количество свечей
        save_to_csv (boolean) - надо ли скачанную инфу сохранить в csv-файле в папке downloads
    Валидация параметров:
        Если параметры не соответствуют бизнес-правилам (например, слишком длинный символ), возвращается ошибка 400 с деталями
    Успешный ответ:
        При корректных параметрах возвращается JSON с рекомендациями по входу в сделки
    Обработка всех других путей:
        Любые другие URL возвращают 404 ошибку
        Обрабатываются все возможные HTTP методы
    Документация:
        Автоматически генерируется Swagger UI с описанием эндпоинта (http://127.0.0.1:8000/docs)
        Указаны возможные коды ответов (200, 400, 404)
"""

model_folder = os.path.join("models", "3m-60forward-10backward-17-05-2025 22-17-14")

app = FastAPI()
bybit_symbols = BybitSymbolsList()
predictor = CryptoModelPredictor(model_folder=model_folder, threshold=0.6)
prognosis_cache = {}


# Модель для успешного ответа
class PrognosisResponse(BaseModel):
    timeframe: int
    candles_num: int
    prognosis: list # данные прогноза

# Модель для успешного ответа
class CandlesResponse(BaseModel):
    symbol: str
    timeframe: int
    candles_num: int
    candles_data: list  # данные свечей

# Модель для ошибки в бизнес-логике
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# эндпоинт
@app.get("/prognosis/", responses={
    200: {"model": PrognosisResponse},
    400: {"model": ErrorResponse},
    404: {"description": "Not found"},
})
async def get_prognosis(timeframe: int, candles_num: int):
    """
    prognosis_cache = {
        symbol_name: {
            last_update: value,
            prognosis: json
        }
    }
    """

    symbols_list = bybit_symbols.get_bybit_symbols_list(limit=1000)
    for symbol in symbols_list:
        last_update = None
        if symbol in prognosis_cache:
            last_update = prognosis_cache[symbol]['last_update'] if symbol in prognosis_cache and 'last_update' in prognosis_cache[symbol] else None

        if last_update is None or last_update < int(time.time()) - 60:
            df = bybit_candles_to_df(await get_candles(symbol, timeframe, candles_num))
            symbol_prognosis = predictor.run_prediction_pipeline(df)
            prognosis_cache[symbol] = {
                'last_update': int(time.time()),
                'symbol_prognosis': symbol_prognosis,
            }

    return prognosis_cache

# эндпоинт
@app.get("/candles/", responses={
    200: {"model": CandlesResponse},
    400: {"model": ErrorResponse},
    404: {"description": "Not found"},
})
async def get_candles(symbol: str, timeframe: int, candles_num: int, save_to_csv: Optional[bool] = False):
    candles_data = get_bybit_candles(symbol, timeframe, candles_num)

    if save_to_csv:
        bybit_candles_to_csv(candles_data)

    return candles_data


# Обработка всех других путей с 404 ошибкой
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def handle_unsupported_paths(path: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Некорректный запрос. Используй /candles/ или /prognosis/"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


