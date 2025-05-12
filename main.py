import uvicorn
from pybit.unified_trading import HTTP
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from services.get_bybit_candles import get_bybit_candles
from services.bybit_candles_handlers import bybit_candles_to_csv

"""
Эта балалайка запрашивает с биржи Bybit данные о торговле конкретной криптовалютной парой с текущего момента времени и до момента в прошлом, определяемого
Как это работает:   
    http://127.0.0.1:8000/prognosis/?symbol=BTCUSDT&candles_num=5
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

app = FastAPI()

# Модель для успешного ответа
class PrognosisResponse(BaseModel):
    symbol: str
    timeframe: int
    candles_num: int
    prognosis: list  # данные прогноза

# Модель для ошибки в бизнес-логике
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Единственный разрешенный эндпоинт
@app.get("/prognosis/", responses={
    200: {"model": PrognosisResponse},
    400: {"model": ErrorResponse},
    404: {"description": "Not found"},
})

async def get_prognosis(symbol: str, timeframe: int, candles_num: int, save_to_csv: Optional[bool] = False):
    bybit_data = get_bybit_candles(symbol, timeframe, candles_num)

    if save_to_csv:
        bybit_candles_to_csv(bybit_data)

    return bybit_data

# Обработка всех других путей с 404 ошибкой
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def handle_unsupported_paths(path: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Некорректный запрос. Используй /prognosis/"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


