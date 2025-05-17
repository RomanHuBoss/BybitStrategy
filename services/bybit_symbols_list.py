import datetime
from fastapi import FastAPI, HTTPException, status
from pybit.unified_trading import HTTP
import time

class BybitSymbolsList:
    """
    получает и кэширует символы ByBit
    https://api-testnet.bybit.com/v5/market/instruments-info/?category=linear
    """
    def __init__(self):
        self.symbols_list = []
        self.last_update = None

    def get_bybit_symbols_list(self):
        """
        Получить информацию о символах ByBit
        """
        # Валидация параметров (бизнес-логика)

        try:
            session = HTTP(testnet=False)

            bybit_data = session.get_instruments_info(
                category="linear",
                limit=1000,
            )

            if self.last_update is None or self.last_update > 1000 * 60:
                self.symbols_list = []
                for symbol_data in bybit_data['result']['list']:
                    self.symbols_list.append(symbol_data['symbol'])

                self.last_update = int(time.time())

            return self.symbols_list

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Не удалось получить данные с биржи Bybit",
                    "details": e.message if hasattr(e, 'message') else e
                }
            )

if __name__ == "__main__":
    symbols = BybitSymbolsList()
    print(symbols.get_bybit_symbols_list())