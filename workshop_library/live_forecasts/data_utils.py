from .. import data_manager
import os
from .. import settings
import datetime

times = {
    'SPY': datetime.date(1994, 1, 1),
    'GLD': datetime.date(1990, 1, 1),
    'VXX': datetime.date(2000, 1, 1),
    'XIV': datetime.date(2000, 1, 1),
    '^VIX': datetime.date(1980, 1, 1),
    'EURUSD=x': datetime.date(1994, 1, 1),
    '^STOXX50E': datetime.date(1990, 1, 1),
    '^N225': datetime.date(1990, 1, 1)
}


def refresh_data_for_symbols():
    symbols = times.keys()
    for symbol in symbols:
        print('loading %s' % symbol)
        data_manager.get_past_5y_of_data(symbol)
        data_manager.add_ti_and_store(symbol)


def initialize_data_for_symbols():
    if os.path.isfile(settings.storage_path):
        os.remove(settings.storage_path)
    for key in times:
        data_manager.load_from_store_or_yahoo(start=times[key], end=datetime.date.today(), symbol=key)
        data_manager.add_ti_and_store(key)
        print(key)
