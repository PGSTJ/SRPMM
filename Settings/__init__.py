import matplotlib as plt
import numpy as np
import sqlite3 as sl
import traceback
import os
import math
import pathlib
from typing import Literal
import csv
import json
import logging
import logging.config


import pandas as pd
from bs4 import BeautifulSoup as bs


BASE_DIR = pathlib.Path(__file__).parent.parent

def process_base_plort_data():
    def _format_header(names:list[str]):
        initial = [name.lower().split(' ') for name in names]
        return [('_').join(i) for i in initial]
    
    with open(BASE_DIR / 'plort_base_stats.csv', 'r') as fn:
        header:list[str] = _format_header(fn.readline().strip().split(','))
        rdr = csv.DictReader(fn, header)
        return header, [i for i in rdr]
    
BP_COLS, BP_DATA = process_base_plort_data()
ALL_PLORTS = [data['plort_name'] for data in BP_DATA]


LOGGING_FILE = BASE_DIR / 'logs.log'
LOGGING_CONFIG = {
    'version': 1,
    'disabled_existing_loggers': False,
    'formatters': {
        'generic': {
            'format': '%(asctime)s|%(levelname)-8s(%(filename)s)| %(message)s'
        },
        'error': {
            'format': '\n%(asctime)s|%(levelname)-8s| %(message)s\n%(pathname)s'
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': LOGGING_FILE,
            'mode': 'w',
            'formatter': 'generic'
        },
        'console': {
            'level': 'WARNING',
            'class': 'logging.StreamHandler',
            'formatter': 'error'
        }
    },
    'loggers': {
        'standard': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


logging.config.dictConfig(LOGGING_CONFIG)


if __name__ == "__main__":
    print(ALL_PLORTS)