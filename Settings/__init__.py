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
import random
import pickle as pl
import datetime
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs, element as bse
import seaborn


BASE_DIR = pathlib.Path(__file__).parent.parent

# modules
UTILS = BASE_DIR / 'utils'
MODEL = BASE_DIR / 'model'

# datasets
REF_DATA_DIR = BASE_DIR / 'reference_data'
WEB_DATA_DIR = BASE_DIR / 'web_data'



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

