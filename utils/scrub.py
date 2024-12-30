from bs4 import BeautifulSoup as bs
import sqlite3 as sl
import traceback
import pandas as pd





def optimal_sale_all(current_day:int, day_range:int, db):
    """Finds optimal sell dates for all plorts"""

    for pred in [optimal_sale_single(plort, current_day, day_range, db) for plort in db.PLORTS]:
        msg = _format_prediction(pred['plort'], pred['day'], pred['value'], pred['stnd'])
        print(msg)

def _format_prediction(plort_type:str, day:int, value:int, stnd:str) -> str:
    """Formats str repr of optimal sell days"""
    return f'\t{plort_type}\nDay: {day}\nSell for: {value} ({stnd} of best)\n'

def optimal_sale_single(plort_type:str, current_day:int, range:int, db) -> dict:
    """ 
    Finds the optimal day to sell a specified plort within a given range.

    For example, to find the best day to sell Pink plorts within the next 30 days, starting from day 50:

        ```
        optimal_sale(50, 30, 'Pink', database)
        ```
    Which outputs:

        ```
        >>>     'Pink'
        >>> 'Day: 65'
        >>> 'Sell for: 18 (88% of best)'
        ``` 
    """
    # extracts raw and standardized values from DB, each day tupled within the overall list
    df = pd.read_sql(f'SELECT Day, {plort_type.capitalize()} FROM PMR_Value WHERE day BETWEEN {current_day} AND {current_day + range}', db.conn)

    mi = df[plort_type].idxmax()
    day, max = df.loc[mi, 'Day'], df.loc[mi, plort_type]

    stdzd = [info[0] for info in db.curs.execute(f'SELECT {plort_type.capitalize()} FROM PMR_Stndrzd WHERE day=?', (int(day),))][0]

    return {'plort':plort_type, 'day':day, 'value':max, 'stnd':stdzd}

def _possible_adjacency():
    """Discovers potential competitive prices up to 25% day range on either end; significance determined by over 10%"""

    



    