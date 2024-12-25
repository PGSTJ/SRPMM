from bs4 import BeautifulSoup as bs
import sqlite3 as sl
import traceback
import pandas as pd


def create(file): 

    with open(file, 'r') as fn:
        soup = bs(fn, 'html.parser')

    return soup

def format(data):
    """extract raw daily data, each list index is a new day - should be 999 items"""
    unformatted = [element.contents for element in data if element['class'][0] == 'day']


    # remove spaces within each listed day
    formatted = [[items for items in contents if items != '\n'] for contents in unformatted]

    return formatted


def extract_values(soup:bs) -> list | bool:
    """Plain sell price per plort and market price as percent"""
    data = _remove_header(soup)

    # similar functionality to formatted variable within format() function - day_list is a list 
    # that is appended to overall all_raw_values list
    try:
        all_raw_values = []
        for day in data:
            day_list = []
            for i in day.contents:
                # try/except to ignore spaces/newlines -> would throw an error thats being ignored
                try:
                    day_list.append(_values_parser(i.contents))
                except:
                    pass
            # convert to tuple for sql table insertion compatability
            all_raw_values.append(tuple(day_list))

        return all_raw_values
    except:
        return False
    
def extract_std_dev(soup:bs) -> list | bool:
    """Extracts standard deviations of each plain value"""
    data = soup.find_all('td')

    plain_all_stdevs = []

    for column in data:
        title = column['title']
        identifier = title.index('%')
        plain_all_stdevs.append(title[identifier-2:identifier+1])
    

        
    day = 1
    temp = (day,)
    final = []

    for num in plain_all_stdevs:
        temp += (num,)
        if len(temp) == 19:
            day += 1
            final.append(temp)
            temp = (day,)

            

    return final

def _values_parser(contents):
    """parses html for raw values"""
    match len(contents):
        case 1:
            return contents[0]
        case 3:
            return contents[2].strip()

def _remove_header(soup:bs) -> list:
    """removes header from list for easier iteration"""
    data = soup.find_all('tr')
    data.pop(0)
    return data


class Database():
    """
    

    Abbreviations:
        PMR = Plort Market Rates
    """
    ACTIVE = False
    curs = None
    conn = None

    DBN = 'ranchin'
    DBF = f'slimeRancher\\{DBN}.db'


    TABLES = []
    PLORTS = [
        'Pink',
        'Rock',
        'Phosphor',
        'Tabby',
        'Rad',
        'Honey',
        'Boom',
        'Puddle',
        'Fire',
        'Crystal',
        'Quantum',
        'Dervish',
        'Hunter',
        'Mosaic',
        'Tangle',
        'Saber',
        'Gold'
    ]

    # NOTE: if multiple tables - should consider more general create table function and self.table_name attribute

    def __init__(self) -> None:
        if self.ACTIVE is False:
            self.conn = sl.connect(self.DBF, check_same_thread=False)
            self.curs = self.conn.cursor()

            self.ACTIVE = True

    def disconnect(self):
        self.conn.close()
        self.ACTIVE = False
        self.conn = None
        self.curs = None

        print(f'Disconnected from the {self.DBN} database.')

    def _active_check(self) -> bool:
        """Checks active status of DB"""
        if self.ACTIVE and self.curs is None:
            print('DB never intialized. Cursor not set.')
            return False
        elif not self.ACTIVE:
            print('DB not active')
            return False
        elif self.ACTIVE:
            return True


    def db_switch(self, activate:bool) -> bool:
        """Toggles DB on/off for access"""
        if activate:
            self.ACTIVE = True
            return True
        elif not activate:
            self.ACTIVE = False
            return True
        else:
            return False
    
    def plort_ids(self):
        """Creates reference table of plort IDs"""
        try:
            self.curs.execute('CREATE TABLE IF NOT EXISTS plorts(id VARCHAR(3) PRIMARY KEY, plort_type VARCHAR(10))')
            self.curs.execute('INSERT INTO plorts(id, plort_type) VALUES (?,?)', ('M0', 'Market'))
            id = 1
            for plort in self.PLORTS:
                pid = 'P' + str(id)
                self.curs.execute('INSERT INTO plorts(id, plort_type) VALUES (?,?)', (pid, plort))
                id += 1
            self.conn.commit()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
        
    def _reset_plorts(self):
        """recreates plort id table"""
        try:
            self.curs.execute('DROP TABLE plorts')
            self.plort_ids()
        except Exception as e:
            traceback.print_exc()
        

    def _create_table(self, table_name:str) -> bool:
        """Creates table with headers being Days and Plorts"""

        if not self._active_check():
            return False

        try:
            hdr_frmt = self._format_header(create=True)
            self.TABLES.append(table_name)

            self.curs.execute(f'CREATE TABLE IF NOT EXISTS {table_name}({hdr_frmt})')
            return True
        except:
            return False
        
    def _format_header(self, create=False, insert=False) -> list | bool:
        """formats DB table header; choose setting between creating or inserting into a table"""
        if create:
            plort_format = [name + ' INT' for name in self.PLORTS]

            header = ['Day INT PRIMARY KEY', 'Market VARCHAR(6)'] + plort_format
            return ', '.join(header)
        elif insert:
            header = ['Day', 'Market'] + self.PLORTS
            return ', '.join(header)
        elif not insert and create:
            print('Must select setting: create or insert')
            return False

    def _reset_table(self, table_name:str) -> bool:
        """Clears table of entries"""
        if not self._active_check():
            return False

        self.curs.execute(f'DROP TABLE {table_name}')
        if self._create_table(table_name):
            return True
        else:
            return False
        
    def generate_table(self, table_name:str, data:list) -> bool:
        """Uploads raw price values into DB table; takes output of extract_values() function"""
        header = self._format_header(insert=True)
        # has extra comma at end, ignore by index
        placeholders = '?,'*len(header.split(', '))
        
        try:
            for valuesXday in data: 
                self.curs.execute(f'INSERT INTO {table_name}({header}) VALUES ({placeholders[:-1]})', valuesXday)
            self.conn.commit()
            return True
        except:
            return False





def optimal_sale_all(current_day:int, day_range:int, db:Database):
    """Finds optimal sell dates for all plorts"""

    for pred in [optimal_sale_single(plort, current_day, day_range, db) for plort in db.PLORTS]:
        msg = _format_prediction(pred['plort'], pred['day'], pred['value'], pred['stnd'])
        print(msg)

def _format_prediction(plort_type:str, day:int, value:int, stnd:str) -> str:
    """Formats str repr of optimal sell days"""
    return f'\t{plort_type}\nDay: {day}\nSell for: {value} ({stnd} of best)\n'

def optimal_sale_single(plort_type:str, current_day:int, range:int, db:Database) -> dict:
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

    



if __name__ == '__main__':
    fn = 'slimeRancher\\plort_market.html'
    file = create(fn)
    db = Database()

    # v = extract_values(file)
    # z = extract_std_dev(file)

    # db.generate_table('PMR_values', v)
    # db.generate_table('PMR_relational', z)
    

    optimal_sale_all(50, 50, db)

    
"""
Creation order:
create soup w creat(filename)
create db instance
    create 2 tables for html data - plain values and relational data
extract data with respective functions
generate table with db


"""
        

    
    



    