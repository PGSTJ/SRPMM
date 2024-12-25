from Settings import (
    sl, traceback, math,
    BP_DATA, BP_COLS
)


MRKT_MIN = 364
MRKT_DEF = 1066
MRKT_MAX = 2347

DBF = 'slimeRancher\\research.db'
conn = sl.connect(DBF, check_same_thread=False)
curs = conn.cursor()


def db_add_plort(plort_name:str, base_value: int, default_value:int, min_value: int, max_value: int, val_halflife: int):
    """adds plort base stat to table"""
    try:
        curs.execute('INSERT INTO base_stats(plort, base_val, default_val, min_val, max_val, val_halflife) VALUES (?,?,?,?,?,?)', (plort_name, base_value, default_value, min_value, max_value, val_halflife))
        conn.commit()
        return True
    except Exception as e:
        traceback.print_exc()
        return False

# def selling_affect(day:int, plort:str, sell_amt:int):
#     """Predicts affected plort value after selling a certain amount on a certain day"""
#     if plort.capitalize() not in PLORTS:
#         print('plort doesn\'t exist')
#         return False

#     half_life = [data[0] for data in curs.execute('SELECT val_halflife FROM base_stats WHERE plort=?', (plort,))]
#     halflife_percentage = sell_amt / half_life[0]
    
#     percentage_drop = halflife_percentage * 0.5
    
#     curr_val = [data[0] for data in curs.execute(f'SELECT {plort.capitalize()} FROM PMR_values WHERE Day=?', (day,))]
#     return round((1 - percentage_drop) * curr_val[0], 0)
    
    





"""
RETACKLING THE PLORT MARKET

- Price formula: basePrice*plortRNG*marketRNG*(1+(1-(min(plortsOnMarket,plortSaturationThreshold)/plortSaturationThreshold)))
- In other words: BP * RM * SM, where RM is the product of plort and market RNGs and SM is
"""
# DB creation
def fill_base_stats():
    pass





# Utils





# Calculations
def days_until_amount(target_amount:int, current_amount:int) -> int|bool:
    """Generic function to calculate how many days until a certain plort amount is reached from the current amount"""
    if target_amount < current_amount:
        ratio = target_amount / current_amount
        return round(math.log(ratio, 0.75), 0)
    return False # TODO redirect to days while halved function

def days_while_halved(selling_amount:int, saturation_threshold:int):
    """Defines the number of days a price will be halved"""
    if selling_amount > saturation_threshold:
        ratio = selling_amount / saturation_threshold
        return round(math.log(ratio, 0.75), 0)
    return False # TODO redict to days until amount function


def market_decay(current_volume:int, days:int=1):
    """Base market decay after other multipliers after 1 to x days"""
    return 0.75^days * current_volume


def sale_effect(selling_amount:int, saturation_threshold:int, current_price:int) -> int:
    """Calculates effect on base price based on sale amount"""
    ratio = selling_amount / saturation_threshold
    return round((1-ratio) * current_price, 0)

def saturation_multiplier(current_amount:int, saturation_limit:int) -> float:
    return 1+(1-(min(current_amount,saturation_limit)/saturation_limit))

def calculate_random_modifier(posted_price:int, base_price:int, saturation_multiplier:float):
    """Finds the RM """
    return posted_price / (base_price * saturation_multiplier)

if __name__ == '__main__':
    d = days_until_amount(10, )