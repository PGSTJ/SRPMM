from Settings import (
    sl, traceback, math, np,
    config, logging
)

logger = logging.getLogger('standard')


MRKT_MIN = 364
MRKT_DEF = 1066
MRKT_MAX = 2347

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
- In other words: BP * RM * SM, where RM is the product of plort and market RNGs and SM is the market saturation
"""
# DB creation
def fill_base_stats():
    pass





# Utils





# Calculations
def calculate_price(base_price:int, plort_rng, plort_data:dict[str,str|bool|int]) -> float:
    """Calculates the price of a plort based on the base price, plort and market RNGs, and the saturation multiplier"""
    
    sm = 1+(1-(np.min(plort_data['market_volume'], plort_data['saturation_threshold'])/plort_data['saturation_threshold']))
    return base_price * plort_rng * sm
    

# days until amount and days while halved are the same function, but different perspectives -> two sides of same coin
# can likely consolidate eventually 
def days_until_amount(target_amount:int, current_amount:int) -> int|bool:
    """Generic function to calculate how many days until a certain plort amount is reached from the current amount"""
    
    if target_amount > current_amount:
        return False # TODO redirect to days while halved function... maybe
    
    # account for empty since you can't divide with 0
    if target_amount == 0:
        target_amount = np.finfo(float).eps

    ratio = target_amount / current_amount
    logger.debug(f'target: {target_amount} | ratio: {ratio}')
    return round(math.log(ratio, 0.75), 0)

def days_while_halved(selling_amount:int, saturation_threshold:int):
    """Defines the number of days a price will be halved"""
    if selling_amount < saturation_threshold:
        return False # TODO redirect to days until amount function
    
    ratio = selling_amount / saturation_threshold
    return round(math.log(ratio, 0.75), 0)


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