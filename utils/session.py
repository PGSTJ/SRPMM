
from Settings import (
    np,
    config
)

logger = config.logger


class SessionSave():
    def __init__(self, save_data:dict[str,str]):
        self.market_volume:float = save_data['market_volume']


    def load(self) -> dict[str,str]:
        """ Formats parameters for loading into a session """



class Session():
    """
    Defines Plort Market (PM) sessions, which are server like entities that represent an active plort market cycle.

    This can be over many in game days, and is most synonymous with a play session. The only reason this wouldn't 
    be accurate would be if a player restarts a PM session during a play session

    """
    def __init__(self, load_session:SessionSave=None):
        self.initial_market_volume:float = None
        self.initial_prices:dict[str,int] = {}
        self.initial_day:int = 0
        

        # self.initial_session: bool = True
        if not load_session:
            self._init_default()
        else:
            self._load_session(load_session)

    def _init_default(self):
        """ Initialize session with default parameters """
        # 50% of each max / saturation value
        self.market_volume = np.sum(config.PBS_DATA['Saturation Value'][:-1] / 2)
        



    def _load_session(self, save:SessionSave):
        """ Initialize session with previously saved parameters """
        logger.info(f'Loading previous PM Session: {save.id}') # TODO define save class with parameter/PK id
        for parameter in save:
            self.__dict__[parameter] = save[parameter]
            
        return