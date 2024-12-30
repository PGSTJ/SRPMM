from . import (
    REF_DATA_DIR, WEB_DATA_DIR,
    logging, pd, bs, Literal, bse, pl, datetime as dt, plt, math,
    os

)

logger = logging.getLogger('standard')

DEFAULT_PBS_NAME = 'plort_base_stats'
PBS_DATA = pd.read_csv(REF_DATA_DIR/ f'{DEFAULT_PBS_NAME}.csv')
# DEFAULT_PICKLE_SAVE = 

CURRENT_WEB_DATA_SCRAPED = WEB_DATA_DIR / 'current_research.html'

HTML_PREDICTIONS_HEADER = ['Day', 'Market'] + PBS_DATA['Plort Name'].tolist()




class PreProcessingError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class PreProcessor():
    """ Processes prediction data from HTML and converts to a dataframe
    
    """

    def __init__(
            self, 
            html_filepath:str = CURRENT_WEB_DATA_SCRAPED,
            pickle_filename:str = 'extracted_game_data',
            csv_directory:str=REF_DATA_DIR
            ):
        self.html_file = html_filepath
        self.csv_directory = csv_directory

        self.soup = self.create_soup(html_filepath)
        self.results:dict[Literal['plain', 'stdev', 'recsales'], pd.DataFrame] = {
            'plain': '',
            'stdev': '',
            'recsales': ''
        }
        
        self.pickle_version:str = dt.datetime.now().strftime('%d%m%Y')
        self.pickle_file = REF_DATA_DIR / f'{pickle_filename}_{self.pickle_version}.pkl'

    @staticmethod
    def create_soup(filepath:str): 
        """ soupifies a file """
        with open(filepath, 'r') as fn:
            soup = bs(fn, 'html.parser')

        return soup

    def _extract_all_plain_values(self) -> list[tuple[str]]:
        """Plain sell price per plort and market price as percent of all days"""
        data = self._remove_header()

        # similar functionality to formatted variable within format() function - day_list is a list 
        # that is appended to overall all_raw_values list
        temp = []
        for day in data:
            day_list = []

            tags_only = [i for i in day.children if isinstance(i, bse.Tag)]

            for i in day.contents:
                # try/except to ignore spaces/newlines -> would throw an error thats being ignored
                try:
                    value = self._values_parser(i.contents)
                    if '%' in value:
                        value = value[:-1]

                    day_list.append(float(value))
                except AttributeError:
                    pass
            # convert to tuple for sql table insertion compatability
            temp.append(dict(zip(HTML_PREDICTIONS_HEADER, day_list)))
        
        self.results['plain'] = pd.DataFrame(temp)
        logger.info(f'Finished extrating plain values from {self.html_file}')

        return 

            
    def _extract_all_std_dev(self) -> list[tuple[str]]:
        """Extracts standard deviations of each plain value"""
        data = self.soup.find_all('td')

        # straight list of every single value as a single row vector
        plain_all_stdevs = []

        for column in data:
            title = column['title']
            identifier = title.index('%')
            plain_all_stdevs.append(float(title[identifier-2:identifier]) / 100)

        # enumeration to attach row since initial loop wasn't consecutive
        w_row = enumerate([tuple(plain_all_stdevs[i:i+18]) for i in range(0, len(plain_all_stdevs), 18)])
        
        # format data as dict with header
        hdr_formatted = [dict(zip(HTML_PREDICTIONS_HEADER, [row+1] + list(data))) for row,data in w_row]
        
        self.results['stdev'] = pd.DataFrame(hdr_formatted)
        logger.info(f'Finished extrating std values from {self.html_file}')
        return

    def _extract_rec_sales(self):
        """ Extracts recommended sales from HTML (where sale value is highlighted) """
        data = self._remove_header()

        temp = []
        for day in data:
            f = [i for i in day.children if isinstance(i, bse.Tag)]
            sale_prefs = [col for col in f if 'class' in col.attrs]
            if len(sale_prefs) > 0:
                for data in sale_prefs:
                    pref_sale_price = int(self._values_parser(data.contents))
                    temp.append(self._extract_sale_preference_metadata(data.attrs['title'], pref_sale_price)) 

        self.results['recsales'] = pd.DataFrame(temp)
        self._add_std_recsales()

        logger.info(f'Finished extrating recommended sale metadata from {self.html_file}')
        return



    @staticmethod        
    def _values_parser(contents:list[str]):
        """parses html for raw values"""
        match len(contents):
            case 1:
                return contents[0]
            case 3:
                return contents[2].strip()

    def _remove_header(self) -> list:
        """removes header from list for easier iteration"""
        data = self.soup.find_all('tr')
        data.pop(0)
        return data
    
    @staticmethod
    def _extract_sale_preference_metadata(title:str, price_value:int) -> dict[str,str]:
        """ Extracts plort name and value at the preferred day """
        hyphen = title.index('-')
        
        plort_start = title[hyphen+2:]
        plort_end = plort_start.index(' ')
        plort_name = plort_start[:plort_end]

        day_end = title[:hyphen]
        day_start = day_end.index(' ')
        day = day_end[day_start:]

        return {
            'plort': plort_name,
            'day': int(day.strip()),
            'value': price_value
        }
    
    def _add_std_recsales(self):
        """ Adds associated StdDev to the recommended sales DF """
        std_df = self.results['stdev'].copy()
        rs_dct:dict[int, dict[str,str|int]] = self.results['recsales'].copy().to_dict(orient='index')

        # extracts stdev values at specified day and plort
        add_std:list[float] = [std_df.loc[std_df['Day'] == int(rs_dct[idx]['day']), rs_dct[idx]['plort']].to_list()[0] for idx in rs_dct]
        
        assert len(rs_dct) == len(add_std), f'Length mismatch - # of recommended sales: {len(rs_dct)} | # of extracted stdevs: {len(add_std)}'

        self.results['recsales'].insert(len(self.results['recsales'].columns), 'stdev', add_std)
        
        # self.results['recsales'].sort_values(by=['plort', 'day']) # for export only
        # pp.csv_export(['recsales'])
        return
    

    def process_html_predictions(self):
        """ Extracts all plain values and standardized values from the HTML """
        self._extract_all_plain_values()
        self._extract_all_std_dev()
        self._extract_rec_sales() # must happen last
        logger.info(f'Finished processing predictions from {self.html_file}')

        if len(self.results['plain']) != len(self.results['stdev']):
            raise PreProcessingError(f'Length mismatch between plain ({len(self.results['plain'])}) and standardized ({len(self.results['stdev'])}) collections')
        
        print('done processing')
        return
        
    def pickle_save(self):
        """ Saves extracted HTML data to pickle file """
        with open(self.pickle_file, 'wb') as fn:
            pl.dump(self.results, fn)
            logger.info(f'Saved procesing results to pickle file: {self.pickle_file}')
        print(f'Saved results to {self.pickle_file}')
        return

    def pickle_load(self, save_path:str=None):
        """ Loads current pickle save, or searches for past saves to load.

        If a save path is specified here, rather than PreProcessor instantiation
        then must include '.pkl' file extension. 
          
        """
        if save_path is None:
            save_path = self.pickle_file


        if not os.path.isfile(save_path):
            # look for older save
            addtn_saves = [file for file in REF_DATA_DIR.glob('*.pkl')]
            
            # no saves found
            if len(addtn_saves) == 0:
                logger.error(f'Attempted to load pickle data from non-existant file')
                raise FileNotFoundError(f'The pickle file: {save_path} does not exist. ')
            
            # multiple saves found
            assert len(addtn_saves) == 1, f'Multiple prior pickle saves detected. Please specify a save_path from {addtn_saves}.'
            
            save_path = addtn_saves[0]

        with open(save_path, 'rb') as fn:
            return pl.load(fn)

    def csv_export(self, result_type:list[Literal['plain', 'stdev', 'recsales']]=None):
        """ Exports specified results to CSV """
        
        if result_type is None:
            queue = self.results.copy()
        else:
            queue = {result:self.results[result] for result in self.results if result in result_type}
        
        for result in queue:
            queue[result].to_csv(f'{self.csv_directory}/{result}.csv', header=True)

        print(f'done exporting CSVs to {self.csv_directory}')
        return




# TODO would like to generalize visualization eventually
# need to figure out what visualizations I would want
# currently have day x value/stdev per plort
def plot_rec_data(rec_df:pd.DataFrame, plorts:list[str]=None):
    # rec_df['day'].astype(int)
    if plorts is None:
        plort_groups = rec_df.copy().groupby('plort')
        plort_group_df:list[dict[str,pd.DataFrame]] = [{'plort':groups[0], 'df':groups[1].sort_values(by='day')} for groups in plort_groups]

    else:
        plort_group_df = [{'plort': plort, 'df':rec_df[rec_df['plort'] == plort.capitalize()].sort_values(by='day')} for plort in plorts]
        # plort_group_df = rec_df[rec_df['plort'] == plort.capitalize()]
        # print('else', plort_group_df)
    
    # return plort_group_df TODO consider dtype verification/assertion
        
    num_plots = len(plort_group_df)

    # Set up the grid for subplots
    cols = math.ceil(num_plots / 6)  # Number of columns
    rows = math.ceil(num_plots / cols)  # Number of rows
    fig, axes = plt.subplots(rows, cols, constrained_layout=True)
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Plot each group in its own subplot
    for idx, group_data in enumerate(plort_group_df):
        ax = axes[idx]
        group_df = group_data['df']
        plort_name = group_data['plort']
        
        # plot value and stdev
        ax.plot(group_df['day'], group_df['value'], marker='o', linestyle='-', label='Value', color='blue')
        ax.plot(group_df['day'], group_df['stdev'], marker='s', linestyle='--', label='Stdev', color='orange')

        # titles and labels
        ax.set_title(plort_name)
        ax.set_xlabel('Day')
        ax.set_ylabel('Value / Stdev')
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if there are any
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Show the plot
    plt.suptitle("Trends by Plort", fontsize=16)
    plt.show()

