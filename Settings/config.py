from . import (
    REF_DATA_DIR, WEB_DATA_DIR,
    logging, pd, bs, Literal, bse, pl, datetime as dt, plt, math,
    os, random, dataclass, field

)

logger = logging.getLogger('standard')

DEFAULT_PBS_NAME = 'plort_base_stats'
PBS_DATA = pd.read_csv(REF_DATA_DIR/ f'{DEFAULT_PBS_NAME}.csv')
# DEFAULT_PICKLE_SAVE = 

CURRENT_WEB_DATA_SCRAPED = WEB_DATA_DIR / 'current_research_2.html'

HTML_PREDICTIONS_HEADER = ['Day', 'Market'] + PBS_DATA['Plort Name'].tolist()


class HtmlProcessor():
    """ Processes prediction data from HTML and converts to a dataframe

    Parameters
    ----------
        html_filepath : str, default=CURRENT_WEB_DATA_SCRAPED
            Path to html file with exported "market price predictor" data. Default is for 
            debug/development purposes.

        html_parser : Literal['html', 'lxml-html', 'lxml-xml', 'html5'], default='html'
            Parsers supplied to beautifulsoup object, as defined by their API. Default is
            the generic html parser in Python's standard library. 
            
        pickle_filename : str, default='extracted_game_data'
            Name of save file of extracted HTML data as Dataframe objects

        export_directory : str, default=REF_DATA_DIR
            Destination of csv output for the csv_export() method 

    Args
    ----
        rrv_k : int, default=10
            The number of rows randomly chosen for row verification
    
    """

    def __init__(
            self, 
            html_filepath:str = CURRENT_WEB_DATA_SCRAPED,
            html_parser:Literal['html', 'lxml-html', 'lxml-xml', 'html5']='html',
            pickle_filename:str = 'extracted_game_data',
            export_directory:str=REF_DATA_DIR,
            *,
            rrv_k:int=10
            ):
        self.html_file = html_filepath
        self.html_parser = html_parser
        self.export_directory = export_directory

        self.soup = self.create_soup(html_filepath, html_parser)
        
        self.validation_metadata:dict[Literal['RRV Days', 'Completion Time', 'Training Size' ,'Feature Size', 'Feature Names'], any] = {'RRV Days': []} # RRV = Random Row Validation - the days/rows randomly chosen for length validation
        self.validate_html_structure(rrv_k=rrv_k)
            
        self.results:dict[Literal['plain', 'perc best', 'recsales'], pd.DataFrame|list[dict[str,int|str]]] = {
            'plain': [],
            'perc best': [],
            'recsales': []
        }
        
        self.pickle_version:str = dt.datetime.now().strftime('%d%m%Y')
        self.pickle_file = self.export_directory / f'{pickle_filename}_{self.pickle_version}.pkl'

    @staticmethod
    def create_soup(filepath:str, parser:Literal['html', 'lxml-html', 'lxml-xml', 'html5']) -> bs: 
        """ soupifies a file """
        bs4_parsers = {
            'html': 'html.parser',
            'lxml-html': 'lxml',
            'lxml-xml': 'xml',
            'html5': 'html5lib'
        }

        with open(filepath, 'r') as fn:
            soup = bs(fn, bs4_parsers[parser])

        return soup

    def validate_html_structure(self, rrv_k:int):
        """ Ensures new uploads are properly formatted """
        validate_start = dt.datetime.now()

        all_rows = self.soup.find_all('tr')
        num_rows = len(all_rows)
        all_rows[0]['class'] = 'header'
        
        header = self.soup.find('tr')
        header_cols = self._extract_column_names(header, type='header')
        num_features = len(header_cols)
        missing_cols = [hdr for hdr in HTML_PREDICTIONS_HEADER if hdr not in header_cols]

        random_rows:list[bse.Tag] = random.choices(self._remove_header(), k=rrv_k)


        assert num_rows == 1000, f'Incorrect number of row (tr tags) in the HTML file. Expected: 1000 | Got: {num_rows}'
        assert num_features == 19, f'Supplied HTML file has an incorrect amount of features. Expected 19 but got {num_features}: {header_cols}'
        assert len(missing_cols) == 0, f'Supplied HTML file is missing columns: {missing_cols}'
        for row in random_rows:
            row_tags = self._get_tags_in_row(row)
            random_row_length = len(row_tags)
            assert random_row_length == num_features, f'Randomly selected row has a different number of columns. Expected {num_features} | Got: {random_row_length}'
            self.validation_metadata['RRV Days'].append(int(row_tags[0].string))
        
        validate_finish = dt.datetime.now()
        valididate_total = validate_finish - validate_start

        self.validation_metadata['Completion Time'] = valididate_total.microseconds
        self.validation_metadata['Training Size'] = num_rows
        self.validation_metadata['Feature Size'] = num_features
        self.validation_metadata['Feature Names'] = header_cols
        
        return

    @staticmethod
    def _get_tags_in_row(row_tag:bse.Tag) ->list[bse.Tag]:
        """ Returns list of all children tags in the given row tag """
        return [i for i in row_tag.children if isinstance(i, bse.Tag)]
    
    def _extract_column_names(self, tag:bse.Tag, type:Literal['header', 'content']):
        """ Extracts column names from given row tag
        
        Parameters
        ----------
            type : Literal['header', 'content']
                'header' parsing doesnt require 
        
        """
        row_children_tags = self._get_tags_in_row(tag)
        
        if type == 'header':
            cols = [i.text.strip() for i in row_children_tags if i.text.strip() != '']
            plorts = [i['title'] for i in row_children_tags[2:]]
            logger.debug(f'cols: {cols} | plorts: {plorts}')
            col_names = cols + plorts
        elif type == 'content':
            col_names = [int(row_children_tags[0].string)]
            for tag in row_children_tags[1:]:
                title_list = tag['title'].split(' ')
                identifier = title_list.index('-') # target word (market or plort name) follows this index
                col_names.append(title_list[identifier+1].capitalize())
        
        return col_names

    def process_html_predictions(self, overwrite_html_file:bool=False):
        """ Extracts all plort market data from the HTML

        Parameters
        ----------
            overrwrite_html_file : bool, default=False
                If true, will essentially add row tag class attributes for the header the days, 
                but does overwrite the entire file (so additional things can be changed if need be).
        
        """
        logger.info(f'Parser: {self.html_parser} | File: {self.html_file}\n')

        for row in self._remove_header():
            self._process_row(row)

        for extracted_data in self.results:
            data = self.results[extracted_data]
            self.results[extracted_data] = pd.DataFrame(data)
        logger.info('Converted all extracted HTML data into dataframes')
        
        assert self.results['plain'].shape and self.results['perc best'].shape == (999,19), f'Incorrectly stitched dataframe(s). Expected shape (999,19) but got shapes plain: {self.results['plain'].shape} perc best: {self.results['perc best'].shape}'
        
        print('done processing')
        logger.info(f'Finished processing HTML data from {self.html_file}')

        if overwrite_html_file:
            with open(self.html_file, 'wb') as fn:
                fn.write(self.soup.prettify('utf-8'))
            logger.info(f'Added row tag classes to HTML : {self.html_file}')
           

        return

    def _process_row(self, row_tag:bse.Tag):
        """ Extracts plort market data from the row """
        row_tag['class'] = 'day'
        row_children_tags = self._get_tags_in_row(row_tag)
        
        day = int(row_children_tags[0].string)
        pvs = [day]
        pbvs = [day]

        # day data already extracted - only look in market/plort columns 
        for cell_tag in row_children_tags[1:]:
            title_list = cell_tag['title'].split(' ')

            col_name = title_list[title_list.index('-')+1].capitalize()
            col_value = [i.strip() for i in cell_tag.contents if not isinstance(i, bse.Tag)][-1]
            col_perc_best = float([i for i in title_list if '%' in i][0][:-1]) / 100
            
            pvs.append(col_value)
            pbvs.append(col_perc_best)

            if 'class' in cell_tag.attrs:
                self.results['recsales'].append({
                    'Plort': col_name,
                    'Day': day,
                    'Value': col_value,
                    'Percentage Best': col_perc_best
                })
            
             
        self.results['plain'].append(dict(zip(self.validation_metadata['Feature Names'], pvs)))
        self.results['perc best'].append(dict(zip(self.validation_metadata['Feature Names'], pbvs)))

        # logger.debug(f'Day: {day} | PVS: {pvs} | PBVS: {pbvs}')
        logger.info(f'Finished processing day {day} data')
        return 

    def _remove_header(self) -> list[bse.Tag]:
        """removes header from list for easier iteration"""
        data = self.soup.find_all('tr')
        data.pop(0)
        return data

        
    def pickle_save(self, override:bool=False):
        """ Saves extracted HTML data to pickle file
        
        Parameters
        ----------
            override : bool, default=5
                If true, any existing pickle files in the reference_data directory will be deleted
        
        """
        if override:
            for file in REF_DATA_DIR.glob('*.pkl'):
                os.remove(os.path.join(REF_DATA_DIR, file))

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

    def csv_export(self, result_type:list[Literal['plain', 'perc best', 'recsales']]=None):
        """ Exports specified results to CSV, or all if None """
        
        if result_type is None:
            queue = self.results.copy()
        else:
            queue = {result:self.results[result] for result in self.results if result in result_type}
        
        for result in queue:
            queue[result].to_csv(f'{self.export_directory}/{result}.csv', header=True)

        print(f'done exporting CSVs to {self.export_directory}')
        return

    def view_all_results(self):
        """ Prints all dataframes """
        for df in self.results:
            print(self.results[df])
        return 



# TODO would like to generalize visualization eventually
# need to figure out what visualizations I would want
# currently have day x value/stdev per plort
def plot_rec_data(rec_df:pd.DataFrame, plorts:list[str]=None):
    assert isinstance(plorts, list), f'Must supply list for plorts, not {type(plorts)}'

    rec_df_c = rec_df.copy()
    if plorts is None: # separate all plorts into grouped DFs to plot
        plort_groups = rec_df_c.groupby('plort')
        plort_group_df:list[dict[Literal['plort', 'df'],pd.DataFrame]] = [{'plort':groups[0], 'df':groups[1].sort_values(by='Day')} for groups in plort_groups]

    else: # separate only the specified plorts into grouped DFs to plot
        plort_group_df = [{'plort': plort, 'df':rec_df_c[rec_df_c['Plort'] == plort.capitalize()].sort_values(by='Day')} for plort in plorts]
    
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
        ax.plot(group_df['Day'], group_df['Value'], marker='o', linestyle='-', label='Value', color='blue')
        ax.plot(group_df['Day'], group_df['Percentage Best'], marker='s', linestyle='--', label='Percentage Best', color='orange')

        # titles and labels
        ax.set_title(plort_name)
        ax.set_xlabel('Day')
        ax.set_ylabel('Value / Percentage Best')
        ax.legend()
        ax.grid(True)

    # Remove empty subplots if there are any
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Show the plot
    plt.suptitle("Trends by Plort", fontsize=16)
    plt.show()

