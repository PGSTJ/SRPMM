from . import (
    HtmlProcessor, seaborn as sns, pd, plt, logging, np, Literal, dataclass,
    train_test_split, mean_squared_error, accuracy_score, log_loss, confusion_matrix, 
    mean_absolute_error, r2_score,
    classification_report, roc_auc_score, TimeSeriesSplit, RandomForestClassifier,
    LinearRegression, LogisticRegression, StandardScaler, PolynomialFeatures,
    GridSearchCV, GradientBoostingRegressor
)

logger = logging.getLogger('standard')

SAVE_METADATA:dict[str,any] = HtmlProcessor().pickle_load()

def create_pair_plot(dataframe:pd.DataFrame):

    # sns.jointplot(x='day', y='value', data=honey_data, kind='reg')
    sns.pairplot(dataframe, kind='scatter', plot_kws={'alpha': 0.4})
    plt.show()

def create_basic_plort_plot(data:pd.DataFrame, plort:str):
    """ Visualize plort value and percentage best over days
    
    Parameters
    ----------
        data : pandas.Dataframe
            Dataframe with data to plot

        plort : str
            Plort data being plotted. Required for graph labeling

        modifier : list[str]
            Keywords to modify the basic plort plot into slightly more complex plots
            
                - RCS: Recommended Sale Comparison. Adds the "Sell" column to each plot 
                to compare against trends against the recommended sale days


    
    """
    fig,axes = plt.subplots(2,1, figsize=(12,5))

    sns.scatterplot(ax=axes[0], data=data, x='Day', y='Plort Value')
    sns.scatterplot(ax=axes[1], data=data, x='Day', y='Percentage Best')

    plt.title(f'{plort} Value and Percentage Best Trends', y=2.2)
    plt.show()
    return
    
def create_comparison_plort_plot(data:pd.DataFrame, plort:str):
    """ Visualize plort value and percentage best against the recommended sale days over days """
    fig,axes = plt.subplots(2,1, figsize=(12,5))

    sns.scatterplot(ax=axes[0], data=data, x='Day', y='Plort Value', hue=data['Sell'], palette={1: 'red',0:'blue'}, legend='full')
    # sns.scatterplot(ax=axes[0], data=data, x='Day', y='Sell')
    sns.scatterplot(ax=axes[1], data=data, x='Day', y='Plort Percentage Best', hue=data['Sell'], palette={1: 'red',0:'blue'}, legend='full')
    # sns.scatterplot(ax=axes[1], data=data, x='Day', y='Sell')

    plt.title(f'{plort} Value and Percentage Best against recommended sale days', y=2.2)
    plt.show()
    return


@dataclass
class MADataset():
    name:str = ''
    X_data:pd.DataFrame = None
    y_data:pd.Series = None

    def __str__(self):
        return f'Dataset: {self.name}\n\tX shape: {self.X_data.shape}\n\ty shape: {self.y_data.shape}'
    
    def positive_target_ratio(self):
        """ ratio of positive targets """
        pos = [i for i in self.y_data if i == 1]
        return len(pos) / len(self.y_data)
    
    def show_data(self):
        print(self)
        print(f'X headers: {self.X_data.columns.to_list()} | target headers: {self.y_data.axes}')
        if self.regression == 'logistic':
            print(f'Positive target ratio: {self.positive_target_ratio()}')


class MarketAnalysisSession():
    """ Handles model interactions with data from a given plort or market dataset 
    
    Parameters
    ----------
        session_name : str
            Name of the session

        session_data : dict
            Currently only allows a dict with set keys referring to training/test/cv subdatasets 
            of the given session 

        model_type : Literal['randomforest', 'xgb']
            Choose the model type 

            
    Args
    ----
        iterations : int, default=None
            Only required when regression is set to 'logistic'

    
    """
    def __init__(
            self, 
            session_name:str, 
            session_data:dict[Literal['Training', 'Test', 'CV'], MADataset],
            model_type:Literal['randomforest', 'xgb'],
            ):
        self.name = session_name
        self.session_data:dict[Literal['Training', 'Test', 'CV'], MADataset] = session_data
        
        self.model = None
        self.model_parameters = {
            'model': model_type,
            'param_grid' : '',
            'gscv_scoring': ''
        }

        self.scaler = StandardScaler()

        self.session_metadata:dict[str,dict[str,any]] = {
            'Training': {'Results': {}, 'Metrics': {}},
            'Test': {'Results': {}, 'Metrics': {}},
            'CV': {'Results': {}, 'Metrics': {}}
        }

    def _get_model(
            self,
            model_type:Literal['rfclassifier', 'xgb'],
            ):
        """ Regression model 
        
        Parameters
        ----------
            X : pandas.DataFrame

            y : pandas.Series

            regression : str

            iterations : int, default=100
        
        """

        if model_type == 'rfclassifier':
            model = RandomForestClassifier(random_state=42)
            self.model_parameters['param_grid'] = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
            }
            self.model_parameters['gscv_scoring'] = 'f1'

        elif model_type == 'xgb':
            model = GradientBoostingRegressor()
            self.model_parameters['param_grid'] = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
            self.model_parameters['gscv_scoring'] = 'neg_mean_absolute_error'

        else:
            raise ValueError(f'Unrecognized model type. Expected: {['rfclassifier', 'xgb']} Got: {model_type}')
        logger.info(f'{type(model)} model initialized')

        return model

    # first iteration
    def run_model(
            self, 
            set_type:Literal['Training', 'Test', 'CV'],
            timeseries_n_splits:int = 5
            ):
        """ Runs model starting 
        
        Parameters
        ----------
            set_type : Literal['Training', 'Test', 'CV']
                Specify type of model run

            timeseries_n_splits : int, default=5
                Number of splits passed into the ```TimeSeriesSplit``` object
        
        """
        self.model = self._get_model(self.model_parameters['model'])
        dataset = self.session_data[set_type]
        logger.debug(f'Training columns: {dataset.X_data.columns}')
        self.session_metadata[set_type]['Warnings'] = {}

        split = TimeSeriesSplit(n_splits=timeseries_n_splits)
        
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.model_parameters['param_grid'], cv=split, scoring=self.model_parameters['gscv_scoring'])
        grid_search.fit(dataset.X_data, dataset.y_data)

        best_model = grid_search.best_estimator_.fit(dataset.X_data, dataset.y_data)
        y_pred = best_model.predict(dataset.X_data)
        y_proba = best_model.predict_proba(dataset.X_data)

        logger.info("Best Parameters:", grid_search.best_params_)
        self.session_metadata[set_type]['Best Model Parameters'] = grid_search.best_params_

        self.session_metadata[set_type]['Results'] = {
            'Predictions':y_pred, 
            'Prediction Probabilities':y_proba, 
            'Accuracy': best_model.score(dataset.X_data,dataset.y_data),
            }
        
        # logger.debug(self.session_metadata)
        return      

    
    def get_results(self, set_type:Literal['Training', 'Test', 'CV']) -> np.ndarray:
        print(f'{set_type} Accuracy (score): ', self.session_metadata[set_type]['Results']['Accuracy'])

        print('Results:\n')
        for result in self.session_metadata[set_type]['Results']:
            print(f'{result}: {self.session_metadata[set_type]["Results"][result]}')

        print('Metrics:\n')
        self._calculate_metrics(set_type)
        for metric in self.session_metadata[set_type]['Metrics']:
            print(f'{metric}: {self.session_metadata[set_type]["Metrics"][metric]}')


        if len(self.session_metadata[set_type]['Warnings']) > 0:
            print(self.session_metadata[set_type]['Warnings'])

        return self.session_metadata[set_type]['Results']['Predictions']
    
    def _calculate_metrics(self, set_type):
        true_y = self.session_data[set_type].y_data
        pred_y = self.session_metadata[set_type]['Results']['Predictions']
        proba_y = self.session_metadata[set_type]['Results']['Prediction Probabilities']
        
        accuracy = accuracy_score(true_y, pred_y)
        self.session_metadata[set_type]['Metrics'] = {'Accuracy': accuracy}

        if isinstance(self.model, RandomForestClassifier):
            self.session_metadata[set_type]['Metrics']['Loss'] = log_loss(true_y, proba_y)
            self.session_metadata[set_type]['Metrics']['Confusion Matrix'] = confusion_matrix(true_y, pred_y)
            self.session_metadata[set_type]['Metrics']['Classification Report'] = classification_report(true_y, pred_y)
            self.session_metadata[set_type]['Metrics']['AUC'] = roc_auc_score(true_y, pred_y)

        elif isinstance(self.model, GradientBoostingRegressor):
            self.session_metadata[set_type]['Metrics']['Loss'] = mean_squared_error(true_y, pred_y)
            self.session_metadata[set_type]['Metrics']['MAE'] = mean_absolute_error(true_y, pred_y)
            self.session_metadata[set_type]['Metrics']['R2'] = r2_score(true_y, pred_y)

        return 


    def determine_best_poly_features(self, max_degree:int):
        t_losses = []
        cv_losses = []
        self.session_metadata['Training'] = {'Warnings': {}}
        self.session_metadata['CV'] = {'Warnings': {}}

        t_dataset = self._check_nans(self.session_data['Training'], 'Training')
        cv_dataset = self._check_nans(self.session_data['CV'], 'CV')
        

        for degree in range(1, max_degree+1):
            logger.info(f'Assessing Degree: {degree}')
            poly = PolynomialFeatures(degree, include_bias=False)
            X_mapped = poly.fit_transform(t_dataset.X_data)
            # can save poly model to list

            X_mapped_norm = self.scaler.fit_transform(X_mapped)
            # can save scaler? not sure why --> maybe have to start instantiating separately

            model = self._model(self.model_parameters['reg'], iterations=self.model_parameters['iters'])
            model.fit(X_mapped_norm, t_dataset.y_data)
            # can save reg model 

            y_probs = model.predict_proba(X_mapped_norm)
            loss = log_loss(t_dataset.y_data, y_probs)
            t_losses.append(loss)


            X_mapped_cv = poly.transform(cv_dataset.X_data)
            X_mapped_cv_norm = self.scaler.transform(X_mapped_cv)


            cv_y_probs = model.predict_proba(X_mapped_cv_norm)
            cv_loss = log_loss(cv_dataset.y_data, cv_y_probs)
            cv_losses.append(cv_loss)
        
        best_degree = np.argmin(cv_losses) + 1
        print(f'Lowest CV loss found in model with degree={best_degree}')
        return t_losses, cv_losses

    @staticmethod
    def dbpf_visualize(test_loss_collection:list[float], cv_loss_collection:list[float]):
        t_data = pd.DataFrame([i for i in enumerate(test_loss_collection, start=1)], columns=['Degree', 'Log Loss'])
        cv_data = pd.DataFrame([i for i in enumerate(cv_loss_collection, start=1)], columns=['Degree', 'Log Loss'])

        fig,axes = plt.subplots(2, 1, figsize=(12,5))

        sns.scatterplot(ax=axes[0], data=t_data, x='Degree', y='Log Loss')
        sns.scatterplot(ax=axes[1], data=cv_data, x='Degree', y='Log Loss')

        axes[0].set_title(f'Training Errors across {len(t_data)} Degrees')
        axes[1].set_title(f'CV Errors across {len(cv_data)} Degrees')

        plt.show()
        return



            


class MarketAnalysisFormatter():
    """ Creates datasets of dataframe feature combinations 

    """

    def __init__(self):
        logger.info('Initializing MA Formatter')
        self.data:dict[Literal['plain', 'perc best', 'recsales'], pd.DataFrame] = SAVE_METADATA['Results']
        self.html_metadata = {i:SAVE_METADATA[i] for i in SAVE_METADATA if i != 'Results'}

    def create_datasets(
            self,
            dataset_name:str,
            data:pd.DataFrame,
            type:Literal['PR', 'PP'],
            x_columns:list[str],
            target:str,
            training_size:float = 0.5
            ) -> dict[Literal['Training', 'Test', 'CV'], MADataset]:
        """ Creates a dataset for model training/testing/CVing

        Depending on session type, will engineer certain features and append to the supplied dataframe

        Parameters
        ----------
            data : pandas.DataFrame
                DataFrame of the data in question

            regression : Literal['linear', 'logistic'] 
                Indicate whether the dataset is for linear or logistic regression

            target : str, default=None
                Corresponds to the column label that represents the target data. Not required for linear regression

        Args
        ----
            type : Literal['PR', 'PP']
                Specifies model purpose type

                - SRP: Sale Recommendation Predictor. Requires window_size to be supplied.
            
            window_size : int
                Number of days to analyze price trends. Only required when type='STP'
         
        """
        c_data = data.copy()
        assert target in c_data.columns.to_list(), f'Selected target: {target} does not exist in dataset. Dataset columns: {c_data.columns}'

        # TODO put engineered features in own functions
        if type == 'SRP':
            c_data['Price Diff'] = c_data['Plort Value'].diff()
            # c_data['local_max'] = (c_data['Plort Value'] > c_data['Plort Value'].shift(1)) & (c_data['Plort Value'] > c_data['Plort Value'].shift(-1))
            c_data['rolling_avg'] = c_data['Plort Value'].rolling(window=5).mean()
            c_data['rolling_std'] = c_data['Plort Value'].rolling(window=5).std()

            # EMA, RSI, and Bollinger Bands
            c_data['EMA'] = c_data['Plort Value'].ewm(span=12, adjust=False).mean()
            c_data['RSI'] = 100 - (100 / (1 + c_data['Plort Value'].diff().clip(lower=0).rolling(window=14).mean() /
                                        abs(c_data['Plort Value'].diff()).rolling(window=14).mean()))
            c_data['Upper BB'] = c_data['rolling_avg'] + 2 * c_data['rolling_std']
            c_data['Lower BB'] = c_data['rolling_avg'] - 2 * c_data['rolling_std']

            fe_columns = ['Price Diff', 'rolling_avg', 'rolling_std', 'EMA', 'RSI', 'Upper BB', 'Lower BB']
            x_columns += fe_columns

        elif type == 'PP':
            c_data["rolling_avg"] = c_data["Plort Value"].rolling(window=7).mean()
            c_data["rolling_std"] = c_data["Plort Value"].rolling(window=7).std()
            
            # RSI - by parts
            delta = c_data["Plort Value"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            c_data["RSI"] = 100 - (100 / (1 + (gain / loss)))
            
            # Fourier features for seasonality
            c_data["sin_day"] = np.sin(2 * np.pi * c_data["Day"] / 150) # TODO alter "total days" in year, currently 150
            c_data["cos_day"] = np.cos(2 * np.pi * c_data["Day"] / 150)
            
            # Saturation multiplier approximation TODO fix saturation multiplier
            # c_data["Saturation_Multiplier"] = 1 + (1 - np.minimum(c_data["Value"], metadata["Saturation_Threshold"]) / metadata["Saturation_Threshold"])

            fe_columns = ["rolling_avg", "rolling_std", "RSI", "sin_day", "cos_day", "Saturation_Multiplier"]
            x_columns += fe_columns

        X_data = c_data[x_columns].dropna()
        y_data = c_data[target]

        split_data = self.split_data_temporally(X_data, y_data, training_size=training_size)
        return {set:MADataset(name=dataset_name, X_data=split_data[set][0], y_data=split_data[set][1]) for set in split_data}
        
    def start_session(
            self, 
            session_type:Literal['PR', 'PP'],
            x_columns:list[str],
            plort:str=None, 
            target_column:str=None
            ):
        """ Formats data for specified session type then returns a Session
        
        Parameters
        ----------              
            session_type : Literal['PR', 'PP']
                Indicates the type of training session to format data for

                - 'PR' : Plort Recommendation; Prepares session to train a model on recommended sale data to predict recommended sale days
                - 'PP' : Price Prediction; Prepares a session to train a model on sell value trends to predict future prices

            x_columns : list[str]
                List of column names to use in training (raw HTML data only - engineered features are automatically created)

            target_column : str, default=None
                For classification tasks only. Refers to the column containing target for classification
            
            
            plort : str, default=None
                Specifies the specific plorts' trend to learn
        
        """
        if plort:
            combined_raw_df = self.combine_plort_dfs(plort)
            ds_name = f'{plort.capitalize()} {session_type}' # Sale Recommendation Prediction
        else:
            combined_raw_df = self.combine_market_df()
            ds_name = f'Market {session_type}'
        

        try:
            datasets = self.create_datasets(dataset_name=ds_name, data=combined_raw_df, x_columns=x_columns, target=target_column, type=session_type)
        except KeyError as e:
            logger.error(f'KeyError: {e} | {combined_raw_df.columns}')
            raise KeyError(e)
        
        
        return MarketAnalysisSession(
            session_name=ds_name, 
            session_data=datasets,
            model_type='rfclassifier'
        )

    def combine_plort_dfs(self, plort:str) -> pd.DataFrame: # USED FOR LOGISTIC REGRESSION
        """ Combines the price and percentage best values for the given plort along as well as whether it was a recommended sale 
        
        Returns DF with shape (999,4)
        
        """
        days = np.arange(start=1,stop=1000) # TODO generalize in class after moving past initial static predictions
        plort_recsales = self.data['recsales'][self.data['recsales']['Plort']==plort.capitalize()] 

        plort_value = self.data['plain'][plort.capitalize()]
        percetange_best = self.data['perc best'][plort.capitalize()]
        days_recommended_sale:list[bool] = [1 if day in plort_recsales['Day'].tolist() else 0 for day in days] # target array

        combined_data = {
                'Day': days,
                'Plort Value': plort_value,
                'Plort Percentage Best': percetange_best,
                'Sell': days_recommended_sale
            }


        logger.info(f'Combined {plort} data into one DF with columns: {[i for i in combined_data]}')
        return pd.DataFrame(data=combined_data)

    def combine_market_df(self) -> pd.DataFrame: # USED FOR LINEAR REGRESSION
        """ Combines the market price and best values 
        
        Returns DF with shape (999,3)
        
        """

        market_value_days = self.data['plain'][['Day','Market']] # first two columns
        market_percentage_best = self.data['perc best']['Market']

        market_value_days.insert(len(market_value_days.columns), 'Market Percentage Best', market_percentage_best) # third column
        # TODO rename 'Market' column from original DF to 'Market Value'

        logger.info(f'Combined market data into one DF')
        return market_value_days
        
    # function to find local min/max across range of days
    def extract_local_extremes(self, plort:str, period:int, chunk_review_limit:int=None):
        """ Extracts local maxima/mimima in a specified range of days 
        
        Parameters
        ----------
            plort : str

            period : int
                Range of days to partition data

            chunk_review_limit : int
                Limits number of chunks that show their graph
        
        """

        plort_data = self.combine_plort_dfs(plort)
        chunks = [plort_data.iloc[i:i+period] for i in range(0,len(plort_data), period)]
        
        if chunk_review_limit is None:
            chunk_queue = chunks.copy()
        else:
            chunk_queue = chunks[:chunk_review_limit]
        logger.info(f'Reviewing {len(chunk_queue)} chunks for local maxima analysis')

        for df in chunk_queue:
            # create_comparison_plort_plot(df, plort)
            print(df)
            return



    def split_data_temporally(self, xdata:pd.DataFrame, ydata:pd.Series, training_size:float=0.5):
        """ Splits data into training/test/CV based on training size percentage """

        assert len(xdata) == len(ydata), f'Unable to split data due to mismatching data lengths - x: {len(xdata)} | y: {len(ydata)}'
        train_index_split = int(round(training_size * len(xdata), 0))
        cv_test_size = int(round((training_size / 2) * len(xdata), 0))
        
        cv_index_range = (train_index_split, int(train_index_split+cv_test_size))
        test_index_range = int(train_index_split+cv_test_size)

        train_x = xdata.iloc[:train_index_split]
        train_y = ydata.iloc[:train_index_split]
        logger.info(f'train: {train_x.shape}\n{train_x}\n')

        cv_x = xdata.iloc[train_index_split:cv_index_range[1]]
        cv_y = ydata.iloc[train_index_split:cv_index_range[1]]
        logger.info(f'cv: {cv_x.shape}\n{cv_x}\n')

        test_x = xdata.iloc[test_index_range:]
        test_y = ydata.iloc[test_index_range:]
        logger.info(f'test: {test_x.shape}\n{test_x}\n')

        return {
            'Training': (train_x, train_y),
            'CV': (cv_x, cv_y),
            'Test': (test_x, test_y)
        }

        # print(
        #     f'index 0 -> {train_index_split = }\n'
        #     f'{cv_test_size = }\n'
        #     f'cv index range - {train_index_split} : {cv_index_range}\n'
        #     f'{test_index_range = } until end\n'
        # )
        return

