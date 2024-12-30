from . import (
    PreProcessor, config,
    seaborn as sns, pd, plt, logging, np, Literal, dataclass,
    train_test_split, mean_squared_error, accuracy_score, log_loss, confusion_matrix, 
    classification_report, roc_auc_score,
    LinearRegression, LogisticRegression, StandardScaler, PolynomialFeatures
)

logger = logging.getLogger('standard')

ALL_DATA_DF:dict[str,pd.DataFrame] = PreProcessor().pickle_load()

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

    
    """
    def __init__(
            self, 
            session_name:str, 
            session_data:dict[Literal['Training', 'Test', 'CV'], MADataset],
            regression:Literal['linear', 'logistic'],
            iterations:int=None
    ):
        self.name = session_name
        self.session_data:dict[Literal['Training', 'Test', 'CV'], MADataset] = session_data
        
        self.model_parameters = {
            'reg': regression,
            'iters': iterations
        }

        self.scaler = StandardScaler()

        self.session_metadata:dict[str,dict[str,any]] = {}

    def _model(
            self,
            regression:Literal['linear', 'logistic'],
            iterations:int=None
            ):
        """ Regression model 
        
        Parameters
        ----------
            X : pandas.DataFrame

            y : pandas.Series

            regression : str

            iterations : int, default=100
        
        """

        if regression == 'linear':
            model = LinearRegression()
        elif regression == 'logistic':
            model = LogisticRegression(
                max_iter=iterations,
                # penalty='elasticnet',
                solver='liblinear',
                random_state=1
            )
        else:
            raise ValueError(f'Must specify regression type as Linear or Logistic, not', regression)
        logger.info(f'{type(model)} model initialized')

        # plot_data_results(x=X, type='pre', y=y, title='Pre Normalization')

        return model

    # TODO might deprecate
    def set_iterations(self, amount:int):
        original_length = self.model_parameters['iters'] if self.model_parameters['iters'] is not None else 100
        self.model_parameters['iters'] = amount
        logger.info(f'Updated iteration length for Session: {self.name} from {original_length} to {amount}')

    # first iteration
    def run_prediction(self, set_type:Literal['Training', 'Test', 'CV']):
        model = self._model(self.model_parameters['reg'], iterations=self.model_parameters['iters'])
        dataset = self._check_nans(self.session_data[set_type], set_type)
        logger.debug(f'Training columns: {dataset.X_data.columns}')
        self.session_metadata[set_type]['Warnings'] = {}
        
        X_norm = self.scaler.fit_transform(dataset.X_data)
        self.session_metadata[set_type] = {'Normalized Data': X_norm}


        assert len(X_norm) == len(dataset.y_data), f'X and y length mismatch: x: {len(X_norm)} | y: {len(dataset.y_data)}\nRecheck NA cleaning.'
        # return dataset.y_data
        
        model.fit(X_norm, dataset.y_data)
        y_pred = model.predict(X_norm)
        y_proba = model.predict_proba(X_norm)


        self.session_metadata[set_type]['Results'] = {
            'Predictions':y_pred, 
            'Prediction Probabilities':y_proba, 
            'Accuracy': model.score(X_norm,dataset.y_data),
            'AUROC': roc_auc_score(dataset.y_data, y_proba[:, 1])
            }
        logger.debug(self.session_metadata)
        # print(classification_report(dataset.y_data, y_pred))
        return model
    
    def _check_nans(self, original_data:MADataset, set_type):
        """ Documents raw values where normalized versions are NaN """
        header = original_data.X_data.columns
        clean_df = original_data.X_data.dropna()

        list_all = original_data.X_data.to_dict(orient='index')
        no_nas = clean_df.to_dict(orient='index')

        # return list_all, no_nas

        if len(list_all) != len(no_nas):
            dd = {idx:list_all[idx] for idx in list_all if idx not in no_nas}
            dropped_f = {'indexes':[i for i in dd], 'og_values':[dd[i] for i in dd]}
            self.session_metadata[set_type]['Warnings']['Dropped Data'] = dropped_f
            logger.warning(f'Removed rows with NaN values: {dd}')

        

        if len(original_data.y_data) != len(clean_df):
            yd = original_data.y_data.to_list()
            for idx in self.session_metadata[set_type]['Warnings']['Dropped Data']['indexes']:
                yd.remove(yd[idx])
                    
            
            original_data.y_data = pd.Series(yd)
        original_data.X_data = clean_df
        

        return original_data

        

    
    def get_results(self, set_type:Literal['Training', 'Test', 'CV']) -> np.ndarray:
        print(f'{set_type} Accuracy (score): ', self.session_metadata[set_type]['Results']['Accuracy'])

        acc,loss,cm = self._calculate_metrics(set_type)

        print(
            f'Metrics:\n',
            f'\tAccuracy (acc_score): {acc}\n',
            f'\tLog Loss: {loss}\n',
            f'\tConfusion Matrix: {cm}\n',
        )

        if len(self.session_metadata[set_type]['Warnings']) > 0:
            print(self.session_metadata[set_type]['Warnings'])

        return self.session_metadata[set_type]['Results']['Predictions']
    
    def _calculate_metrics(self, set_type):
        true_y = self.session_data[set_type].y_data
        pred_y = self.session_metadata[set_type]['Results']['Predictions']
        proba_y = self.session_metadata[set_type]['Results']['Prediction Probabilities']
        
        accuracy = accuracy_score(true_y, pred_y)
        loss = log_loss(true_y, proba_y)
        cm = confusion_matrix(true_y, pred_y)

        return accuracy,loss,cm


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
        for result in ALL_DATA_DF:
            self.__dict__[result] = ALL_DATA_DF[result].copy()
            logger.info(f'Added result data parameter: {result}')


    def create_datasets(
            self,
            dataset_name:str,
            data:pd.DataFrame,
            target:str=None,
            *,
            window_size:int,
            type:Literal['SRP']=None
            ) -> dict[Literal['Training', 'Test', 'CV'], MADataset]:
        """ Creates a dataset for model training/testing/CVing

        Parameters
        ----------
            data : pandas.DataFrame
                DataFrame of the data in question

            regression : Literal['linear', 'logistic'] 
                Indicate whether the dataset is for linear or logistic regression

            target : str, default=None
                Corresponds to the column label that represents the target data. Not required for linear regression

            type : Literal['SRP']
                Specifies model purpose type

                - SRP: Sale Recommendation Predictor. Requires window_size to be supplied.
            
            window_size : int
                Number of days to analyze price trends. Only required when type='STP'
         
        """
        c_data = data.copy()
        assert target in c_data.columns.to_list(), f'Selected target: {target} does not exist in dataset. Dataset columns: {c_data.columns}'

        if type == 'SRP':
            c_data['diff'] = c_data['Plort Value'].diff()
            c_data['local_max'] = (c_data['Plort Value'] > c_data['Plort Value'].shift(1)) & (c_data['Plort Value'] > c_data['Plort Value'].shift(-1))
            c_data['rolling_avg'] = c_data['Plort Value'].rolling(window=window_size).mean()
            c_data['rolling_std'] = c_data['Plort Value'].rolling(window=window_size).std()

        target_data = c_data[target]
        x = c_data.drop(columns=target)

        X_train, x_, y_train, y_ = train_test_split(x, target_data, test_size=0.6, random_state=1)
        X_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

        del x_,y_

        return {
            'Training': MADataset(name=dataset_name,X_data=X_train, y_data=y_train),
            'Test': MADataset(name=dataset_name,X_data=x_test, y_data=y_test),
            'CV': MADataset(name=dataset_name,X_data=X_cv, y_data=y_cv)
        }
        
    def create_plort_recommendations_session(
            self, 
            plort:str, 
            regression:Literal['linear', 'logistic'],
            iterations:int=100, # TODO likely a better place for iteration and regression definition
            remove_columns:list[str]|str='Day',
            *,
            window_size:int
            
    ) -> MarketAnalysisSession:
        """ Creates dataset for learning plort sale recommendations and a new Session """ # TODO account for exising sessions?
        plort_data_dfs = self.combine_plort_dfs(plort)
        ds_name = f'{plort.capitalize()} SRP' # Sale Recommendation Prediction
        
        plort_data_dfs_clean = plort_data_dfs.drop(columns=remove_columns)

        
        try:
            datasets = self.create_datasets(dataset_name=ds_name, data=plort_data_dfs_clean, target='Sell', type='SRP', window_size=window_size)
        except KeyError as e:
            logger.error(f'KeyError: {e} | {plort_data_dfs_clean.columns}')
            raise KeyError(e)
        except Exception as e:
            logger.error(f'{ds_name} - {plort_data_dfs_clean = }')
            raise e
        
        return MarketAnalysisSession(
            session_name=f'{plort} Analysis', 
            session_data=datasets,
            regression=regression,
            iterations=iterations
        )

    def combine_plort_dfs(self, plort:str) -> pd.DataFrame: # USED FOR LOGISTIC REGRESSION
        """ Combines the price and percentage best values for the given plort along as well as whether it was a recommended sale """
        days = np.arange(start=1,stop=1000) # TODO generalize in class after moving past initial static predictions
        plort_recsales = self.recsales[self.recsales['plort']==plort.capitalize()] 

        plort_value = self.plain[plort.capitalize()]
        percetange_best = self.stdev[plort.capitalize()]
        days_recommended_sale:list[bool] = [1 if day in plort_recsales['day'].tolist() else 0 for day in days] # target array

        combined_data = {
                'Day': days,
                'Plort Value': plort_value,
                'Plort Percentage Best': percetange_best,
                'Sell': days_recommended_sale
            }


        logger.info(f'Combined {plort} data into one DF with columns: {[i for i in combined_data]}')
        return pd.DataFrame(data=combined_data)

    def combine_market_df(self) -> pd.DataFrame: # USED FOR LINEAR REGRESSION
        """ Combines the market price and best values """

        market_value_days = ALL_DATA_DF['plain'][['Day','Market']] # first two columns
        market_percentage_best = ALL_DATA_DF['stdev']['Market']

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
    

