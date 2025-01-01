
from Settings import (
    os, seaborn, pd, plt, logging, np,
    Literal, dataclass
)

from utils.htmlprocessor import HtmlProcessor

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor


