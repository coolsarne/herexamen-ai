# import all the libraries that you need at the top of the notebook
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor, \
    StackingRegressor
import joblib
from sklearn.svm import SVR

# Define BUCKET_ROOT (for now a dummy value will do, this will become clear in Part 3)
BUCKER_ROOT = 'dummy'
# Define DATA_DIR (initially this will be a local directory, but later it will be a Google Cloud Storage bucket)
DATA_DIR = './'


# helper methods for dataset loading
def categorize_price(price):
    if price == 0:
        return "FREE"
    elif 0 < price < 5.0:
        return "CHEAP"
    else:
        return "EXPENSIVE"


def remove_outliers(df, columns, n_std):
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()

        df = df[(df[col] <= mean + (n_std * sd))]

    return df


# the method below will return the data transformed, standarized, outliers removed, and feature engineered
def load_data(data_dir):
    apps = pd.read_csv(data_dir)
    apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')
    apps['Size'] = apps['Size'].apply(
        lambda x: float(x.replace('k', '')) / 1024 if 'k' in str(x) else x)  # Convert k to M
    apps['Size'] = apps['Size'].apply(
        lambda x: None if x == 'Varies with device' else x)  # Handles "Varies with device" values
    apps['Size'] = apps['Size'].str.replace('M', '').str.replace('k', '').astype(float)
    apps['Installs'] = apps['Installs'].str.replace('+', '').str.replace(',', '').astype('int64')
    apps['Price'] = apps['Price'].str.replace('$', '').astype(float)
    apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])
    apps = apps.drop_duplicates(subset='App', keep='first')
    apps = remove_outliers(apps, ['Price'], 3)
    apps = apps.dropna(subset=['Rating'])
    apps['Size'] = apps['Size'].fillna(0)
    encoder = LabelEncoder()
    apps['Category_Encoded'] = encoder.fit_transform(apps['Category'])
    apps['Content Rating_Encoded'] = encoder.fit_transform(apps['Content Rating'])
    apps['Android Ver_Encoded'] = encoder.fit_transform(apps['Android Ver'])
    current_date = pd.to_datetime('2019-01-01')  # All data is from before 2019
    apps['Time Since Last Update'] = (current_date - apps['Last Updated']).dt.days
    apps['App_Name_Length'] = apps['App'].apply(len)
    apps['Price_Category_Labels'] = apps['Price'].apply(categorize_price)
    apps['Price_Category_Encoded'] = apps['Price_Category_Labels'].map({'FREE': 0, 'CHEAP': 1, 'EXPENSIVE': 2})
    numeric_columns = apps.select_dtypes(include=['number'])
    numeric_columns = numeric_columns.drop(['Price'], axis=1)
    apps = apps[numeric_columns.columns]
    scaler = StandardScaler()
    numeric_columns = ['Reviews', 'Size', 'Installs', 'Category_Encoded', 'Content Rating_Encoded',
                       'Android Ver_Encoded', 'Time Since Last Update', 'App_Name_Length', 'Price_Category_Encoded']
    apps[numeric_columns] = scaler.fit_transform(apps[numeric_columns])
    selector = SelectPercentile(score_func=f_regression, percentile=80)
    X = selector.fit_transform(apps.select_dtypes(include=np.number), apps.Rating)
    best_features = selector.get_support(indices=True)
    apps = apps.select_dtypes(include=np.number).iloc[:, best_features]
    return apps


def split_data(data):
    apps = data
    x = apps.drop('Rating', axis=1)
    y = apps.Rating
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test, x, y


def train_model(x_train, y_train, x_test, y_test, type):  # type can either be 'voting' or 'bagging' or 'stacking'
    if type == 'voting':
        rf_model = RandomForestRegressor(n_estimators=93)
        tree_model = DecisionTreeRegressor(max_leaf_nodes=15)
        gradboost_model = GradientBoostingRegressor(max_depth=5, n_estimators=30)

        ensemble = VotingRegressor(
            estimators=[('rf', rf_model),
                        ('tree', tree_model),
                        ('gradboost', gradboost_model)]
        )
        ensemble.fit(x_train, y_train)
        for name, model in ensemble.named_estimators_.items():
            print(f'{name} = {model.score(x_test, y_test)}')
        print(f"voting ensemble score: {ensemble.score(x_test, y_test)}")
        return ensemble
    elif type == 'bagging':

        ensemble = BaggingRegressor(DecisionTreeRegressor(),
                                    n_estimators=500, max_samples=100,
                                    n_jobs=-1, random_state=42)
        ensemble.fit(x_train, y_train)
        print(f"bagging ensemble score: {ensemble.score(x_test, y_test)}")
        return ensemble
    elif type == 'stacking':
        rf_model = RandomForestRegressor(n_estimators=93)
        tree_model = DecisionTreeRegressor(max_leaf_nodes=15)
        gradboost_model = GradientBoostingRegressor(max_depth=5, n_estimators=30)
        svr_model = SVR(kernel='linear')

        ensemble = StackingRegressor(
            estimators=[('rf', rf_model),
                        ('tree', tree_model),
                        ('gradboost', gradboost_model)],
            final_estimator=svr_model, n_jobs=-1, cv=5
        )
        ensemble.fit(x_train, y_train)
        print(f"stacking ensemble score: {ensemble.score(x_test, y_test)}")
    else:
        raise Exception("type parameter should be either 'voting', 'bagging' or 'stacking'")


def save_model(model):
    path = f'{DATA_DIR}/model_artifacts/model.joblib'
    joblib.dump(model, path)
    return path


data = load_data('./googleplaystore.csv')
x_train, x_test, y_train, y_test, x, y = split_data(data)

# voting ensemble score: 0.14006841785330926
# bagging ensemble score: 0.11867157399190797
# stacking ensemble score: 0.15291239117867617
# > We use the best scoring ensemble > stacking

# type = 'voting'
# trained_model = train_model(x_train, y_train, x_test, y_test, type=type)
# save_model(trained_model, type=type)
#
# type = 'bagging'
# trained_model = train_model(x_train, y_train, x_test, y_test, type=type)
# save_model(trained_model, type=type)

type = 'stacking'
trained_model = train_model(x_train, y_train, x_test, y_test, type=type)
save_model(trained_model)
