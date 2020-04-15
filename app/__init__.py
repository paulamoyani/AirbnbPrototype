# Gevent needed for sockets
from gevent import monkey
monkey.patch_all()

# Imports
import os
from flask import Flask, render_template, request, json
# from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
# import filters
import time
import pickle
import numpy as np
import json
import pickle
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import math
import csv


import numpy as np
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer

import matplotlib
import sklearn
from IPython.core.display import display, HTML

""
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.9)

# TO WORK WITH
import pandas as pd
import numpy as np
from numpy import set_printoptions

# HIDE WARNINGS
import warnings
warnings.filterwarnings('ignore')

# PREPROCESSING & MODEL SELECTION
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import SCORERS
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# print(SCORERS.keys())

# PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn import tree
from graphviz import Source
from matplotlib.pylab import rcParams
import matplotlib.lines as mlines
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
import plotly.express as px
import scipy.cluster.hierarchy as sch
from sklearn.metrics import classification_report


# STANDARD MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ENSEMBLE
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# XGBOOST
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import xgboost as xgb

# CLUSTERING
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# PICKLE
from pickle import dump
from pickle import load

from datetime import datetime



start_time = time.time()
# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])


# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
# db = SQLAlchemy(app)

# Import + Register Blueprints
# from app.accounts import accounts as accounts
# app.register_blueprint(accounts)
# from app.irsystem import irsystem as irsystem
# app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

# HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/filters', methods=['GET'])
def filters():

    def str_to_bool(s):
        if s == 'true':
             return True
        elif s == 'false':
             return False

    # text inputs
    host_response_rate = float(request.args.get("host_response_rate"));
    date = request.args.get("date");
    datetimeobject = datetime.strptime(date,'%Y-%m-%d')
    date = str(datetimeobject.strftime('%-m/%-d/%Y'))

    security_deposit = float(request.args.get("security_deposit"));
    cleaning_fee = float(request.args.get("cleaning_fee"));
    bedrooms = float(request.args.get("bedrooms"));
    beds = float(request.args.get("beds"));
    bathrooms = float(request.args.get("bathrooms"));
    accomodates = float(request.args.get("accomodates"));

    data = pd.read_csv("Data/listings_8.csv")
    scaler = MinMaxScaler(feature_range=(0,1))

    sec = data[["security_deposit"]].copy()
    df2 = pd.DataFrame([[security_deposit]], columns=['security_deposit'])
    sec = sec.append(df2)
    sec = scaler.fit_transform(sec)
    security_deposit = float(sec[-1:])

    clean = data[["cleaning_fee"]].copy()
    df2   = pd.DataFrame([[cleaning_fee]], columns=['cleaning_fee'])
    clean = clean.append(df2)
    clean = scaler.fit_transform(clean)
    cleaning_fee = float(clean[-1:])

    hrr = data[["host_response_rate"]].copy()
    df2 = pd.DataFrame([[host_response_rate]], columns=['host_response_rate'])
    hrr = hrr.append(df2)
    hrr = scaler.fit_transform(hrr)
    host_response_rate = float(hrr[-1:])

    # binary inputs
    description = str_to_bool(request.args.get("description"));
    transit_bool = str_to_bool(request.args.get("transit_bool"));
    host_about_bool = str_to_bool(request.args.get("host_about_bool"));
    nopets_bool = str_to_bool(request.args.get("nopets_bool"));
    nosmok_bool = str_to_bool(request.args.get("nosmok_bool"));
    host_identity_verified = str_to_bool(request.args.get("host_identity_verified"));
    is_location_exact = str_to_bool(request.args.get("is_location_exact"));
    Wheelchair_bool = str_to_bool(request.args.get("Wheelchair_bool"));
    TV_bool = str_to_bool(request.args.get("TV_bool"));
    Hair_Dryer_bool = str_to_bool(request.args.get("Hair_Dryer_bool"));
    twentyfour_Hour_Check_in_bool = str_to_bool(request.args.get("twentyfour_Hour_Check_in_bool"));
    Doorman_bool = str_to_bool(request.args.get("Doorman_bool"));
    Kitchen_bool = str_to_bool(request.args.get("Kitchen_bool"));
    Smoke_Detector_bool = str_to_bool(request.args.get("Smoke_Detector_bool"));
    Heating_bool = str_to_bool(request.args.get("Heating_bool"));
    Clothes_Dryer_bool = str_to_bool(request.args.get("Clothes_Dryer_bool"));
    Clothes_Washer_bool = str_to_bool(request.args.get("Clothes_Washer_bool"));
    Pets_live_in_flat_bool = str_to_bool(request.args.get("Pets_live_in_flat_bool"));
    Free_Parking_bool = str_to_bool(request.args.get("Free_Parking_bool"));
    Internet_bool = str_to_bool(request.args.get("Internet_bool"));
    AC_bool = str_to_bool(request.args.get("AC_bool"));
    instant_bookable = str_to_bool(request.args.get("instant_bookable"));
    require_guest_profile_picture = str_to_bool(request.args.get("require_guest_profile_picture"));
    require_guest_phone_verification = str_to_bool(request.args.get("require_guest_phone_verification"));
    extra_price_75 = str_to_bool(request.args.get("extra_price_75"));
    Gym_bool = str_to_bool(request.args.get("Gym_bool"));
    Elevator_in_building_bool = str_to_bool(request.args.get("Elevator_in_building_bool"));

    # dropdown inputs
    neighborhood = request.args.get("neighborhood");
    property_type = request.args.get("property_type");
    room_type = request.args.get("room_type");
    # accomodates = request.args.get("accomodates");
    # bedrooms = request.args.get("bedrooms");
    # bathrooms = request.args.get("bathrooms");
    # extra_guests = request.args.get("extra_guests");
    # beds = request.args.get("beds");
    response_time = request.args.get("response_time");
    cancellations = request.args.get("cancellations");


    def recommendations(coefs, airbnb, X_wnei, data, recom):
        '''
        Makes recommendations based on the specific Airbnb's characteristics and the important features recognized by
        the best model in best_models()

        '''

        X_wnei_cols = X_wnei.columns

        for_improvement = []
        for col in range(1,len(X_wnei.columns)):
            if coefs[col] > 0 :
                if airbnb[0][col] < data.iloc[:,col].mean():
                    if "cancellations" not in recom.iloc[0, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                        if "messages" not in recom.iloc[0, col] and ("You should answer messages sooner" not in for_improvement):
                            sent = recom.iloc[0, col]
                            for_improvement.append(sent)
            elif coefs[col] < 0:
                if airbnb[0][col] > data.iloc[:,col].mean():
                    if "cancellations" not in recom.iloc[1, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                        if "messages" not in recom.iloc[1, col] and ("You should answer messages sooner" not in for_improvement):
                            sent = recom.iloc[1, col]
                            for_improvement.append(sent)

    #     print(for_improvement)
        # If there are improvements to make...
        if len(for_improvement) != 0:
        #   Concatenating messages
            mess1 = "In order for your Airbnb to be truly competitive, make sure you do the following things: \n"
            mess2 = "\n -- ".join(for_improvement)
            message = mess1 + "\n" + mess2

            return message
        # If there are NO improvements to make...
        else:
            return "Your Airbnb is very competitive compared to all the other Airbnbs in Boston!"




    def best_models(data):
        '''
        Implements Ridge, Lasso and ElasticNet to determine best model, based on inputted data.

        '''
        MSEs = []


        data=data.dropna()

        data=data[data["number_of_reviews"]>=1]
        data=data[data["guests_included"]>=1]

        data['price_per_person'] = data['price_per_night']/(data['guests_included']+data['extra_people'])

        data['output'] = (data['number_of_reviews']*data['review_scores_rating'])/data['availability_365']

        data = data.drop(["property_type","room_type","description","house_rules", "amenities", "id",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_value", "number_of_reviews",
                          "reviews_per_month", "availability_365", "neighbourhood_cleansed","bathrooms",
                          "host_is_superhost", "accommodates", "bedrooms","beds", "guests_included","price_per_night",
                          "extra_people", "host_is_superhost", "Kitchen_boolean", "Gym_bool", "Elevator_in_building_bool",
                          "Clothes_Washer_bool", "Internet_bool", "price_per_person"],axis=1)

        scaler=MinMaxScaler(feature_range=(0,1))


        data[['security_deposit']] = scaler.fit_transform(data[['security_deposit']])
        data[['cleaning_fee']] = scaler.fit_transform(data[['cleaning_fee']])
        data[['host_response_rate']] = scaler.fit_transform(data[['host_response_rate']])
    #     data[['price_per_person']] = scaler.fit_transform(data[['price_per_person']])

        data.to_numpy()
        data=pd.DataFrame(data)

        data=data.dropna()

        data['output'][data['output'] > 260] = 260

        y = data["output"] # Target variable (price)
        X_wnei = data.drop(["output"],axis=1)

        # Creating new DF without neighborhood names
        X_wnei.to_csv("X_wnei.csv", index=False)
        data.to_csv("data.csv", index=False)

        #########################################################################################################
        # Ridge Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=Ridge()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("Ridge Regression", results.mean(), clf.coef_))


        #########################################################################################################
        # Lasso Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=Lasso()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("Lasso Regression", results.mean(), clf.coef_))

        #########################################################################################################
        # Elastic Net Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=ElasticNet()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("ElasticNet Regression", results.mean(), clf.coef_))
    #     print(MSEs)

        #########################################################################################################

        return (min(MSEs, key = lambda t: t[1])[0], X_wnei, y, min(MSEs, key = lambda t: t[1])[2], data)



    #########################################################################################################
    # #############    MAIN

    data = pd.read_csv("Data/listings_8.csv")
    recom = pd.read_csv("Data/recom.csv")
    best_model = best_models(data)

    method = best_model[0]
    X_wnei = best_model[1]
    y = best_model[2]
    coefs = best_model[3].reshape(33,1)
    data = best_model[4]

    airbnb = pd.read_csv("X_wnei.csv").iloc[2].values.reshape(1,-1)

    a = {  "transit_bool": [transit_bool],
            "nopets_bool":[nopets_bool],
            "nosmok_bool":[nosmok_bool],
            "host_about_bool":[host_about_bool]}

    df = pd.DataFrame(a)

    if response_time=="1Hour":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [True]
        df["fewhours_response_time"] = [False]
    elif response_time=="Hours":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [True]
    elif response_time=="1Day":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [True]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [False]
    elif response_time=="Days":
        df["fewdays_response_time"] = [True]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [False]

    df["host_response_rate"] = [host_response_rate]
    df["host_identity_verified"] = [host_identity_verified]
    df["is_location_exact"] = [is_location_exact]
    df["Wheelchair_bool"] = [Wheelchair_bool]
    df["TV_bool"] = [TV_bool]
    df["Hair Dryer_bool"] = [Hair_Dryer_bool]
    df["24-Hour_Check-in_bool"] = [twentyfour_Hour_Check_in_bool]
    df["Doorman_bool"] = [Doorman_bool]
    df["Kitchen_bool"] = [Kitchen_bool]
    df["Smoke_Detector_bool"] = [Smoke_Detector_bool]
    df["Clothes_Washer_bool"] = [Clothes_Dryer_bool]
    df["Pets_live_in_flat_bool"] = [Pets_live_in_flat_bool]
    df["Free_Parking_bool"] = [Free_Parking_bool]
    df["Heating_bool"] = [Heating_bool]
    df["Wireless_Internet_bool"] = [Internet_bool]
    df["AC_bool"] = [AC_bool]
    df["security_deposit"] = [security_deposit]
    df["cleaning_fee"] = [cleaning_fee]
    df["instant_bookable"] = [instant_bookable]

    if cancellations=="SuperStrict":
        df["super_strict_canc"] = [True]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Moderate":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [True]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Strict":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [True]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Flexible":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [True]

    df["require_guest_profile_picture"] = [require_guest_profile_picture]
    df["require_guest_phone_verification"] = [require_guest_phone_verification]

    df = df.to_numpy()
    airbnb = df

    # [print(type(df))]
    # [print(type(airbnb))]
    # [print(df.shape)]
    # [print(airbnb.shape)]
    # print(df)
    # print(airbnb)

    # print(recommendations(coefs, df, X_wnei, data, recom))

    improvements = recommendations(coefs, airbnb, X_wnei, data, recom)
    # print(improvements)

#############################################################################
############### PRICING #####################################################
#############################################################################

    def split(data):
        '''
        Clean the data and return an array for the target variable and all the input variables
        '''
        data=data.dropna()

        data=data[data["price_per_night"]>=12]


        data=data[data["price_per_night"]<850]
        data=data[data["number_of_reviews"]>=1]
        data=data[data["guests_included"]>=1]



        data['extra_price']=data['security_deposit']+data['cleaning_fee']+data['extra_people']
        data['extra_price'].describe()


        for test in data['extra_price']:

            if test<25:
                data['extra_price']=25


            if 25<test and test<100:
                data['extra_price']=75


            if 100<test and test<235:
                data['extra_price']=125


            if 235<test:
                data['extra_price']=235

        scaler=MinMaxScaler(feature_range=(0,1))
        data[['extra_price']] = scaler.fit_transform(data[['extra_price']])

    #     data[['bathrooms']] = scaler.fit_transform(data[['bathrooms']])
    #     data[['bedrooms']] = scaler.fit_transform(data[['bedrooms']])
    #     data[['beds']] = scaler.fit_transform(data[['beds']])
    #     data[['accommodates']] = scaler.fit_transform(data[['accommodates']])

        for categorical_feature in ['neighbourhood_cleansed',"property_type","room_type"]:
            data = pd.concat([data,pd.get_dummies(data[categorical_feature], prefix=categorical_feature, prefix_sep='_',)], axis=1)


        data= data.drop(['neighbourhood_cleansed', 'property_type', 'room_type','description','host_about_bool',
                        'amenities',"house_rules","review_scores_cleanliness","review_scores_rating",'review_scores_accuracy',
                        'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value',
                         'id',"property_type", "host_response_rate", 'reviews_per_month','number_of_reviews'
                         ,'security_deposit','cleaning_fee','extra_people','extra_price','availability_365',"guests_included"
                         ,"host_is_superhost","transit_bool", "nosmok_bool", "fewdays_response_time", 'fewdays_response_time',
                         '1day_response_time', '1hour_response_time', 'fewhours_response_time', 'host_identity_verified',
                         'Internet_bool', 'super_strict_canc', 'moderate_cancellation', 'strict_cancellation', 'flexible_cancellation',
                         'require_guest_profile_picture', 'require_guest_phone_verification','Kitchen_boolean','Clothes_Dryer_bool'
                        ],axis=1)

        # Target variable (price_per_night)

        y = data["price_per_night"]
        data=data.drop(["price_per_night"],axis=1)

        #guests included to re
        data_1=data[["bathrooms","bedrooms","beds","accommodates"]]
        data=data.drop(["bathrooms","bedrooms","beds","accommodates"],axis=1)

        data=data.astype('bool')

        data=pd.concat([data, data_1], axis=1, sort=False)

        X = data

    #     print(y)
    #     print(X)

        return (X,y)

    def XGB(X,y,airbnbPricing,date):
        '''
        Calculating RMSE with XGBoost Hyperparameters

        '''
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.2,
                    max_depth = 7, alpha = 20, n_estimators = 70)

        xg_reg.fit(X_train,y_train)

    #     preds = xg_reg.predict(X_test)
    #     rmse = np.sqrt(mean_squared_error(y_test, preds))

        pricePred = xg_reg.predict(airbnbPricing)

        print(date)

        # If SPECIAL DAY increase price by 1.5
        if date in ["10/7/2020", "10/8/2020", "10/9/2020", "10/10/2020", "10/14/2020", "10/15/2020", "10/16/2020", "10/17/2020",
         "10/18/2020", "10/21/2020", "10/22/2020", "10/23/2020", "10/26/2020", "10/27/2020", "10/28/2020", "10/29/2020",
         "4/13/2020", "4/14/2020", "4/15/2020", "4/16/2020", "4/17/2020", "4/18/2020", "4/19/2020", "4/20/2020", "4/21/2020",
         "4/22/2020", "6/8/2020", "6/9/2020", "6/10/2020", "6/11/2020", "6/12/2020", "6/13/2020", "6/14/2020", "6/15/2020",
         "6/16/2020", "6/17/2020", "6/30/2020", "7/1/2020", "7/7/2020", "7/8/2020", "7/12/2020", "7/13/2020", "7/14/2020",
         "7/15/2020", "7/17/2020", "7/21/2020", "7/22/2020", "7/24/2020", "7/25/2020", "7/26/2020", "7/27/2020", "7/28/2020",
         "7/29/2020", "7/30/2020", "7/31/2020", "8/1/2020", "8/2/2020", "8/4/2020", "8/5/2020", "8/10/2020", "8/11/2020",
         "8/12/2020", "8/14/2020", "8/16/2020", "8/17/2020", "8/18/2020", "8/19/2020", "8/21/2020", "8/23/2020", "8/25/2020",
         "8/26/2020", "8/28/2020", "8/29/2020", "9/1/2020", "9/2/2020", "9/3/2020", "9/4/2020", "9/5/2020"]:
            pricePred=pricePred*1.5

        # If WEEKEND increase price by 1.2
        if date in ["1/3/2020", "1/4/2020", "1/10/2020", "1/11/2020", "1/17/2020", "1/18/2020", "1/24/2020",
        "1/25/2020", "1/31/2020", "2/1/2020", "2/7/2020", "2/8/2020", "2/14/2020", "2/15/2020", "2/21/2020",
        "2/22/2020", "2/28/2020", "2/29/2020", "3/6/2020", "3/7/2020", "3/13/2020", "3/14/2020", "3/20/2020",
        "3/21/2020", "3/27/2020", "3/28/2020", "4/3/2020", "4/4/2020", "4/10/2020", "4/11/2020", "4/17/2020",
        "4/18/2020", "4/24/2020", "4/25/2020", "5/1/2020", "5/2/2020", "5/8/2020", "5/9/2020", "5/15/2020",
        "5/16/2020", "5/22/2020", "5/23/2020", "5/29/2020", "5/30/2020", "6/5/2020", "6/6/2020", "6/12/2020",
        "6/13/2020", "6/19/2020", "6/20/2020", "6/26/2020", "6/27/2020", "7/3/2020", "7/4/2020", "7/10/2020",
        "7/11/2020", "7/17/2020", "7/18/2020", "7/24/2020", "7/25/2020", "7/31/2020", "8/1/2020", "8/7/2020",
        "8/8/2020", "8/14/2020", "8/15/2020", "8/21/2020", "8/22/2020", "8/28/2020", "8/29/2020", "9/4/2020",
        "9/5/2020", "9/11/2020", "9/12/2020", "9/18/2020", "9/19/2020", "9/25/2020", "9/26/2020", "10/2/2020",
        "10/3/2020", "10/9/2020", "10/10/2020", "10/16/2020", "10/17/2020", "10/23/2020", "10/24/2020",
        "10/30/2020", "10/31/2020", "11/6/2020", "11/7/2020", "11/13/2020", "11/14/2020", "11/20/2020",
        "11/21/2020", "11/27/2020", "11/28/2020", "12/4/2020", "12/5/2020", "12/11/2020", "12/12/2020",
        "12/18/2020", "12/19/2020", "12/25/2020", "12/26/2020"]:
            pricePred=pricePred*1.2


    #     print("RMSE: %f" % (rmse))
        return pricePred[0]


    data = pd.read_csv("Data/listings_8.csv")
    X,y= split(data)



    neighborhoods = X.iloc[:,18:43].columns

    a = {   'neighbourhood_cleansed_Allston': [False],
            'neighbourhood_cleansed_Back Bay': [False],
            'neighbourhood_cleansed_Bay Village': [False],
            'neighbourhood_cleansed_Beacon Hill': [False],
            'neighbourhood_cleansed_Brighton': [False],
            'neighbourhood_cleansed_Charlestown': [False],
            'neighbourhood_cleansed_Chinatown': [False],
            'neighbourhood_cleansed_Dorchester': [False],
            'neighbourhood_cleansed_Downtown': [False],
            'neighbourhood_cleansed_East Boston': [False],
            'neighbourhood_cleansed_Fenway': [False],
            'neighbourhood_cleansed_Hyde Park': [False],
            'neighbourhood_cleansed_Jamaica Plain': [False],
            'neighbourhood_cleansed_Leather District': [False],
            'neighbourhood_cleansed_Longwood Medical Area': [False],
            'neighbourhood_cleansed_Mattapan': [False],
            'neighbourhood_cleansed_Mission Hill': [False],
            'neighbourhood_cleansed_North End': [False],
            'neighbourhood_cleansed_Roslindale': [False],
            'neighbourhood_cleansed_Roxbury': [False],
            'neighbourhood_cleansed_South Boston': [False],
            'neighbourhood_cleansed_South Boston Waterfront': [False],
            'neighbourhood_cleansed_South End': [False],
            'neighbourhood_cleansed_West End': [False],
            'neighbourhood_cleansed_West Roxbury': [False]}
    dfneighborhoods = pd.DataFrame(a)

    for i in neighborhoods:
        i = i.split("_")[2].replace(" ", "")
        if i == neighborhood:
            i = re.sub( r"([A-Z])", r" \1", i).split()
            i = " ".join(i)
            name = "neighbourhood_cleansed_" + i
            dfneighborhoods[name] = [True]

    properties = X.iloc[:,43:55].columns
    a = {   'property_type_Apartment': [False],
            'property_type_Bed & Breakfast': [False],
            'property_type_Boat': [False],
            'property_type_Condominium': [False],
            'property_type_Dorm': [False],
            'property_type_Entire Floor': [False],
            'property_type_Guesthouse': [False],
            'property_type_House': [False],
            'property_type_Loft': [False],
            'property_type_Other': [False],
            'property_type_Townhouse': [False],
            'property_type_Villa': [False]}
    dfproperties = pd.DataFrame(a)

    for i in properties:
        i = i.split("_")[2].replace(" ", "")
        if i == property_type:
            i = re.sub( r"([A-Z])", r" \1", i).split()
            i = " ".join(i)
            name = "property_type_" + i
            dfproperties[name] = [True]


    rooms = X.iloc[:,55:58].columns
    a = {   'room_type_Entire home/apt': [False],
            'room_type_Private room': [False],
            'room_type_Shared room': [False]}
    dfrooms = pd.DataFrame(a)

    for i in rooms:
        i = i.split("_")[2].replace(" ", "")
        if i == room_type:
            i = re.sub( r"([A-Z])", r" \1", i).split()
            i = " ".join(i)
            name = "property_type_" + i
            dfrooms[name] = [True]



    pricing = { "nopets_bool":[nopets_bool],
                "is_location_exact":[is_location_exact],
                "Wheelchair_bool":[Wheelchair_bool],
                "TV_bool":[TV_bool],
                "Hair Dryer_bool":[Hair_Dryer_bool],
                "24-Hour_Check-in_bool":[twentyfour_Hour_Check_in_bool],
                "Doorman_bool":[Doorman_bool],
                "Gym_bool":[Gym_bool],
                "Kitchen_bool":[Kitchen_bool],
                "Smoke_Detector_bool":[Smoke_Detector_bool],
                "Elevator_in_building_bool":[Elevator_in_building_bool],
                "Pets_live_in_flat_bool":[Pets_live_in_flat_bool],
                "Free_Parking_bool":[Free_Parking_bool],
                "Heating_bool":[Heating_bool],
                "Clothes_Washer_bool":[Clothes_Washer_bool],
                "Wireless_Internet_bool":[Internet_bool],
                "AC_bool":[AC_bool],
                "instant_bookable":[instant_bookable]}

    df = pd.DataFrame(pricing)

    df = pd.concat([df, dfneighborhoods], axis=1, sort=False)
    df = pd.concat([df, dfproperties], axis=1, sort=False)
    df = pd.concat([df, dfrooms], axis=1, sort=False)
    df["bathrooms"] = bathrooms
    df["bedrooms"] = bedrooms
    df["beds"] = beds
    df["accommodates"] = accomodates

    pred = XGB(X,y,df,str(date))

    print(improvements+";"+str(pred))

    return (improvements+";"+str(pred))



end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed, "seconds")
