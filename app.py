import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from math import sqrt


def remove_outliers(df):
    q1 = np.quantile(df.height, 0.25)
    q3 = np.quantile(df.height, 0.75)
    interquartile_range = q3 - q1
    outliers = []
    for i in df.height:
        if i > q3 + (1.5 * interquartile_range):
            outliers.append(i)
    above_05_height = df['height'] < 0.5
    df = df[above_05_height]

def lin_reg(feature):
    x = df[feature].values
    y = df['rings'].values
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    train_size = st.slider('Train Size (%):', min_value=10, max_value=90, value=80, step=10)
    train_size = train_size / 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    errors = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.text(f'Train accuracy: {model.score(x_train, y_train)*100} %')
    st.text(f'Test accuracy: {model.score(x_test, y_test)*100} %')
    st.text(f'Rooted Mean Squared Error: {errors}')
    st.text(f'R2 score: {r2}')

    with st.expander('Show Plot'):
        fig, ax = plt.subplots(figsize=(5,5))
        plt.scatter(x_train, y_train)
        plt.plot(x_test, y_pred, c='black')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Rings')
        st.write(fig)

def multi_lin_reg(df, features):
    df_reg = df.copy()
    x = df_reg[features]
    y = df_reg['rings']

    train_size = st.slider('Train Size (%):', min_value=10, max_value=90, value=70, step=10)
    train_size = train_size / 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    errors = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.text(f'Train accuracy: {model.score(x_train, y_train)*100} %')
    st.text(f'Test accuracy: {model.score(x_test, y_test)*100} %')
    st.text(f'RMSE: {errors}')
    st.text(f'R2: {r2}')

def random_forest_regressor(df, est):
    x = df.drop(['sex', 'rings'], axis=1).values
    y = df['rings'].values

    train_size = st.slider('Train Size (%):', min_value=10, max_value=90, value=60, step=10)
    train_size = train_size / 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

    model = RandomForestRegressor(n_estimators=est, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    st.text(f'Train accuracy: {model.score(x_train, y_train)*100} %')
    st.text(f'Test accuracy: {model.score(x_test, y_test)*100} %')
    st.text(f'RMSE: {sqrt(mean_squared_error(y_pred, y_test))}')


header = st.container()
with header:
    st.title('Abalone Dataset')
    st.caption('The dataset used in this project comes from the UCI machine learning repository. The original dataset was originally made for a non machine learning related study in 1994. For more information please refer to https://archive.ics.uci.edu/ml/datasets/abalone')
    st.caption('')
    st.subheader('In this study I will try different algorithms with the ultimate goal of predicting the age of the abalone from its physical measurements. Its age is usually measured by counting the number of rings via microscope.')

df = pd.read_csv('./abalone.data', delimiter=',', header=None)
df.columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']

data_exploration = st.container()
with data_exploration:
    st.header('Part 1: Exploring the data')
    st.sidebar.download_button('DOWNLOAD (.data)', df.to_csv() , file_name='abalone.data')
    show_head = st.checkbox('Show the first lines of the dataset.')
    if show_head:
        st.write(df.head())
    st.caption('This dataset does not contain Any NaN value, so there was no need for a deep cleaning.')
    st.subheader('Structure of the dataset: ')
    with st.expander('Show Info'):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    with st.expander('Show Description Matrix'):
        st.write(df.describe().T)
    st.subheader('Boxplots')
    with st.expander('"Length", "Diameter", "Height" features boxplots'):
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()
        ax.boxplot([df['length'], df['diameter'], df['height']])
        plt.xticks([1,2,3], ['Length', 'Diameter', 'Height'], rotation=20)
        st.write(fig)
        st.caption('We have quite a few outliers, but I removed only the most relevant ones to not compromise the dataset')
    with st.expander('Remaining features boxplots'):
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()
        ax.boxplot([df['whole_weight'], df['shucked_weight'], df['viscera_weight'], df['shell_weight']])
        plt.xticks([1,2,3,4], ['Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight'], rotation=20)
        st.write(fig)

    remove_outliers(df)

    st.subheader('"Sex" feature values')
    with st.expander('Show Count and Means'):
        st.write(df.sex.value_counts(), df.groupby('sex').mean())
    with st.expander('Show Pie Chart'):
        fig, ax = plt.subplots(figsize=(3,3))
        plt.pie(df.sex.value_counts().to_list(), labels=df.sex.unique().tolist(), autopct='%.2f%%', startangle=90, radius=2, explode=(0.05,0.05,0.05), shadow=True)
        plt.legend(bbox_to_anchor=(1.3,1.3))
        st.write(fig)

_="""
    st.subheader('More (secondary) Plots')
    with st.expander('Show Correlation Matrix'):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(df.corr(), annot=True, cmap='vlag')
        st.write(fig)
    with st.expander('Show Pairplot'):
        fig = sns.pairplot(df, hue='rings', palette='husl', height=1.5)
        st.pyplot(fig) """ #not going to run these lines because they take a lot of time to execute (and are pretty useless to visualize)


regression = st.container()
with regression:
    st.header('Part 2: Linear Regression')
    st.subheader('In this section we will analyse the dataset through a linear regression algorithm')
    option = st.selectbox('What feature would you like to analyse against the number of rings?', ('length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight'))
    st.write('You selected:', option)
    lin_reg(option)

DICT = {'I':1, 'M':2, 'F':3}
df['enc_sex'] = df['sex'].replace(DICT)

multiple_linear_regression = st.container()
with multiple_linear_regression:
    st.header('Multiple Linear Regression')

    features = []
    features_dict = {'Sex':'enc_sex',
    'Length':'length', 
    'Diameter':'diameter', 
    'Height':'height', 
    'Whole Weight':'whole_weight', 
    'Shucked Weight':'shucked_weight', 
    'Viscera Weight':'viscera_weight', 
    'Shell Weight':'shell_weight'}
    options = st.multiselect('Selection: ', (features_dict.keys()))
    for i in options:
        features.append(features_dict[i])
    try:
        multi_lin_reg(df, features)
    except ValueError:
        st.text('PLEASE SELECT AT LEAST ONE FEATURE')

random_forest_regression = st.container()
with random_forest_regression:
    st.header('Part 3: Random Forest Regression')
    estimators = st.slider('Select number of estimators', min_value=20, max_value=120, step=1, value=100)
    try:
        random_forest_regressor(df, est=estimators)
    except ValueError:
        st.text('PLEASE SELECT AT LEAST ONE FEATURE')

















