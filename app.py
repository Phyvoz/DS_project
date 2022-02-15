import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

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
        st.pyplot(fig) """ #not going to run these lines because they take a lot of time to execute












