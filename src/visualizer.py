import os
import plotly as p
import numpy as np
import pandas as pd
import plotly.express as px
from pandas import DataFrame
from typing import Dict, List
import matplotlib.pyplot as plt
from html2image import Html2Image
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def get_line_chart(data: Dict, x_legend: str, y_legend: str, title: str, height: int):
    """
    This function will plot a line graph using keys of dictionary (data) as x-axis and values as y-axis

    :param data: dictionary where keys are x-axis and values are y-axis
    :param x_legend: label of x-axis
    :param y_legend: label of y-axis
    :param title: title of figure
    :param height: height of figure
    """
    df = pd.DataFrame()
    df[x_legend] = list(data.keys())
    df[y_legend] = list(data.values())
    fig = px.line(df, x=x_legend, y=y_legend, template='plotly_dark', title=title, height=height)
    return fig


def get_box_plot(data: DataFrame, x_legend: str, y_legend: str, title: str, ):
    """
    This function will plot box-plot on the given data
    :return:
    """
    fig = px.box(data, x=x_legend, y=y_legend, title=title)
    return fig


def get_hist_plot(data: DataFrame, x: str, color: str, title: str, ):
    """
    This function will plot overlay histograms for fraud and non-fraud data
    :return:
    """
    fig = px.histogram(data, x=x, color=color, nbins=100)
    fig.update_layout(title=title, barmode='overlay', bargap=0.2)
    return fig


def compute_hist(data: DataFrame):
    """
    This function will count the records of target classes.

    :param data: The input dataframe
    """
    df_fraud = data[['Txn Amount', 'isFraud']]
    hist_plot = get_hist_plot(data=df_fraud, x='Txn Amount', color='isFraud', title='Histogram of Transactions')
    return hist_plot


def get_bar_chart(data: Dict, x_legend: str, y_legend: str, title: str, height: int, width: float):
    """
    This function will plot a bar chart using keys of dictionary (data) as x-axis and values as y-axis

    :param data: dictionary where keys are x-axis and values are y-axis
    :param x_legend: label of x-axis
    :param y_legend: label of y-axis
    :param title: title of figure
    :param height: height of figure
    :param width: custom width of bar
    """
    df = pd.DataFrame()
    df[x_legend] = list(data.keys())
    df[y_legend] = list(data.values())
    fig = px.bar(df, x=x_legend, y=y_legend, template='plotly_dark', title=title, height=height, width=height,
                 orientation='h')
    if width > 0:
        data["width"] = [width for _ in fig.data]
    return fig


def compute_count(data: DataFrame):
    """
    This function will count the records of target classes.

    :param data: The input dataframe
    """
    df_fraud = data.loc[data.isFraud == 'Y']
    df_non_fraud = data.loc[data.isFraud == 'N']
    fraud_data = {'0': df_non_fraud.isFraud.count(), '1': df_fraud.isFraud.count()}
    bar_chart = get_bar_chart(fraud_data, x_legend='Fraud', y_legend='Count',
                              title='Count of Legitimate / Fraudulent Transaction', height=500,
                              width=0.2)
    return bar_chart


def get_roc_curve(y_gt: List, y_pred: List, model_name: str):
    """
    This function will measure area-under-the-curve (AUC)
    :param model_name: The name of running model
    :param y_gt: The list of ground truth
    :param y_pred: The list of predicted values
    :return: It will return figure
    """
    fpr, tpr, thresholds = roc_curve(y_gt, y_pred)
    fig = px.area(x=fpr, y=tpr, title=f'{model_name} ROC Curve (AUC={auc(fpr, tpr):.4f})', width=700, height=500,
                  labels=dict(x='False Positive Rate', y='True Positive Rate'))
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")

    f_name = './tmp/{model_name}.html'
    p.offline.plot(fig, filename=f_name, auto_open=False)

    hti = Html2Image()
    hti.output_path = './tmp/'
    with open(f_name) as f:
        hti.screenshot(f.read(), save_as=f'{model_name}.png')

    return fig


def get_models_reports(y_gt: List, y_pred: List, model_name: str):
    """
    This function will
    :param y_gt: The available ground truth
    :param y_pred: The predicted labels
    :param model_name: The name of model
    :return:
    """
    print(f"Classification Report for {model_name}: \n", classification_report(y_gt, y_pred))
    print(f"Confusion Matrix of {model_name}: \n", confusion_matrix(y_gt, y_pred))


def get_coorelation_heatmap(data_corr: DataFrame):
    """
    This function will calculate the correlation between all variables and return heatmap
    :param data_corr: The input dataframe
    :return: The heatmap
    """
    # data_corr = data_corr.iloc[:, 1:]
    # data_corr = data_corr.iloc[:, :-1]
    data_corr = data_corr.corr().abs()
    x, y, z = list(data_corr.columns), list(data_corr.index), np.array(data_corr)
    fig = ff.create_annotated_heatmap(z, x=x, y=y,
                                      annotation_text=np.around(z, decimals=2),
                                      hoverinfo='z',
                                      colorscale='Viridis')
    fig.show()
    return fig


def show_bar_plot(data: DataFrame):
    """
    This function will generate bar plot
    :param data:
    :return:
    """
    df_fraud = data.loc[data.isFraud == 'Y']
    df_non_fraud = data.loc[data.isFraud == 'N']
    x = ['Legitimate', 'Fraudulent']
    h = [df_non_fraud.isFraud.count(), df_fraud.isFraud.count()]
    plt.bar(x, height=h, width=0.2)
    plt.xticks(fontsize=20) #, rotation=90)
    plt.yticks(fontsize=20)
    plt.show()


def show_histogram_plot(data: DataFrame, first_feature: str, second_feature: str, title: str, x_label: str,
                        y_label: str):
    """
    This function will plot the histogram before and after feature scaling
    :param data:
    :param first_feature:
    :param second_feature:
    :param title:
    :param x_label:
    :param y_label:
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(data[first_feature])
    ax.hist(data[second_feature])
    ax.set_title(title, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.set_xlabel(x_label, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.set_ylabel(y_label, fontdict={'fontsize': 12, 'fontweight': 'medium'})
