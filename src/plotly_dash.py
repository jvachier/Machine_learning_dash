from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
 

import xgboost as xgb 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Diseases prediction from Genes"

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

markdown_intro = '''
# Diseases prediction from Genes

This challenge consists of the establishment of
relations between genes and a set of diseases,
based on our current knowledge of the human genome.

## Outline
- Data
- Exploration
- Results
'''

markdown_data = '''
## Data
'''

markdown_pre_processing = '''
## Exploration

- Model(s) & ROC curves
- Features Importance
'''

markdown_post_processing = '''
## Post-processing

- Normalization & Scaling
- Main features 
'''


markdown_results= '''
## Results
'''

markdown_info_targets='''
* CKD = Chronic kidney disease
* COPD = Chronic obstructive pulmonary disease
'''



def generate_table(dataframe, max_rows=5):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

 

## Figure
df_train = pd.read_csv('./data/train.csv')
targets = df_train[['breast_cancer_target','lung_cancer_target','CKD_target','COPD_target']]
feature_columns = df_train.drop(columns=targets.columns)
feature_columns = feature_columns.drop(columns=df_train.columns[0])


better_name = {
    'breast_cancer_target':'Breast Cancer',
    'lung_cancer_target':'Lung Cancer',   
    'CKD_target':'CKD',    
    'COPD_target':'COPD',  
}

fig_RF_WON = go.Figure()
fig_RF_WON.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for i in targets.columns:
    X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets[i], test_size=0.2, random_state = 1)
    rdf = RandomForestClassifier(n_estimators = 50, max_depth = 5, random_state = 1).fit(X_train,y_train)
    y_score = rdf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)
    fig_RF_WON.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{better_name[i]} - AUC=({score:.4f})', mode='lines'))

fig_RF_WON.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate')

fig_RF_WON.update_layout(title="ROC curve for the different Targets - Random Forest without Normalization")


fig_RF_WN = go.Figure()
fig_RF_WN.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

numeric_x_vars = feature_columns.select_dtypes(include='number').columns.tolist()
numeric_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(feature_columns[numeric_x_vars]),
    columns=numeric_x_vars
)

for i in targets.columns:
    X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets[i], test_size=0.2, random_state = 1)
    rdf = RandomForestClassifier(n_estimators = 50, max_depth = 5, random_state = 1).fit(X_train,y_train)
    y_score = rdf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)
    fig_RF_WN.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{better_name[i]} - AUC=({score:.4f})', mode='lines'))

fig_RF_WN.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate')

fig_RF_WN.update_layout(title="ROC curve for the different Targets - Random Forest with Normalization")


## Feature Engineering: Normalization and Selection
### Without Norm
fig_LG_WON = go.Figure()
fig_LG_WON.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for i in targets.columns:
    X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets[i], test_size=0.2, random_state = 1)
    rdf = LogisticRegression(max_iter=1000, random_state=1, n_jobs=4).fit(X_train,y_train)
    y_score = rdf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)
    fig_LG_WON.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{better_name[i]} - AUC=({score:.4f})', mode='lines'))

fig_LG_WON.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate')

fig_LG_WON.update_layout(title="ROC curve for the different Targets - Logictic Regression without Normalization")



### With Norm
fig_LG_WN = go.Figure()
fig_LG_WN.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

numeric_x_vars = feature_columns.select_dtypes(include='number').columns.tolist()
numeric_scaled = pd.DataFrame(
    MinMaxScaler().fit_transform(feature_columns[numeric_x_vars]),
    columns=numeric_x_vars
)

for i in targets.columns:
    X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets[i], test_size=0.2, random_state = 1)
    rdf = LogisticRegression(max_iter=1000, random_state=1, n_jobs=4).fit(X_train,y_train)
    y_score = rdf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)
    fig_LG_WN.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{better_name[i]} - AUC=({score:.4f})', mode='lines'))

fig_LG_WN.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate')

fig_LG_WN.update_layout(title="ROC curve for the different Targets - Logictic Regression with Normalization")



## Correlations
corr_pearson = feature_columns.corr(method='pearson')
corr_spearman = feature_columns.corr(method='spearman')

fig4 = go.Figure()
fig4.add_trace(
    go.Heatmap(
        x = corr_pearson.columns,
        y = corr_pearson.index,
        z = np.array(corr_pearson),
        text=corr_pearson.values,
        texttemplate='%{text:.2f}'
    )
)
fig4.layout.height = 800
fig4.layout.width = 800
fig4.update_layout(title="Pearson Correlation")
fig4.update_layout(
    xaxis = {
     'tickvals': list(range(63)),
     'ticktext': corr_pearson.index.str.slice(0,4).tolist(),
    }
)
fig4.update_layout(
    yaxis = {
     'tickvals': list(range(63)),
     'ticktext': corr_pearson.index.str.slice(0,4).tolist(),
    }
)


fig5 = go.Figure()
fig5.add_trace(
    go.Heatmap(
        x = corr_spearman.columns,
        y = corr_spearman.index,
        z = np.array(corr_spearman),
        text=corr_spearman.values,
        texttemplate='%{text:.2f}', 
    )
)
fig5.layout.height = 800
fig5.layout.width = 800
fig5.update_layout(title="Spearman Correlation")
fig5.update_layout(
    xaxis = {
     'tickvals': list(range(63)),
     'ticktext': corr_spearman.index.str.slice(0,4).tolist(),
    }
)
fig5.update_layout(
    yaxis = {
     'tickvals': list(range(63)),
     'ticktext': corr_spearman.index.str.slice(0,4).tolist(),
    }
)

 
df_train = pd.read_csv('./data/train.csv')
targets = df_train[['breast_cancer_target','lung_cancer_target','CKD_target','COPD_target']]
feature_columns = df_train.drop(columns=targets.columns)

MODELS = {'Random Forest': RandomForestClassifier,
            'Logistic Regression': LogisticRegression,
            'Gradient Boosting Classifier': GradientBoostingClassifier,
            'Stochastic Gradient Descent Classifier':SGDClassifier,
            'Extreme Gradient Boosting': xgb.XGBClassifier}


TARGETS = {'breast cancer': 'breast_cancer_target',
          'lung cancer': 'lung_cancer_target',
          'CKD': 'CKD_target',
          'COPD': 'COPD_target'}

app.layout = html.Div([
    html.Div([
        dcc.Markdown(children=markdown_intro)
    ]),

    html.Div([
        dcc.Markdown(children=markdown_data)
    ]),

    html.Div([
        html.H4(children='Gene Name & Features'),
        generate_table(feature_columns)
    ]),
    html.Div([
        html.H4(children='Targets'),
        dcc.Markdown(children=markdown_info_targets),
        generate_table(targets)
    ]),

    html.Div([
        dcc.Markdown(children=markdown_pre_processing)
    ]),

    html.Div([
        html.H4("Analysis of the ML model's results using ROC and PR curves"),
        html.Div([
            html.P("Select model:"),
            dcc.Dropdown(
                id='model1',
                options=[
                    {'label': "Random Forest", 'value': "Random Forest"},
                    {'label': "Logistic Regression", 'value': "Logistic Regression"},
                    {'label': "Gradient Boosting Classifier", 'value': "Gradient Boosting Classifier"},
                    {'label': "Stochastic Gradient Descent Classifier", 'value': "Stochastic Gradient Descent Classifier"},
                    {'label': "Extreme Gradient Boosting", 'value': "Extreme Gradient Boosting"},
                ],
                value='Random Forest',
                clearable=False,
                multi=False
            ),
        ]),
        html.Div([
            html.P("Select targets:"),
            dcc.Dropdown(
                id='target',
                options=[
                    {'label': "breast cancer", 'value': "breast cancer"},
                    {'label': 'lung cancer', 'value': 'lung cancer'},
                    {'label': 'CKD', 'value': 'CKD'},
                    {'label': 'COPD', 'value': 'COPD'},
                ],
                value="breast cancer",
                clearable=False,
                multi=False
            ),
        ]),

        dcc.Graph(id="my_graph_1"),
        dcc.Graph(id="my_graph_2"),
        dcc.Graph(id="my_graph_3"),
        dcc.Graph(id="my_graph_4"),
        dcc.Graph(figure=fig4),
        dcc.Graph(figure=fig5),
        dcc.Graph(figure=fig_RF_WON),
        dcc.Graph(figure=fig_RF_WN),
        dcc.Graph(figure=fig_LG_WON),
        dcc.Graph(figure=fig_LG_WN),
    ]),
    html.Div([
        dcc.Markdown(children=markdown_results)
    ]),

    html.Div([
        html.Div([
            html.P("Select model:"),
            dcc.Dropdown(
                id='model2',
                options=[
                    {'label': "Random Forest", 'value': "Random Forest"},
                    {'label': "Logistic Regression", 'value': "Logistic Regression"},
                    {'label': "Gradient Boosting Classifier", 'value': "Gradient Boosting Classifier"},
                    {'label': "Stochastic Gradient Descent Classifier", 'value': "Stochastic Gradient Descent Classifier"},
                    {'label': "Extreme Gradient Boosting", 'value': "Extreme Gradient Boosting"},
                ],
                value='Random Forest',
                clearable=False,
                multi=False
            ),
        ]),
        dcc.Graph(id="my_graph_5"),
        dash_table.DataTable(id="output"),
        dcc.Graph(id="heatmap_corr"),
    ]),
    
])



@app.callback(
    Output("my_graph_1", "figure"),
    Output("my_graph_2", "figure"),
    Output("my_graph_3", "figure"),
    Output("my_graph_4", "figure"),
    [Input('model1', "value"),
        Input('target', "value")])


def train_and_display(model_name,target_name):
    df_train = pd.read_csv('./data/train.csv')
    targets = df_train[['breast_cancer_target','lung_cancer_target','CKD_target','COPD_target']]
    feature_columns = df_train.drop(columns=targets.columns)
    feature_columns = feature_columns.drop(columns=df_train.columns[0])

    target = TARGETS[target_name]


    numeric_x_vars = feature_columns.select_dtypes(include='number').columns.tolist()
    numeric_scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(feature_columns[numeric_x_vars]),
        columns=numeric_x_vars
    )


    #X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets[target], random_state = 1)
    X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets[target], random_state = 1)

    if model_name == "Random Forest":
        model = MODELS[model_name](n_estimators = 25, max_depth = 8, max_features= 'sqrt', random_state = 1, n_jobs=4)
    if model_name == "Logistic Regression":
        model = MODELS[model_name](max_iter=1000, random_state=1, n_jobs=4)
    if model_name == "Gradient Boosting Classifier":
        model = MODELS[model_name](n_estimators=50, learning_rate=1e-6, max_depth=5, random_state=1)
    if model_name == "Stochastic Gradient Descent Classifier":
        model = MODELS[model_name](loss='log_loss', max_iter=1000, random_state=1, n_jobs=4)
    if model_name == "Extreme Gradient Boosting":
        model = MODELS[model_name]()

    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    score = metrics.auc(fpr, tpr)

    pi = permutation_importance(model, X_test, y_test,n_repeats=30,random_state=0)

# - The values at the top of the table are the most important features in our model, while those at the bottom matter least.
# - The first number in each row indicates how much model performance decreased with random shuffling, using the same performance metric as the model (in this case, R2 score).
# - The number after the ± measures how performance varied from one-reshuffling to the next, i.e., degree of randomness across multiple shuffles.
# - Negative values for permutation importance indicate that the predictions on the shuffled (or noisy) data are more accurate than the real data.
# This means that the feature does not contribute much to predictions (importance close to 0), but random chance caused the predictions on 
# shuffled data to be more accurate. This is more common with small datasets.



    forest_importances = pd.Series(pi.importances_mean, index=X_train.columns).sort_values(ascending=False)
    df_pi = pd.DataFrame({'Feature Name': X_train.columns, 'Importance Score':forest_importances, 'std': pi.importances_std})
    fig3 = px.bar(
        df_pi,
        x='Feature Name',
        y='Importance Score',
        error_y = 'std',
        title=f'Feature Importance - Permutation Importance'
    )

    fig3.update_layout(
        xaxis = {
        'tickvals': list(range(63)),
        'ticktext': corr_pearson.index.str.slice(0,4).tolist(),
        }
    )

    if model_name == "Random Forest":
        normalized_score = (model.feature_importances_ - min(model.feature_importances_))/(max(model.feature_importances_)-min(model.feature_importances_))
        feature_scores = pd.Series(normalized_score, index=X_train.columns).sort_values(ascending=False)
        df_importance = pd.DataFrame({'Feature Name': X_train.columns, 'Importance Score':feature_scores})
        fig4 = px.bar(
            df_importance,
            x='Feature Name',
            y='Importance Score',
            title=f'Feature Importance - Mean Decrease in Impurity (Normalized)'
        )
        fig4.update_xaxes(
            showgrid=True,
            ticklen=2
        )
        fig4.update_layout(
            xaxis = {
            'tickvals': list(range(63)),
            'ticktext': corr_pearson.index.str.slice(0,4).tolist(),
            }
        )
    else:
        fig4 = {}

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f}) - {target}',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    df_output = pd.DataFrame({'Gene Name': df_train['gene_name'][:len(y_score)],
        target: y_score,
    })
    fig2 = px.line(df_output,x='Gene Name',y=target,title=f'Prediction - {target}',)
 
    return fig, fig2, fig3, fig4




@app.callback(
    Output("my_graph_5", "figure"),
    Output('output','data'),
    Output("heatmap_corr", "figure"),
    [Input('model2', "value")])


def results(model_name):
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test_no_labels.csv')
    df_test = df_test.iloc[: , 1:]
    df_test_wt_gene_name = df_test.drop(columns='gene_name')
    targets = df_train[['breast_cancer_target','lung_cancer_target','CKD_target','COPD_target']]
    feature_columns = df_train.drop(columns=targets.columns)
    feature_columns = feature_columns.drop(columns=df_train.columns[0])


    if model_name == "Random Forest":
        model = MODELS[model_name](n_estimators = 25, max_depth = 8, max_features= 'sqrt', random_state = 1, n_jobs=4)
        X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets['breast_cancer_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_breast_cancer_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets['lung_cancer_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_lung_cancer_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets['CKD_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_CKD_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(feature_columns, targets['COPD_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_COPD_target = model.predict_proba(df_test_wt_gene_name)
    if model_name != "Random Forest":
        if model_name == "Logistic Regression":
            model = MODELS[model_name](max_iter=1000, random_state=1, n_jobs=4)
        if model_name == "Gradient Boosting Classifier":
            model = MODELS[model_name](n_estimators=50, learning_rate=1e-6, max_depth=5, random_state=1)
        if model_name == "Stochastic Gradient Descent Classifier":
            model = MODELS[model_name](loss='log_loss', max_iter=1000, random_state=1, n_jobs=4)
        if model_name == "Extreme Gradient Boosting":
            model = MODELS[model_name]()
        
        numeric_x_vars = feature_columns.select_dtypes(include='number').columns.tolist()
        numeric_scaled = pd.DataFrame(
            MinMaxScaler().fit_transform(feature_columns[numeric_x_vars]),
            columns=numeric_x_vars
        )

        X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets['breast_cancer_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_breast_cancer_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets['lung_cancer_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_lung_cancer_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets['CKD_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_CKD_target = model.predict_proba(df_test_wt_gene_name)

        X_train, X_test, y_train, y_test = train_test_split(numeric_scaled, targets['COPD_target'], test_size=0.2, random_state = 1)
        model.fit(X_train,y_train)
        y_pred_COPD_target = model.predict_proba(df_test_wt_gene_name)

    df_output = pd.DataFrame({'gene': df_test['gene_name'],
        'breast_cancer_target': y_pred_breast_cancer_target[:,1],
        'lung_cancer_target': y_pred_lung_cancer_target[:,1],
        'CKD_target': y_pred_CKD_target[:,1],
        'COPD_target': y_pred_COPD_target[:,1],
    })

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    fig.append_trace(go.Scatter(
        x=df_output['gene'],
        y=df_output['breast_cancer_target'],
        name = 'Breast Cancer',
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=df_output['gene'],
        y=df_output['lung_cancer_target'],
        name = 'Lung Cancer',
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=df_output['gene'],
        y=df_output['CKD_target'],
        name = 'CKD',
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=df_output['gene'],
        y=df_output['COPD_target'],
        name = 'COPD',
    ), row=4, col=1)


    maxValueIndex_rf = df_output[['breast_cancer_target','lung_cancer_target','CKD_target','COPD_target']].idxmax(axis=0) 
    result = df_output.loc[maxValueIndex_rf]

    lung = df_output[['gene','lung_cancer_target']].loc[df_output['lung_cancer_target'].idxmax(axis=0)]

    breast = df_output[['gene','breast_cancer_target']].loc[df_output['breast_cancer_target'].idxmax(axis=0)]
    ckd = df_output[['gene','CKD_target']].loc[df_output['CKD_target'].idxmax(axis=0)]
    copd = df_output[['gene','COPD_target']].loc[df_output['COPD_target'].idxmax(axis=0)]

    combined = pd.concat([breast,lung,ckd,copd],axis=1)
    combined = combined.T
    combined = combined.set_index('gene')
    combined = combined.fillna(0)
    combined.iloc[0,0] = 1.0
    combined.iloc[1,1] = 1.0
    combined.iloc[2,2] = 1.0
    combined.iloc[3,3] = 1.0

    fig_heatmatp_correlation = go.Figure()
    fig_heatmatp_correlation.add_trace(
        go.Heatmap(
            x = combined.columns,
            y = combined.index,
            z = np.array(combined),
            text=combined.values,
            texttemplate='%{text:.2f}', 
        )
    )

    return fig, result.to_dict(orient='records'), fig_heatmatp_correlation






if __name__ == '__main__':
    app.run_server(debug=True)