import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from typing import Tuple
import pickle
import numpy as np
import pandas as pd
from linearRegression import Normal

# Initialize the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    id='form-container',
    style={'fontFamily': "Arial", 
        'margin': 'auto',
        'width': '80%',
        'padding': '20px',
        'border': '2px solid #f0f0f0',
        'border-radius': '10px',
        'background-color': '#f9f9f9',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
        },
    children=[
        html.H1('Car Price Prediction', style={"textAlign": "center", "color": "#007ACC"}),
        html.H3('Instructions:', style={'color': '#555',}),
        html.P('1) In order to predict the car price, please enter the values for Engine, Mileage, Km_driven and Year in the respectively.',
               style={
                   'font-size': '16px',
                   'color': '#555',
                   'line-height': '1',
                   
        }),
        html.P('2) The prediction model will use these inputs to estimate the car price and if you do not know the max_power and mileage then it will take the default values.',
                   style={
                   'font-size': '16px',
                   'color': '#555',
                   'line-height': '1'
        }),
        html.P('3) Finally, click "Submit" button to view the predicted car price.',
                   style={
                   'font-size': '16px',
                   'color': '#555',
                   'margin-bottom': '30px',
                   'line-height': '1'
        }),
        html.Label('Insert the values to see the result.', style={'font-weight': 'bold', 'font-size': '18px'}),
        html.Br(),
        html.Br(),
        html.Label('Engine (CC):', style={'font-weight': 'bold', 'font-size': '15px'}),
        html.Br(),
        dcc.Input(id='engine', type='number', placeholder='Input the value of Engine', style={
            'width': '60%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Label('Mileage (kmpl):', style={'font-weight': 'bold', 'font-size': '15px'}),
        html.Br(),
        dcc.Input(id='mileage', type='number', placeholder='Input the value of Mileage', style={
            'width': '60%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Label('Km_driven:', style={'font-weight': 'bold', 'font-size': '15px'}),
        html.Br(),
        dcc.Input(id='km_driven', type='number', placeholder='Input the value of Km_driven', style={
            'width': '60%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Label('Year', style={'font-weight': 'bold', 'font-size': '15px'}),
        html.Br(),
        dcc.Input(id='year', type='number', placeholder='Input Year', style={
            'width': '60%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Br(),
        html.Button('Submit', id='submit', n_clicks=0, style={
            'background-color': '#007bff',
            'color': 'white',
            'padding': '10px 15px',
            'border': 'none',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-size': '16px',
            'display': 'block',
            'margin': '20px 0'
        }),
        html.Div(id='output-predict', style={
            'font-size': '16px',
            'margin-top': '20px',
            'color': '#333'
        })
    ],
)

def prediction(engine: float, mileage: float, km_driven: float ,year: float) -> float:
    try:
        no_of_cols = ['engine', 'mileage', 'km_driven', 'year']
        features = {
            'engine': engine,
            'mileage': mileage,
            'km_driven': km_driven,
            'year': year
        }
        model = pickle.load(open('model/car_A2.pkl', 'rb'))
        old_model = pickle.load(open("model/car.model",'rb'))
        scaler = pickle.load(open("model/a2_scaled.pkl",'rb'))

        data = np.array([[engine, mileage, km_driven, year]])
        X = pd.DataFrame(features, index=[0])
        X[no_of_cols]= scaler.transform(X[no_of_cols])
        prediction = np.round(np.exp(model.predict(X).values[0]),2)

        old_data = np.array([[engine, mileage, km_driven, year]])
        prediction_old_sam = old_model.predict(old_data)

        return prediction, prediction_old_sam[0] # Accessing the first element
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")
    
def getDefaultValue():
    try:
        df = pd.read_csv("./data/Cars.csv")
        df['owner'] = df['owner'].map({
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5
        })
        df = df[(df['fuel'] != 'CNG') & (df['fuel'] != 'LPG')]
        df['mileage'] = df['mileage'].str.split().str[0].astype(float)
        df['engine'] = df['engine'].str.split().str[0].str.replace('CC', '').astype(float)
        df['name'] = df['name'].str.split().str[0]
        df = df.drop(columns=['torque'])
        df = df[df['owner'] != 5]

        median_engine = df['engine'].median()
        median_year = df['year'].median()
        mean_mileage = df['mileage'].mean()
        median_km_driven = df['km_driven'].median()
        return median_year, median_engine, mean_mileage, median_km_driven
    except Exception as e:
        raise ValueError(f"Error in processing data: {str(e)}")

@app.callback(
    Output('output-predict', 'children'),
    [Input('submit', 'n_clicks')],
    [State('engine', 'value'),
     State('mileage', 'value'),
     State('km_driven', 'value'),
     State('year', 'value')]
)

def update_output(n_clicks, user_engine, user_mileage,user_km_driven, user_year):
    try:
        if n_clicks > 0:
            default_year, default_engine, default_mileage , default_km_driven = getDefaultValue()
            
            user_year = user_year if user_year else default_year
            user_engine = user_engine if user_engine else default_engine
            user_mileage = user_mileage if user_mileage else default_mileage
            user_km_driven = user_km_driven if user_km_driven else default_km_driven

            new_pred_val, old_pred_val = prediction(float(user_engine), float(user_mileage),float(user_km_driven), float(user_year))
            return f" Predicted Price: {new_pred_val:.2f} Baht and Old price: {np.exp(old_pred_val):.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"
    
    return 'Click "Submit" to view the predicted price.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8050)