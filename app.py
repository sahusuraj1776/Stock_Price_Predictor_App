import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock ID","TCS.NS")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-40,end.month,end.day)

TCS_data = yf.download(stock,start,end)

model = load_model("IBM_1.8_200_acc.keras")

st.subheader("Stock Data")
st.write(TCS_data)

splitting_len = int(len(TCS_data)*0.7)
x_test = pd.DataFrame(TCS_data.Close[splitting_len:])

def plot_graph(values,full_data,extra_data = 0,extra_dataset=None):
    fig = plt.figure(figsize=(15,6))
    plt.plot(values,'Orange')
    plt.plot(full_data.Close,'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader(f"{stock} Original Close Price and MA for 250 days")
TCS_data['MA_for_250_days'] =   TCS_data.Close.rolling(250).mean()
st.pyplot(plot_graph(TCS_data['MA_for_250_days'],TCS_data,0))

st.subheader(f"{stock} Original Close Price and MA for 200 days")
TCS_data['MA_for_200_days'] =   TCS_data.Close.rolling(200).mean()
st.pyplot(plot_graph(TCS_data['MA_for_200_days'],TCS_data,0))

st.subheader(f"{stock} Original Close Price and MA for 100 days")
TCS_data['MA_for_100_days'] =   TCS_data.Close.rolling(100).mean()
st.pyplot(plot_graph(TCS_data['MA_for_100_days'],TCS_data,0))

st.subheader(f"{stock} Original Close Price and MA for 100 days and MA for 250 days")
TCS_data['MA_for_100_days'] =   TCS_data.Close.rolling(100).mean()
st.pyplot(plot_graph(TCS_data['MA_for_100_days'],TCS_data,1,TCS_data['MA_for_250_days']))

scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(x_test[['Close']])

x_data = []
y_data = []
count = 0
for i in range(200,len(scaled_data)):
    x_data.append(scaled_data[i-200:i])
    y_data.append(scaled_data[i])
    count = i
x_data,y_data = np.array(x_data),np.array(y_data)

predictions = model.predict(x_data)
print(predictions.shape)
inv_pred= scalar.inverse_transform(predictions)
inv_y_test = scalar.inverse_transform(y_data)

plotting_data = pd.DataFrame(
    {'Original_test_data': inv_y_test.reshape(-1),
     'Predictions': inv_pred.reshape(-1)
    },index = TCS_data.index[splitting_len+200:]
)
st.subheader(f"{stock} Original Price vs Predicted Price")
st.write(plotting_data)

st.subheader(f'{stock} Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([TCS_data.Close[:splitting_len+50],plotting_data],axis=0))
plt.legend(['Data - Not used','Original Test Data','Predicted Test Data'])
st.pyplot(fig)


new_data = scaled_data[count-199:count+1]
new_data = new_data.reshape(1,200,1)
new_data = np.array(new_data)
print(new_data.shape)
print(type(new_data))
tw_pred = model.predict(new_data)
print(tw_pred.shape)
inv_tw_pred = scalar.inverse_transform(tw_pred)
value = inv_tw_pred[0,0]
st.subheader(f"{stock} Tommorrow Prediction: {value}")