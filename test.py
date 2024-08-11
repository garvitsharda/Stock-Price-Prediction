gstock_data = pd.read_csv('data.csv')
gstock_data .head()

gstock_data = gstock_data [['date','open','close']] 
gstock_data ['date'] = pd.<a onclick="parent.postMessage({'referent':'.pandas.to_datetime'}, '*')">to_datetime(gstock_data ['date'].apply(lambda x: x.split()[0])) 
gstock_data .set_index('date',drop=True,inplace=True) 
gstock_data .head()

fg, ax =plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.subplots'}, '*')">subplots(1,2,figsize=(20,7))
ax[0].plot(gstock_data ['open'],label='Open',color='green')
ax[0].set_xlabel('Date',size=15)
ax[0].set_ylabel('Price',size=15)
ax[0].legend()
ax[1].plot(gstock_data ['close'],label='Close',color='red')
ax[1].set_xlabel('Date',size=15)
ax[1].set_ylabel('Price',size=15)
ax[1].legend()
fg.show()

from sklearn.preprocessing import MinMaxScaler
Ms = MinMaxScaler()
gstock_data [gstock_data .columns] = Ms.fit_transform(gstock_data )
training_size = round(len(gstock_data ) * 0.80)
train_data = gstock_data [:training_size]
test_data  = gstock_data [training_size:]

def create_sequence(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset):
  <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences = []
  <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels = []

  <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx = 0

  for <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx in range(50,len(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset)): 
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences.append(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset.iloc[<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx:<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx])
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels.append(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..dataset'}, '*')">dataset.iloc[<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..stop_idx'}, '*')">stop_idx])
    <a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..start_idx'}, '*')">start_idx += 1
  return (np.<a onclick="parent.postMessage({'referent':'.numpy.array'}, '*')">array(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..sequences'}, '*')">sequences),np.<a onclick="parent.postMessage({'referent':'.numpy.array'}, '*')">array(<a onclick="parent.postMessage({'referent':'.kaggle.usercode.22406117.81090952.create_sequence..labels'}, '*')">labels))
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)
test_inverse_predicted = MMS.inverse_transform(test_predicted)

# Merging actual and predicted data for better visualization
gs_slic_data = pd.concat([gstock_data .iloc[-202:].copy(),pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],index=gstock_data .iloc[-202:].index)], axis=1)
gs_slic_data[['open','close']] = MMS.inverse_transform(gs_slic_data[['open','close']])
gs_slic_data.head()
gs_slic_data[['open','open_predicted']].plot(figsize=(10,6))
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.xticks'}, '*')">xticks(rotation=45)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.xlabel'}, '*')">xlabel('Date',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.ylabel'}, '*')">ylabel('Stock Price',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.title'}, '*')">title('Actual vs Predicted for open price',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.show'}, '*')">show()

gs_slic_data[['close','close_predicted']].plot(figsize=(10,6))
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.xticks'}, '*')">xticks(rotation=45)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.xlabel'}, '*')">xlabel('Date',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.ylabel'}, '*')">ylabel('Stock Price',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.title'}, '*')">title('Actual vs Predicted for close price',size=15)
plt.<a onclick="parent.postMessage({'referent':'.matplotlib.pyplot.show'}, '*')">show()