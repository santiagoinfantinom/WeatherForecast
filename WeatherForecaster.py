import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("DailyDelhiClimateTrain.csv")
#print(data.head())
#print(data.describe)

figure = px.line(data,x='date',y='meantemp',title="Mean Temp over the years")
#Other possible y's are humidity, wind_speed, meantemp. Another possible x is 'humidity
#figure.show()

#Now change the date to datetime
#print(data['date'])
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
#print(data.head( ))

#Let's check the changes over a couple of years
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,10))
plt.title("T° change in Dehli over different years")
sns.lineplot(data=data,x='month',y='meantemp',hue='year')
#plt.show()

forecast_data = data.rename(columns={"date":"ds",'meantemp':"y"})
#print(forecast_data)

#Now, lets use facebook's prophet to predict the weather
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)
plot_plotly(model, predictions) #Is currently not plotting :(