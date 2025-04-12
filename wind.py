import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
df_1=pd.read_csv("/home/xmahoragax/PYTHON/WIND/scada_data.csv")
print(df_1.head())

print(df_1.isnull().sum())
#VISUALIZATION 1
plt.figure(figsize=(10, 6))
plt.hexbin(
    df_1['WEC: max. windspeed'],  
    df_1['WEC: Production kWh'],  
    gridsize=15,  
    cmap='YlGnBu',
    mincnt=1, 
)
plt.colorbar(label='Count of Data Points')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Power Output (kWh)', fontsize=12)
plt.title('Wind Turbine: Power Output vs Wind Speed (Hexbin Plot)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()  
#VISUALIZATION 2
temp_cols = [
    'Front bearing temp.', 'Rear bearing temp.', 
    'Pitch cabinet blade A temp.', 'Nacelle temp.',
    'Transformer temp.', 'Rotor temp. 1', 'Stator temp. 1'
]

corr = df_1[temp_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, vmax=1,  
    mask=np.triu(np.ones_like(corr))  
)
plt.title('Temperature Sensor Correlations (Predictive Maintenance)', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
#VISUALIZATION 3
df_1['DateTime'] = pd.to_datetime(df_1['DateTime'])

plt.figure(figsize=(14, 6))
plt.plot(df_1['DateTime'], df_1['WEC: ava. Power'], label='WEC: Available Power', color='blue', linewidth=1)
plt.plot(df_1['DateTime'], df_1['WEC: Production kWh'], label='WEC: Production kWh', color='green', linewidth=1)
plt.plot(df_1['DateTime'], df_1['WEC: ava. windspeed'], label='WEC: Available Wind Speed', color='red', alpha=0.7, linewidth=1)

plt.title('Wind Turbine Power Production vs Wind Speed Over Time')
plt.xlabel('DateTime')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

