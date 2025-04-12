import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
solar_df_1 = pd.read_csv("/home/xmahoragax/PYTHON/SOLAR/solar_data.csv")
print("THE FIRST FIVE ROWS OF THE SOLAR DATASET\n")
print(solar_df_1.head())
print("DESCRIPTION OF THE DATASET\n")
print(solar_df_1.describe())
print("INFO OF THE DATASET\n")
print(solar_df_1.info())
print("DISPLAY OF THE MISSING")
print(solar_df_1.isnull().sum())
print("#-------------------------------------------------------------------------------------------------------------------------------------------------#") 
print("FILLING MISSING VALUES WITH THE RESPRECTIVE COLUMN MEDIAN VALUES\n")
solar_df_1['Cos zenith']=solar_df_1['Cos zenith'].fillna(solar_df_1['Cos zenith'].median())
solar_df_1['ETR direct']=solar_df_1['ETR direct'].fillna(solar_df_1['ETR direct'].median())
solar_df_1['Date'] = pd.to_datetime(solar_df_1['Date'], errors='coerce')  # Handle any invalid date format
solar_df_1.set_index('Date', inplace=True)
solar_df_1['Zenith (refracted)'] = solar_df_1['Zenith (refracted)'].interpolate()
print(solar_df_1.isnull().sum())
print("AS THERE ARE NO MISSING VALUES WE DON'T NEED TO PERFORM ANY OF DATA CLEANING METHODS,\n THE DATA WE HOLD IS GOOD ENOUGH TO PROCEED WITH NEXT OPERATIONS\n")
print("#---------------------------------------------------------------VISUALIZATIONS--------------------------------------------------------------------#")
#VISUALIZATION 1
x = solar_df_1['Cos zenith']
y = solar_df_1['ETR direct']

plt.figure(figsize=(10, 6))
plt.hist2d(x, y, bins=3, cmap='viridis')  
plt.xlabel('Cosine of Zenith Angle')
plt.ylabel('Solar Radiation (ETR Direct)')
plt.title('Heatmap of Solar Radiation vs. Cosine of Zenith Angle', loc = 'center', pad=20)
plt.colorbar(label='Density')
plt.show()



#VISUALIZATION 2
plt.figure(figsize=(14, 8))
features = ['ETR tilt', 'Air mass', 'Cos incidence', 'Zenith (refracted)', 'Azimuth angle']

solar_df_melt = solar_df_1[features].melt(var_name='Feature', value_name='Value')

sns.boxplot(data=solar_df_melt, x='Feature', y='Value', palette='Set2')
plt.title('Distribution of Key Solar Metrics for Predictive Maintenance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#VISUALIZATION 3
plt.figure(figsize=(10, 6))
plt.scatter(solar_df_1['Zenith (refracted)'], solar_df_1['ETR tilt'], color='orange', alpha=0.7)
plt.title('ETR Tilt vs Zenith (Refracted)')
plt.xlabel('Zenith (Refracted) Angle')
plt.ylabel('ETR Tilt (Solar Output)')
plt.grid(True)
plt.tight_layout()
plt.show()

