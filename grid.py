import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import plotly.express as px
import numpy as np
df=pd.read_csv("/home/xmahoragax/PYTHON/GRID/smart_grid_stability_augmented.csv")
#VISUALIZATION 1
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df, 
    x='tau2',  
    y='p1',    
    hue='stabf', 
    fill=True, 
    palette={'stable':'green', 'unstable':'red'},
    alpha=0.6, 
    levels=5
)
plt.title('Grid Instability Risk: tau2 vs. p1 (Density Zones)', fontsize=14)
plt.xlabel('Reaction Time (tau2)', fontsize=12)
plt.ylabel('Power Input (p1)', fontsize=12)
plt.legend(title='Stability')
plt.grid(alpha=0.2)
plt.show()'''
#VISUALIZATION 2
features = ['tau1', 'tau2', 'tau3', 'tau4',
            'p1', 'p2', 'p3', 'p4',
            'g1', 'g2', 'g3', 'g4']

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['stabf'] = df['stabf']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='stabf', palette='coolwarm', alpha=0.7)
plt.title('PCA of Grid Stability Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Stability')
plt.grid(True)
plt.tight_layout()
plt.show()

#VISUALIZATION 3
plt.figure(figsize=(14, 8))
params = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
df_melted = df[params + ['stabf']].melt(id_vars='stabf', var_name='Feature', value_name='Value')

sns.boxplot(data=df_melted, x='Feature', y='Value', hue='stabf', palette='Set2')
plt.title('Grid System Parameters by Stability (stabf)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

