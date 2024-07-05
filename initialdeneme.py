###selamlar efe


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

### pandas ile tabloyu çizme
filepath = "neyse"
data = pd.read_csv(filepath)
data = data.sort_values('Date') #bizdeki date variableı veya başkası neyse
data.head() #bu command bir tablo çıkartıyor

#graph çizme

sns.set_style("darkgrid")
plt.figure(figsize = (15,9))
plt.plot(data[['Close']])
plt.xticks(range(0,data.shape[0],500),data['Date'].loc[::500],rotation=45) #date var
plt.title("Microsoft Stock Price",fontsize=18, fontweight='bold')
plt.xlabel('Date',fontsize=18) 
plt.ylabel('Close Price (USD)',fontsize=18)
plt.show()