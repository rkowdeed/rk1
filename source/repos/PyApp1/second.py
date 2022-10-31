#Import the libraries
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

#plt.style.use('seaborn')
df = pd.read_csv(r'C:\Users\P7167137\source\repos\PythonApplication1\GOOG_Volume_Days.csv')

df.set_index('Date').plot()
#plt.show()
print(df)
