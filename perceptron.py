import numpy as np
import pandas as pd

data = pd.read_csv('wine.data.txt', header=None, names=['Wine type','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols', 'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])
print(data['Magnesium'])
