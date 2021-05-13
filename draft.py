import numpy as np
from sklearn.metrics import classification_report as report


a = np.array([1,0,0,2,2,1,0,0,0,0])
b = np.array([1,1,2,2,0,0,1,1,0,0])
res = np.array([100,100,-50,-50,20,-5,100,100,-30,-50])
positive = (a==1) & (b==1) | (a==1) & (b==0)
negative = (a==2) & (b==2) | (a==2) & (b==0)
print(report(a,b))

print(res[positive].sum() - res[negative].sum())
