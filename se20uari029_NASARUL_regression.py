import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,r2_score

train=pd.read_csv("/content/train_FD001.txt",sep=" ",names=columns)   ## to read txt into dataframe
test=pd.read_csv("/content/test_FD001.txt",sep=" ",names=columns)
test_result=pd.read_csv("/content/RUL_FD001.txt",sep=" ",header=None)
train.head()
test_result.head()
test_result.columns=["rul","null"]
test_result.head()
test_result.drop(["null"],axis=1,inplace=True)
test_result["id"]=test_result.index+1
test_result.head()
rul=pd.DataFrame(test.groupby('id')['cycle'].max()).reset_index()
rul.columns=['id','max']
rul.head()
test_result['rul_failed']=test_result["rul"]+rul["max"]
test_result.head()
test_result.drop(['rul'],axis=1,inplace=True)
test=test.merge(test_result,on=['id'],how='left')
test["remaining_cycle"]=test["rul_failed"]-test["cycle"]
test.head()
test.isnull().sum()
df_train=train.drop(["sensor22","sensor23"],axis=1)
df_test=test.drop(["sensor22","sensor23"],axis=1)
df_train["remaining_cycle"]= df_train.groupby(['id'])['cycle'].transform(max)-df_train['cycle']
df_train.head()
op_set=["op"+str(i) for i in range(1,4)]
sensor=["sensor"+str(i) for i in range(1,22)]
plt.style.use("seaborn-dark")
ax=sns.pairplot(test.query("cycle"),x_vars=op_set,y_vars=sensor,palette="hus1")
df_test.drop(["id","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19"],axis=1,inplace=True)
x=df_train.drop(["id","cycle","op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19","remaining_cycle"],axis=1)
y=df_train['remaining_cycle']

print('x shape :',x.shape)
print('y shape :',y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=3)
print('X_train shape : ',x_train.shape)
print('X_test shape : ',x_test.shape)
print('y_train shape : ',y_train.shape)
print('y_test shape : ',y_test.shape)
df_train
x.head()
y
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
rf_regressor=RandomForestRegressor(n_jobs=-1,n_estimators=400)
rf_regressor.fit(x_train,y_train)
y_pred_rf=rf_regressor.predict(x_test)


mae=mean_absolute_error(y_test,y_pred_rf)
mse=mean_squared_error(y_test,y_pred_rf)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred_rf)
print(f'Model : RandomForestRegressor \n  MAE : {mae}\n  MSE : {mse}\n RMSE: {rmse}\n R2 score: {r2}\n')
df_test
y_pred_rf=rf_regressor.predict(df_test.drop(["cycle","rul_failed","remaining_cycle"],axis=1))
df_excel=pd.DataFrame(y_pred_rf,columns=["RUL-predicted"])
df_excel["TRUE-RUL"]=df_test["remaining_cycle"]
plt.figure(figsize=(12,6))
plt.plot(df_test["remaining_cycle"],label='TRUE RUL')
plt.plot(y_pred_rf,label="Predicted RUL - Random Forest")
plt.xlabel('Index')
plt.ylabel('RUL')
plt.title('True and Predicted RUL')
plt.legend()
plt.show()
