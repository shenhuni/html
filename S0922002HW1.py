#!/usr/bin/env python
# coding: utf-8

# In[1]:


#------以下進行分析與評估資料------#


# In[2]:


#---載入套件---#
import pandas as pd  #載入pandas套件，命名引用名稱pd
import numpy as np  #載入numpy套件，命名引用名稱np 
import matplotlib.pyplot as plt  #載入matplotlib.pyplot套件，命名引用名稱plt
import seaborn as sns  #載入seaborn套件，命名引用名稱sns
import plotly.express as px  #載入plotly.express套件，命名引用名稱px


# In[3]:


#---讀取檔案---#
data = pd.read_csv(r'C:\Users\shen\Desktop\diabetes.csv')  #讀取csv檔，並把資料命名為'data'
data.head(10)  #顯示前10筆資料，確認有讀取成功


# In[4]:


#---檢查數據類型---#
data.dtypes  #查看資料集內的特徵資料型態


# In[5]:


#---查看資料分布筆數---#
Count = data.groupby(["Pregnancies"], as_index=False)["Pregnancies"].agg({"cnt":"count"})
print(Count)  #顯示Pregnancies

Count = data.groupby(["Glucose"], as_index=False)["Glucose"].agg({"cnt":"count"})
print(Count)  #顯示Glucose

Count = data.groupby(["BloodPressure"], as_index=False)["BloodPressure"].agg({"cnt":"count"})
print(Count)  #顯示BloodPressure

Count = data.groupby(["SkinThickness"], as_index=False)["SkinThickness"].agg({"cnt":"count"})
print(Count)  #顯示SkinThickness

Count = data.groupby(["Insulin"], as_index=False)["Insulin"].agg({"cnt":"count"})
print(Count)  #顯示Insulin

Count = data.groupby(["BMI"], as_index=False)["BMI"].agg({"cnt":"count"})
print(Count)  #顯示BMI

Count = data.groupby(["DiabetesPedigreeFunction"], as_index=False)["DiabetesPedigreeFunction"].agg({"cnt":"count"})
print(Count)  #顯示DiabetesPedigreeFunction

Count = data.groupby(["Age"], as_index=False)["Age"].agg({"cnt":"count"})
print(Count)  #顯示Age

Count = data.groupby(["Outcome"], as_index=False)["Outcome"].agg({"cnt":"count"})
print(Count)  #顯示Outcome


# In[6]:


#---移除有缺值的欄位---#
columns_to_check = ['Glucose']  #檢查Glucose
data = data.drop(data.loc[(data[columns_to_check] == 0).all(axis=1)].index)  #上述項等於0表示資料有缺，故如果檢測出某列值等於0，則刪除該列，axis=1表示刪除

columns_to_check = ['BloodPressure']  #檢查BloodPressure
data = data.drop(data.loc[(data[columns_to_check] == 0).all(axis=1)].index)  #上述項等於0表示資料有缺，故如果檢測出某列值等於0，則刪除該列，axis=1表示刪除

columns_to_check = ['SkinThickness']  #檢查SkinThickness
data = data.drop(data.loc[(data[columns_to_check] == 0).all(axis=1)].index)  #上述項等於0表示資料有缺，故如果檢測出某列值等於0，則刪除該列，axis=1表示刪除

columns_to_check = ['Insulin']  #檢查Insulin
data = data.drop(data.loc[(data[columns_to_check] == 0).all(axis=1)].index)  #上述項等於0表示資料有缺，故如果檢測出某列值等於0，則刪除該列，axis=1表示刪除

columns_to_check = ['BMI']  #檢查BMI
data = data.drop(data.loc[(data[columns_to_check] == 0.0).all(axis=1)].index)  #上述項等於0.0表示資料有缺，故如果檢測出某列值等於0.0，則刪除該列，axis=1表示刪除

data = data.reset_index(drop=True)  #刪除列資料後，需要reset資料的索引index


# In[7]:


#---檢查數值特徵分布---#
data.columns = [col for col in data.columns if data[col].dtype != 'object']  #檢查資料型態不是object的數量

plt.figure(figsize = (20,10))
plot_number = 1

for column in data.columns:
    if plot_number <= 14:
        ax = plt.subplot(3, 3, plot_number)
        sns.histplot(data[column])
        plt.xlabel(column)
    plot_number +=1
plt.tight_layout()
plt.show()


# In[8]:


#---校正資料型態(把資料型態校正回正確的資料型態)---#
data.dtypes  #查看整份資料所有欄位資料型態


# In[9]:


data.Outcome.unique() #確認預測目標之資料型態為整數(int):1,0


# In[10]:


data.head(10)  #顯示前10筆資料，確認有刪除成功


# In[11]:


#---進行描述性統計，定義畫圖的公式---#
def violin(col):  #定義提琴圖，y軸為指定資料欄位，x軸為Outcome，color根據Outcome決定圖形顏色，template表示使用plotly內建的深色主題
    fig = px.violin(data, y=col, x="Outcome", color="Outcome", box=True, template = 'plotly_dark')
    return fig.show()

def kde(col):  #定義分布估計圖，hue參數根據Outcome的欄位進行繪製，height和aspect控制圖形大小和比例，使用map將sns.kdeplot應用到指定資料欄位，並用add_legend在旁附註曲線的分類
    grid = sns.FacetGrid(data, hue="Outcome", height = 6, aspect = 2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    
def scatter(col1, col2):  #繪製散布圖，x軸為指定欄位col1，y軸為指定欄位col2，color根據Outcome決定圖形顏色，template表示使用plotly內建的深色主題
    fig = px.scatter(data, x= col1, y=col2, color="Outcome", template = 'plotly_dark')
    return fig.show()


# In[12]:


violin('Glucose') 


# In[13]:


violin('SkinThickness')


# In[14]:


violin('Insulin')


# In[15]:


violin('BMI')


# In[16]:


violin('Age')


# In[17]:


kde('Glucose')


# In[18]:


kde('Insulin')


# In[19]:


kde('Age')


# In[20]:


scatter('Glucose', 'Insulin')


# In[21]:


scatter('Glucose', 'DiabetesPedigreeFunction')


# In[22]:


scatter('Glucose', 'BMI')


# In[23]:


scatter('SkinThickness', 'BMI')


# In[24]:


#根據提琴圖、分布估計圖、散布圖針對Glucose的分析可以發現，當Glucose數值低於120，沒有得糖尿病的人占多數。
#根據提琴圖、分布估計圖、散步圖針對Insulin的分析可以發現，當Insulin數值約90時，沒有患糖尿病的人占多數，而當Insulin數值開始高於160後，患有糖尿病的人就逐漸增加
#根據提琴圖、分布估計圖、散步圖針對Age的分析可以發現，年齡越大的人患有糖尿病的人越多


# In[25]:


#---採用One-hot-encoding編碼方式，轉換成數值型態，使模型能夠使用可量化的資料---#
data_one_hot = pd.get_dummies(data)
data_one_hot.head(10)


# In[26]:


#---區分特徵欄位和預測目標欄位---#
clear_data = data_one_hot.drop(['Outcome'],axis=1)  #用drop和axis=1將Outcome這個欄位移除，並將剩餘欄位命名為 clear_data(特徵欄位)
label = data_one_hot['Outcome']  #將Outcome欄位命名為 label(預測目標欄位)，因為這是要預測的，所以不能放在特徵資料中


# In[27]:


clear_data.head()  #顯示前5筆資料，確認有刪除成功


# In[28]:


#-----以下進行模型開發-----#


# In[29]:


# 將資料分成輸入特徵 (x_features) 和輸出標籤 (y_label)
x_features = data_one_hot.iloc[:,:-1] 
y_label = data_one_hot.iloc[:,-1] 
x_features 


# In[30]:


#---切分資料集---#
from sklearn.model_selection import KFold #導入KFold類別，將數據集劃分為k個子集，每個子集均做一次驗證，其餘k-1個子集則用於訓練
kf = KFold(n_splits=5, shuffle=True, random_state=27) #創建物件實例kf，n_splits指定將數據集分為5個子集，shuffle指定是否對數據集進行洗牌，減少數據集中的偏差，random_state指定隨機種子


# In[31]:


#---選擇模型---#
#導入線性回歸模型
from sklearn.linear_model import LinearRegression 
model = LinearRegression() 


# In[32]:


#---交叉驗證---#
#建立四個空的列表，用來記錄模型訓練和驗證的結果
test_bias = [] 
test_variance = []
train_errors = []
test_errors = []

#利用K-Fold交叉驗證來評估模型的表現
for train_index, test_index in kf.split(x_features):
    # 將資料分成訓練集和測試集
    X_train, y_train = x_features.iloc[train_index], y_label[train_index]
    X_test, y_test = x_features.iloc[test_index], y_label[test_index]
    
    #使用訓練數據來訓練模型
    model.fit(X_train, y_train) 
    
    #使用模型對訓練集和測試集進行預測
    y_train_pred = model.predict(X_train) 
    y_test_pred = model.predict(X_test)
    
    #計算測試集的偏差和方差
    bias = np.sum((y_test_pred - np.mean(y_test)) ** 2) / len(y_test_pred)
    variance = np.sum((y_test_pred - np.mean(y_test_pred)) ** 2) / len(y_test_pred)
    bias_2 = np.mean((y_test_pred - np.mean(y_test)) ** 2)
    variance_2 = np.var(y_test_pred)
    
    #將偏差和方差加入對應的列表中
    test_bias.append(bias)
    test_variance.append(variance)
    
    #計算訓練集和測試集的誤差並加入對應的列表中    
    train_error = np.sum((y_train_pred - y_train) ** 2) /len(y_train_pred)
    test_error = np.sum((y_test_pred - y_test) ** 2) / len(y_test_pred)
    train_errors.append(train_error) 
    test_errors.append(test_error)


# In[33]:


#輸出偏差、方差、訓練誤差和測試誤差的平均值
print("Bias:",test_bias)
print("Variance:",test_variance)

print("平均訓練誤差：",sum(train_errors) / len(train_errors))
print("平均測試誤差：",sum(test_errors) / len(test_errors))


# In[ ]:




