## 介紹
因為是第一次接觸Kaggle，就選擇了大家都推薦的Titanic作為練習，他的目標就是要預測鐵達尼號乘客是否生還。
下載檔案後會有兩個csv檔案，Train和Test，分別是用來「訓練模型」和評估模型的「預測效果」，簡單講就是學校的學習和考試資料。  
  
接下來是我的解題步驟，我是用最簡單的方式來進行預測，具體步驟如下：    

1.讀檔  
2.檢查數據信息 (info、describe、isnull.sum)  
3.處理缺失值(fillna、drop、map、get_dummies)  
4.加入需要的特徵Features  
5.再次檢查缺失值  
6.訓練模型(Random Forest)  
7.評估模型(Accuracy_Score)  

## 以下將詳細介紹：  
一開始一定是先讀取檔案！     
我這邊是用絕對位置的方式來讀檔  

train_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/train.csv')  
test_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/test.csv')  
print(train_data.head())   #檢視前五筆資料，確定有正確讀檔  

接下來，就是查看data的基本資料  

print(train_data.info())  # 查看資料概要  
print(train_data.describe())  # 查看統計摘要  

再來看有沒有缺失值  

print(train_data.isnull().sum())  # 檢查缺失值  

然後將有問題的數據進行資料清洗  
我們可以看到Age跟Embarked都是有缺失值的，所以分別使用了中位數和眾數進行填補  

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)  
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)  
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)  

然後Carbin資料過少，沒有甚麼用處，我們就選擇刪除   

train_data.drop('Cabin',axis=1,inplace=True)  
test_data.drop('Cabin',axis=1,inplace=True)  

接下來，我將性別(字串)轉成0、1(數值)  

train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})  
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})  

因為Embarked是一個多類別特徵，所以我們使用one hot encoding來解讀，才能避免順序問題，確保類別資料被正確解讀  

train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)  
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)  

然後加上我們最重要的特徵，我會選擇這些是因為他們跟存活率都有明確的關聯，舉例來說，可以看出女性和艙等存活率都特別高，又或者是小孩跟老人的存活率明顯不同    
把她轉換成X跟Y是為了特徵分割，分別將X用來訓練，Y用來預測  


features =  ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X_train = train_data[features]
Y_train = train_data['Survived']
X_test = test_data[features]  

我們這邊又填補Fare的原因是因為，Fare的缺失值只存在於測試集  
我們在print一次就能夠發現他有缺失值  

print(X_train.isnull().sum())
print(X_test.isnull().sum())
X_test['Fare'].fillna(X_test['Fare'].median(),inplace=True)  

最後就是訓練我們的模組，我採用的是RandomForest，並設定他的種子值為42，樹的數量為100，因為她能夠處理多種特徵類型，也不容易發生過度擬合  

model = RandomForestClassifier(random_state=42,n_estimators=100)  
model.fit(X_train,Y_train)  
predictions = model.predict(X_test)  
train_predictions = model.predict(X_train)  

最後就是創建提交文件  
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})    
submission.to_csv('submission.csv', index=False)  

就完成啦






