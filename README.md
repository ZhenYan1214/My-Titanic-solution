# Kaggle 實作分享：Titanic 生存預測

## 介紹

這是我第一次接觸 Kaggle，比賽選擇了大家推薦的入門題目 —— **Titanic 生存預測** 作為練習。  
本比賽的目標是預測鐵達尼號乘客是否生還，並使用提供的資料進行特徵工程和模型訓練。  
比賽提供了兩個 CSV 文件：
- **Train.csv**：用於訓練模型（學習資料）。
- **Test.csv**：用於測試模型預測效果（考試資料）。

以下是我的解題步驟：

1. 讀檔  
2. 檢查數據信息（info、describe、isnull.sum）  
3. 處理缺失值（fillna、drop、map、get_dummies）  
4. 加入需要的特徵（Features）  
5. 再次檢查缺失值  
6. 訓練模型（Random Forest）  
7. 評估模型（Accuracy_Score）  
8. 創建提交文件  

---

## 解題步驟

### **1. 讀檔**
首先，讀取比賽提供的 CSV 資料集，並檢查前五筆資料：

```python
train_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/train.csv')  
test_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/test.csv')  

print(train_data.head())  # 檢視前五筆資料


接下來，就是查看data的基本資料  

print(train_data.info())  # 查看資料概要  
print(train_data.describe())  # 查看統計摘要  
```
### **2. 檢查數據信息**
使用基本方法檢查數據的結構和統計摘要，並查看是否有缺失值：

```python
# 查看資料概要
print(train_data.info())

# 查看統計摘要
print(train_data.describe())

# 檢查缺失值
print(train_data.isnull().sum())
```
### **3. 處理缺失值**
根據檢查結果，對缺失值進行填補或刪除：

Age：用中位數填補。
Embarked：用眾數填補。
Cabin：刪除該欄位，因為缺失值過多。
```python
# 填補缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 刪除 Cabin 欄位
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
```
### **4. 特徵工程**
將性別（文字類型）轉換為數值類型（0 和 1）。
使用 One-Hot Encoding 處理 Embarked 多類別特徵。
選擇與生存率關係較強的特徵作為模型輸入。
```python
# 性別轉換
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# 選擇特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X_train = train_data[features]
Y_train = train_data['Survived']
X_test = test_data[features]
```
補充處理測試集中 Fare 的缺失值：
```python
X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True)
```
### **5. 訓練模型**
選擇 Random Forest 作為分類模型，並設定隨機種子（random_state=42）和 100 顆決策樹。

```python
from sklearn.ensemble import RandomForestClassifier

# 建立並訓練模型
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, Y_train)

# 預測訓練集
train_predictions = model.predict(X_train)
```
### **6. 模型評估**
使用 Accuracy Score 評估模型在訓練集上的準確率：

```python
from sklearn.metrics import accuracy_score

# 計算訓練集準確率
accuracy = accuracy_score(Y_train, train_predictions)
print(f"模型準確率: {accuracy:.4f}")
```
### **7. 預測與提交**
使用測試集進行預測，並將結果保存為提交格式的 CSV 檔案：

```python
# 預測測試集
predictions = model.predict(X_test)

# 創建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
```
### **結果**
這樣就成功完成 Titanic 生存預測啦！  
模型準確率達到 0.76076%。
