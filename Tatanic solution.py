import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#讀檔
train_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/train.csv')
test_data = pd.read_csv(r'C:/Users/User/OneDrive/桌面/Py/Pandas/titanic/test.csv')

#填補缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
#刪除Cabin
train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)
#用one hot encoding
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)
#加入Features，X訓練Y預測
features =  ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X_train = train_data[features]
Y_train = train_data['Survived']
X_test = test_data[features]
X_test['Fare'].fillna(X_test['Fare'].median(),inplace=True)

#訓練模組
model = RandomForestClassifier(random_state=42,n_estimators=100)
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
train_predictions = model.predict(X_train)

# 創建提交文件
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)

print(submission.head())




