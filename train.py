import pandas
from sklearn import linear_model
from sklearn.metrics import explained_variance_score

train = pandas.read_csv("train.csv")
train_data = []
train_label = []
#test['test'] = pandas.Series()
for index, row in train[train['TownName']=="松山區"].iterrows():
    train_data.append([row['Hagibis'],row['Matmo'],row['Fung-wong'],row['Chan-hom']])

for index, row in train[train['TownName']=="松山區"].iterrows():
    train_label.append(row['Soudelor'])

reg = linear_model.LinearRegression()
reg.fit(train_data[:int(0.6*len(train_data))], train_label[:int(0.6*len(train_label))])
predict_label = reg.predict(train_data[int(0.6*len(train_data)):])
print(reg.coef_)
print(explained_variance_score(train_label[int(0.6*len(train_label)):], predict_label))
#test.to_csv('out.csv')
