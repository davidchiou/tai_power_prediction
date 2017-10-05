import pandas
from sklearn import linear_model

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
print(reg.predict([train_data[32]]))
print(reg.coef_)
#test.to_csv('out.csv')
