### Multinomial Regression ####
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:\\Users\\Gopinaath\\OneDrive\\Documents\\Deployment files 5\\Recommendedfoodslist.csv")

df.columns

df.isna().sum()

df['MainIngredient'].unique()

df['MainIngredient'].value_counts()

df['MainIngredient'].replace({'almonds':0,'amaranthflour':1,'apples':2,'atta':3,'banana':4,'basmatirice':5,'boiledchicken':6,'bonelesschicken':7,'bread':8,'breadscrumbs':9,'broccoli':10,'buns':11,'butter':12,'caromseeds':13,'cashewnuts':14,'cauliflower':15,'cheese':16,'chicken':17,'chicken&prawn':18,'chickenbreast':19,'chickenbroth':20,'chickenchunks':21,'chickendrumsticks':22,'chickenlegs':23,'chickenmasala':24,'chickenpieces':25,'chickenstock':26,'chickenthighs':27,'chickenwings':28,'chickpeas':29,'chillipowder':30,'coconut':31,'coconutoil':32,'cookedrice':33,'corn':34,'cornflour':35,'dicedchicken':36,'dressedchicken':37,'elaichipowder':38,'flour':39,'fusilli':40,'gingergarlicpaste':41,'gramflour':42,'kiwi':43,'lambmince':44,'magaz':45,'mangoes':46,'mincedchicken':47,'mincedgarlic':48,'mincedmeat':49,

'minedmutton':50,'mushroom':51,'noodles':52,'oats':53,'onion':54,'orange':55,'paneer':56,'picklegravy':57,'potatoes':58,'rava':59,'redchilli':60,'rice':61,'rotis':62,'semolina':63,'sesameseeds':64,'shreddedchicken':65,'skinnedchicken':66,'slicedchicken':67,'tandoorichicken':68,'tomato':69,'turmeric':70,'walnuts':71,'wheatflour':72},inplace = True)
                              
df = df[['PreferredMealType','PreferredFoodType','MainIngredient','HighlypreferredIngredient','LowpreferredIngredient','Recommendedfoods']]

df.head()

## To check frequency distribution of values in categorical columns
df.MainIngredient.value_counts()

df.PreferredMealType.value_counts()

df.PreferredFoodType.value_counts()

df.HighlypreferredIngredient.value_counts()

df.LowpreferredIngredient.value_counts()

df.Recommendedfoods.value_counts()

## Creating dummy variables for input variables using Labelencoder
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df["PreferredMealType"] = labelencoder.fit_transform(df["PreferredMealType"])

df["PreferredFoodType"] = labelencoder.fit_transform(df["PreferredFoodType"])

df["HighlypreferredIngredient"] = labelencoder.fit_transform(df["HighlypreferredIngredient"])

df["LowpreferredIngredient"] = labelencoder.fit_transform(df["LowpreferredIngredient"])

df.dtypes

#df.to_csv('Recommendedfoods.csv',encoding="utf-8")
#import os
#os.getcwd()

#To check Input contains Nan value
np.isnan(df.any())

#To check if all inputs are finite values
np.isfinite(df.all())

train, test = train_test_split(df, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :5], train.iloc[:, 5])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, :5]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:, 5], test_predict)

train_predict = model.predict(train.iloc[:, :5]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:, 5], train_predict) 

X = df.iloc[:, :5]

y = df.iloc[:, 5]

regressor = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

#Fitting model with trainig data
regressor.fit(X, y)


import pickle

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
print(model1.predict([[2,0,68,10,23]]))




