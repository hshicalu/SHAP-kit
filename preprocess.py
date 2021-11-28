import pandas as pd
import numpy as np


# class Preprocess:
#     def __init__(self, data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
#         self.data = data
#         self.train = train
#         print(type(data))
#         print(type(train))
#         if self.data is not None:
#             self._preprocess(self.data, self.train)
#         else:
#             pass
def preprocess(data, train):
    data['Age'].fillna(train.Age.mean(), inplace=True)
    data["Embarked"].fillna(train.Embarked.mean(), inplace=True)
    data["Fare"].fillna(train.Fare.mean(), inplace=True)
    
    combine = [data]

    for data in combine: 
            data['Salutation'] = data.Name.str.extract(' ([A-Za-z]+).', expand=False) 

    for data in combine: 
        data['Salutation'] = data['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['Salutation'] = data['Salutation'].replace('Mlle', 'Miss')
        data['Salutation'] = data['Salutation'].replace('Ms', 'Miss')
        data['Salutation'] = data['Salutation'].replace('Mme', 'Mrs')
        del data['Name']

    Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 

    for data in combine: 
        data['Salutation'] = data['Salutation'].map(Salutation_mapping) 
        data['Salutation'] = data['Salutation'].fillna(0)
            
    for data in combine: 
        data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(x)[0])
        data['Ticket_Lett'] = data['Ticket_Lett'].apply(lambda x: str(x)) 
        data['Ticket_Lett'] = np.where((data['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['Ticket_Lett'], np.where((data['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
        data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x)) 
        del data['Ticket'] 

    data['Ticket_Lett'] = data['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)            

    for data in combine: 
        data['Cabin_Lett'] = data['Cabin'].apply(lambda x: str(x)[0]) 
        data['Cabin_Lett'] = data['Cabin_Lett'].apply(lambda x: str(x)) 
        data['Cabin_Lett'] = np.where((data['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),data['Cabin_Lett'], np.where((data['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
        del data['Cabin']

    data['Cabin_Lett'] = data['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1)
    
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    for data in combine:
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    return data
    
