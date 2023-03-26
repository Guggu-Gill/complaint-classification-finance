import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn import preprocessing
from tqdm import tqdm
import datetime
import tensorflow as tf
import tensorflow_hub as hub

#reading the data frame
df=pd.read_csv("../complaints.csv")

#these columns will be selected for the model
COLN=['Date received','Product','Sub-product','Issue','Consumer complaint narrative','Company','State','ZIP code','Company response to consumer','Consumer disputed?']
#these features will be used for one hot encoding
ONE_HOT_ENC=['product','sub-product','issue', 'company', 'state','response']
#missing values of below features will be imputed
FILL_MISSING=["sub-product","state","zip-code","narrative"]

#This function prints the unique values in a column
def print_unique_values(df,coln):
    arr=np.unique(df[coln].to_list())
    for i in range(len(arr)):
        print(i,"--",arr[i])
    del arr
    
#This helper function fill in missing values in column
def fill_in_missing(df,coln):
    df[coln]=df[coln].fillna("")
    #this returns dataframe
    return df
    
#This helper funmction is used to select and change column names
def transform_name(df):
    df=df[df['Consumer disputed?'].notna()]
    df=df.drop(["Complaint ID","Timely response?","Date sent to company","Tags","Submitted via","Company public response","Consumer consent provided?","Sub-issue"],axis=1)
    df.columns=['date-received', 'product', 'sub-product', 'issue','narrative', 'company', 'state', 'zip-code','response', 'disputed']
    #this returns data frame
    return df

#This helper function is used for one hot encoding
def one_hot_enc(df,coln):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(df[coln]))
    s=enc.transform(np.array(df[coln]))
    #this returns array
    return s.toarray()

#This helper function is used to encode target variable into binary format
def binarize_target(x):
    lb = preprocessing.LabelBinarizer()
    lb.fit(['No','Yes'])
    #this returns array
    return lb.transform(x)


#This helper function is used for binning of zipcodes
def convert_zip_code(zip_code):
    if zip_code == '':
        zip_code = "00000"
    zip_code=re.sub("[^0-9]", "",zip_code)
    zip_code = re.sub(r'X{0,5}', "0", zip_code)
    zip_code = np.float32(zip_code)
    return zip_code


#Mappimg values of sub-proudct & product

# Credits https://www.kaggle.com/code/ashwinids/cleaning-exploring-consumer-complaints-data
mapping_old2new = {
    "Auto": "Auto debt",
    "Credit card": "Credit card debt",
    "Federal student loan": "Federal student loan debt",
    "Medical": "Medical debt",
    "Mortgage": "Mortgage debt",
    "Non-federal student loan": "Private student loan debt",
    "Other (i.e. phone, health club, etc.)": "Other debt",
    "Payday loan": "Payday loan debt",
    "Non-federal student loan": "Private student loan",
    "Federal student loan servicing": "Federal student loan",
    "Credit repair": "Credit repair services",
    "Credit reporting": "Credit reporting",
    "Conventional adjustable mortgage (ARM)": "Conventional home mortgage",
    "Conventional fixed mortgage": "Conventional home mortgage",
    "Home equity loan or line of credit": "Home equity loan or line of credit (HELOC)",
    "Other": "Other type of mortgage",
    "Other mortgage": "Other type of mortgage",
    "Second mortgage":"Other type of mortgage",
    "Credit card": "General-purpose credit card or charge card",
    "General purpose card": "General-purpose prepaid card",
    "Gift or merchant card": "Gift card",
    "Electronic Benefit Transfer / EBT card": "Government benefit card",
    "Government benefit payment card": "Government benefit card",
    "ID prepaid card": "Student prepaid card",
    "Other special purpose card":  "Other prepaid card",
    "Store credit card": "Other prepaid card",
    "Transit card": "Other prepaid card",
    "(CD) Certificate of deposit": "CD (Certificate of Deposit)",
    "Other bank product/service": "Other banking product or service",
    "Cashing a check without an account": "Other banking product or service",
    "Vehicle lease": "Lease",
    "Vehicle loan": "Loan",
    "Check cashing": "Check cashing service",
    "Mobile wallet": "Mobile or digital wallet",
    "Traveler’s/Cashier’s checks": "Traveler's check or cashier's check"
}
prod2sub = {
    "Auto debt": "Debt collection",
    "Credit card debt": "Debt collection",
    "Federal student loan debt": "Debt collection",
    "I do not know": "Debt collection",
    "Medical debt": "Debt collection",
    "Mortgage debt": "Debt collection",
    "Private student loan debt": "Debt collection",
    "Other debt": "Debt collection",
    "Payday loan debt": "Debt collection",
    "Credit repair services": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Other personal consumer report": "Credit reporting, credit repair services, or other personal consumer reports",
    "Conventional home mortgage": "Mortgage",
    "FHA mortgage": "Mortgage",
    "Home equity loan or line of credit (HELOC)": "Mortgage",
    "Other type of mortgage": "Mortgage",
    "Reverse mortgage": "Mortgage",
    "VA mortgage": "Mortgage",
    "General-purpose credit card or charge card": "Credit card or prepaid card",
    "General-purpose prepaid card": "Credit card or prepaid card",
    "Gift card": "Credit card or prepaid card",
    "Government benefit card": "Credit card or prepaid card",
    "Student prepaid card": "Credit card or prepaid card",
    "Payroll card": "Credit card or prepaid card",
    "Other prepaid card": "Credit card or prepaid card",
    "CD (Certificate of Deposit)": "Checking or savings account",
    "Checking account": "Checking or savings account",
    "Other banking product or service": "Checking or savings account",
    "Savings account": "Checking or savings account",
    "Lease": "Vehicle loan or lease",
    "Loan": "Vehicle loan or lease",
    "Federal student loan": "Student loan",
    "Private student loan": "Student loan",
    "Installment loan": "Payday loan, title loan, or personal loan",
    "Pawn loan": "Payday loan, title loan, or personal loan",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Personal line of credit": "Payday loan, title loan, or personal loan",
    "Title loan": "Payday loan, title loan, or personal loan",
    "Check cashing service": "Money transfer, virtual currency, or money service",
    "Debt settlement": "Money transfer, virtual currency, or money service",
    "Domestic (US) money transfer": "Money transfer, virtual currency, or money service",
    "Foreign currency exchange": "Money transfer, virtual currency, or money service",
    "International money transfer": "Money transfer, virtual currency, or money service",
    "Mobile or digital wallet": "Money transfer, virtual currency, or money service",
    "Money order": "Money transfer, virtual currency, or money service",
    "Refund anticipation check": "Money transfer, virtual currency, or money service",
    "Traveler's check or cashier's check": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Not given":"Not given"
}

#This helper function is used to map new subproduct values
# Credits https://www.kaggle.com/code/ashwinids/cleaning-exploring-consumer-complaints-data
def get_subprods(x):
    
    if x['sub-product'] in mapping_old2new:
        if x['sub-product']=="Other":
            if x['product']=='Mortage':
                return("Other type of mortgage")
            else:
                return("Other debt")
        else:
            return(mapping_old2new[x['sub-product']])
    else:
        return(x['sub-product'])



# Credits https://www.kaggle.com/code/ashwinids/cleaning-exploring-consumer-complaints-data

prodmap = {
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit card": "Credit card or prepaid card"
}
#This helper gunction is used to map product values 
def get_product(x):
    
    if not isinstance(x['sub-product'], str):
        if x['product'] in prodmap:
            return(prodmap[x['product']])
        else:
            return(x['product'])
    else:
        return(prod2sub[x['sub-product']])



#This helper function uses get_subprods() & get_product() to prepocess values
def map_values(df):
    df['sub-product'] = df[['product','sub-product']].apply(lambda x: get_subprods(x), axis =1)
    df['product'] = df[['product','sub-product']].apply(lambda x: get_product(x), axis =1)
    return df

def preprocess(df):
    df=transform_name(df)
    #map new values of features and sub-product feature
    df=map_values(df)
    #fill in missing values
    df=fill_in_missing(df,FILL_MISSING)
    #We are doing time based spiting checking whether model is not overfitting & generalises good to future data.

    df=df.sort_values("date-received")
    train=df.iloc[0:int(np.floor(df.shape[0]*0.67)),:]
    test=df.iloc[int(np.floor(df.shape[0]*0.67)):,:]
    #returns train & test dataframe
    return train,test


def preprocess_2(df,filename):
    #bucketing of zipcode & appending the results into array
    arr=[]
    lent=df.shape[0]
    zip_code=list(map(str,df['zip-code'].tolist()))
    for i in tqdm(range(0,lent)):
        arr.append(convert_zip_code(zip_code[i]))
    arr=np.array(arr)
    del zip_code
    
    #doing one hot encoding of defined features
    frst=one_hot_enc(train,ONE_HOT_ENC)
    scnd=np.array(arr)
    del arr
    
    #appending ONE HOT ENCODED array & ZIP CODE array
    arr2=[]
    for i in tqdm(range(scnd.shape[0])):
        arr2.append(np.concatenate((frst[i],[scnd[i]]),axis=0))
    arr2=np.array(arr2)
    del frst 
    del scnd
    
    #converting the appended array to sparse matrix to for saving sisk space 
    sparse_matrix=scipy.sparse.csc_matrix(np.array(arr2))
    del arr2
    #saving the sparse matrix into disk
    np.save(filename,sparse_matrix)



#doing pre-processing 

start= datetime.datetime.now()

train,test=preprocess(df)
del test
del df
preprocess_2(train,"train.npy")

np.save("y_train.npy",np.array(binarize_target(train['disputed']).tolist()))
del train
end= datetime.datetime.now()

print("preprocessing took:{} sec".format(end-start))

#loading the sparse array into memory

# print(np.load('train.npy',allow_pickle=True))

#doing transformer based UNIVERSAL SENTENCE ENCODING on narrative text based feature

#https://arxiv.org/abs/1803.11175


train=pd.read_csv("train_ap.csv",on_bad_lines='skip')
test=pd.read_csv("test_ap.csv",on_bad_lines='skip')

#fill in missing values
train=train['narrative'].fillna(" ")
test=test['narrative'].fillna(" ")

#applying encoding
embed_narrative_1 = embed(train)
embed_narrative_2 = embed(test)

#saving 512 dimensional embedding into disk
np.save("train.npy",embed_narrative_1)
np.save("test.npy",embed_narrative_2)
