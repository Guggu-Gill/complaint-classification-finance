import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn import preprocessing

import datetime


df=pd.read_csv("../complaints.csv")

COLN=['Date received','Product','Sub-product','Issue','Consumer complaint narrative','Company','State','ZIP code','Company response to consumer','Consumer disputed?']
ONE_HOT_ENC=['product','sub-product','issue', 'company', 'state','response']
FILL_MISSING=["sub-product","state","zip-code","narrative"]
def print_unique_values(df,coln):
    arr=np.unique(df[coln].to_list())
    for i in range(len(arr)):
        print(i,"--",arr[i])
    del arr
    
#pre processing function
def fill_in_missing(df,coln):
    df[coln]=df[coln].fillna("")
    return df
    

def transform_name(df):
    df=df[df['Consumer disputed?'].notna()]
    df=df.drop(["Complaint ID","Timely response?","Date sent to company","Tags","Submitted via","Company public response","Consumer consent provided?","Sub-issue"],axis=1)
    df.columns=['date-received', 'product', 'sub-product', 'issue','narrative', 'company', 'state', 'zip-code','response', 'disputed']
    return df

def one_hot_enc(df,coln):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(df[coln]))
    s=enc.transform(np.array(df[coln]))
    return s.toarray()

def binarize_target(x):
    lb = preprocessing.LabelBinarizer()
    lb.fit(['No','Yes'])
    return lb.transform(x)


def convert_zip_code(zip_code):
    if zip_code == '':
        zip_code = "00000"
    zip_code = re.sub(r'X{0,5}', "0", zip_code)
    zip_code = np.float32(zip_code)
    return zip_code


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
def get_product(x):
    
    if not isinstance(x['sub-product'], str):
        if x['product'] in prodmap:
            return(prodmap[x['product']])
        else:
            return(x['product'])
    else:
        return(prod2sub[x['sub-product']])




def map_values(df):
    df['sub-product'] = df[['product','sub-product']].apply(lambda x: get_subprods(x), axis =1)
    df['product'] = df[['product','sub-product']].apply(lambda x: get_product(x), axis =1)
    return df

def preprocess(df):
    df=transform_name(df)
    df=map_values(df)
    df=fill_in_missing(df,FILL_MISSING)
    #time based splitting
    df=df.sort_values("date-received")
    train=df.iloc[0:int(np.floor(df.shape[0]*0.67)),:]
    test=df.iloc[int(np.floor(df.shape[0]*0.67)):,:]
    #time based splitting
    return train,test


start= datetime.datetime.now()

train,test=preprocess(df)

end= datetime.datetime.now()

print("preprocessing took:{} sec".format(end-start))