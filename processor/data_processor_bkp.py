# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: AI Model processor component. This is a special component having an AI based predictions processor.

Reference: 


Utility class for loading daat from database and processing it as per requirment of AI model

Reference
- 

# Change History

"""
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
import calendar
from utils.configloader import ConfigUtil
from utils.logging_init              import logging
from utils.mssqlutil import MSSQLUtil
import db.constants as pr_constant
from datetime import date, datetime

class DataLoaderUtil:
    """
    Utility class for loading and processing data from database
    """
    __instance  = None

    @staticmethod
    def getInstance():
        if DataLoaderUtil.__instance == None:
            DataLoaderUtil()
        return DataLoaderUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if DataLoaderUtil.__instance == None:
            DataLoaderUtil.__instance = self
            self.__run()

    def __run(self):
        """
        Load configurations
        """
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        logging.info("loading data from DB...")
        
    def initGenerateData(self, day):
        """ Checks if the day provided is the intended day of initiation of the Weekly Data generantion function
        and then invokes Weekly Data Generation Module."""
        try:
            logging.info('Triggering data generation')
            if date.today().day == day:
                logging.info('Initiatting data generation on {0}'.format(date.today()))
                
                trans_df,features_df = self.retrieveDataFromDb()
                logging.info('Retrieved Features Data and Transactional Data from Database')
                logging.info('Starting Data Labelling')
                features_df = self.get_features(trans_df,features_df)


                return features_df
            else:
                logging.info('Data generation not initiated on {0}'.format(date.today()))
                return
        
        except Exception as e:
            logging.error(e)
            
    def retrieveDataFromDb(self):
        """This method retrieves the ING_Budget and ING_BudgetTransactions datasets from INGAGAE Databases to generate weekly Budget Dataset."""
        try:

            # Change this function

            logging.info("Retrieved Data from Database")
            sqlInstance = MSSQLUtil.getInstance()

            trans_list    = sqlInstance.executeQuery(self.__envConfiguration['db.queryBudgetTrans'])
            np_trans_list = np.array(trans_list)
            trans_df = pd.DataFrame (np_trans_list, columns = ['CustomerID', 'ProductCode', 'TransAmount', 'TransDate', 'TransType'])
            trans_df['CustomerID']            = trans_df['CustomerID'].astype(str)
            trans_df['ProductCode']           = trans_df['ProductCode'].astype(str)
            trans_df['TransAmount']           = trans_df['TransAmount'].astype(float)
            trans_df['TransDate']             = pd.to_datetime(trans_df['TransDate'])
            trans_df['TransType']             = trans_df['TransType'].astype(str)


            features_list = sqlInstance.executeQuery(self.__envConfiguration['db.queryAIFeatures'])
            
            np_features_list = np.array(features_list)
            # features_df = pd.DataFrame (np_features_list, columns = ['CustomerID' ,'SegmentID'])
            # # Set data types of each column
            # features_df['CustomerID']         = features_df['CustomerID'].astype(str)
            # features_df['SegmentID']          = features_df['SegmentID'].astype(int)

            
            features_df = pd.DataFrame (np_features_list, columns = ['CustomerID' ,'Age' ,'Week_1_Avg_Credit' ,'Week_2_Avg_Credit' ,'Week_3_Avg_Credit' ,'Week_4_Avg_Credit', 'Week_1_Avg_Debit' ,'Week_2_Avg_Debit' ,'Week_3_Avg_Debit' ,'Week_4_Avg_Debit',
                                                                    'Weekly_Min_Credit' ,'Weekly_Min_Debit' ,'Weekly_Max_Credit', 'Weekly_Max_Debit' ,'Weekly_Median_Credit' ,'Weekly_Median_Debit' ,'Weekly_Count_Median_Credit', 
                                                                    'Weekly_Count_Median_Debit' ,'Weekly_Count_Variance_Credit' ,'Weekly_Count_Variance_Debit', 'Monthly_Min_Credit' ,'Monthly_Min_Debit' ,'Monthly_Max_Credit', 
                                                                    'Monthly_Max_Debit' ,'Monthly_Median_Credit' ,'Monthly_Median_Debit' ,'Monthly_Count_Median_Credit', 'Monthly_Count_Median_Debit ','Monthly_Count_Variance_Credit' ,'Monthly_Count_Variance_Debit' ,'Last_Month_PCT_CH_Credit' ,'Last_Month_PCT_CH_Debit' ,'SegmentID'])
    
            features_df['CustomerID']         = features_df['CustomerID'].astype(str)
            for col_num in range(1, len(features_df.columns)):
                features_df[features_df.columns[col_num]] = features_df[features_df.columns[col_num]].apply(pd.to_numeric)

            
            return trans_df,features_df

        except Exception as e:
            logging.error(e)



    def get_features(self, trans_df,features_df):
        try:
            # df["TransDate"] = pd.to_datetime(df["TransDate"])
            trans_df["year"] = trans_df.TransDate.dt.year
            df_agg = trans_df.groupby([trans_df.CustomerID, trans_df.ProductCode,trans_df.year, trans_df.TransDate.dt.month, trans_df.TransType])["TransAmount"].sum().reset_index().rename(columns = {"TransDate": "Month"})
            df_pivot = pd.pivot_table(df_agg,index=['CustomerID','ProductCode','year','Month'], columns=['TransType'], values=['TransAmount']).fillna(0).reset_index()
            df_pivot.columns = ["CustomerID","ProductCode","year","Month","Credit","Debit"]
            df_pivot["Balance"] = df_pivot["Credit"] - df_pivot["Debit"]
            final_df = df_pivot.groupby(['CustomerID','ProductCode','year','Month','Credit','Debit']).sum().groupby(level=0).cumsum().reset_index()
            final_df["Day"] = "01"
            final_df["Date"] = final_df["Month"].astype(int).astype(str) + "-" + final_df["Day"].astype(str) + "-" + final_df["year"].astype(int).astype(str)
            final_df["Date"] = pd.to_datetime(final_df["Date"])
            final_df["Date"] = final_df["Date"] + pd.offsets.MonthEnd(0) 
            final_df.drop(["year","Month","Day"], axis = 1, inplace = True)

            # Get Positive months Balance only
            final_df = self.filter_negative_balance(final_df)

            final_df["mean_bal"] = final_df.groupby(["CustomerID","ProductCode"])['Balance'].transform(lambda x: x.rolling(3).mean())
            final_df['Prev_Mean_Bal'] = final_df.groupby(["CustomerID","ProductCode"])['mean_bal'].shift()
            final_df.drop(["mean_bal"], axis = 1, inplace = True)


            final_df["Struggle"] = np.where(final_df['Balance'] < (final_df['Prev_Mean_Bal'] - (0.3 * final_df['Prev_Mean_Bal'])), 1, 0)
            final_df["Prev_Mean_Bal"] = final_df["Prev_Mean_Bal"].fillna(0)
            final_df['Struggle_shifted'] = final_df['Struggle'].shift(-1).where(final_df.CustomerID.eq(final_df.CustomerID.shift(-1)))
            final_df["Struggle_shifted"] = final_df["Struggle_shifted"].fillna(0).astype(int)

            #Append Features
            final_df = pd.merge(final_df, features_df, on = "CustomerID", how = "inner")
            # final_df = final_df[["CustomerID","ProductCode","SegmentID","Date","Credit", "Debit", "Balance","Prev_Mean_Bal","Struggle"]]
            # final_df = final_df[["CustomerID","ProductCode","SegmentID","Date","Credit", "Debit", "Balance","Prev_Mean_Bal","Struggle"]]
            
            # final_df.to_csv(r"G:\POCs\PFM-AI\temp\final_df.csv", index = False)
            return final_df


        
            
        except Exception as e:
            logging.error(e)

    def filter_negative_balance(self, df):
        try:
            df["pos_month_count"] = np.where(df["Balance"] < 0, 0, 1)
            df_agg = df.groupby(["CustomerID","ProductCode"])["pos_month_count"].sum().reset_index()
            df_agg = df_agg[df_agg["pos_month_count"]==8]
            filtered_customers = df_agg["CustomerID"].tolist()
            df = df[df["CustomerID"].isin(filtered_customers)]
            df.drop(["pos_month_count"], axis = 1, inplace = True)
            return df

            

        except Exception as e:
            logging.error(e)
