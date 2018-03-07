#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:38:25 2018

@author: jiangyunzhang
"""

import scipy.optimize as optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
import copy

# Get data 
KeyRate = pd.read_csv('Key Rate.csv')
ZeroCurve = pd.read_csv('Zero Curve.csv')
TwoYear = pd.read_csv('2-6.csv') 
ThreeYear = pd.read_csv('3-12.csv')
FiveYear = pd.read_csv('5-5.csv')
SevenYear = pd.read_csv('7-10.csv')
TenYear = pd.read_csv('10-8.csv')

#Set Innitial Parameters
NumPeriod = len(KeyRate.TimePeriod)
Amount = 10

# Q1
# Get Modified Duration and DV01
TwoYear['Mod'] = TwoYear.MacDur / (1+TwoYear.Yield/2)
TwoYear['DV01'] = TwoYear.Mod * TwoYear.Price/10000

ThreeYear['Mod'] = ThreeYear.MacDur / (1+ThreeYear.Yield/2)
ThreeYear['DV01'] = ThreeYear.Mod * ThreeYear.Price/10000

FiveYear['Mod'] = FiveYear.MacDur / (1+FiveYear.Yield/2)
FiveYear['DV01'] = FiveYear.Mod * FiveYear.Price/10000

SevenYear['Mod'] = SevenYear.MacDur / (1+SevenYear.Yield/2)
SevenYear['DV01'] = SevenYear.Mod * SevenYear.Price/10000

TenYear['Mod'] = TenYear.MacDur / (1+TenYear.Yield/2)
TenYear['DV01'] = TenYear.Mod * TenYear.Price/10000

#Calculate Hedging Errors and Transaction Costs
HE = np.zeros((NumPeriod-1,4))
Units = np.zeros((NumPeriod,4))
TransactionCost = np.zeros((NumPeriod-1,4))

for i in range(0,len(Units)):
     Units[i][0] = 100000 * SevenYear.DV01[i] / TwoYear.DV01[i]
     Units[i][1] = 100000 * SevenYear.DV01[i] / ThreeYear.DV01[i]
     Units[i][2] = 100000 * SevenYear.DV01[i] / FiveYear.DV01[i]
     Units[i][3] = 100000 * SevenYear.DV01[i] / TenYear.DV01[i] 

for i in range(0,len(HE)):     
     HE[i][0] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-Units[i][0] * (TwoYear.Price[i+1]-TwoYear.Price[i])
     HE[i][1] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-Units[i][1] * (ThreeYear.Price[i+1]-ThreeYear.Price[i])
     HE[i][2] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-Units[i][2] * (FiveYear.Price[i+1]-FiveYear.Price[i])
     HE[i][3] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-Units[i][3] * (TenYear.Price[i+1]-TenYear.Price[i])

# function for doing basic statistics
def Stat(x):
        X = {'Mean':np.mean(x,axis=0),\
        'Standard Deviation':np.std(x,axis=0),\
        'Median':np.median(x,axis=0),\
        'Skewness':stats.skew(x,axis=0),\
        'Maximum':np.max(x,axis=0)}   
        X = pd.DataFrame(data=X)
        return X
StatHE = Stat(HE)

# Assume transaction fee equals to 0.01% of transaction amount
for i in range(0,len(TransactionCost)-1):
        TransactionCost[i][0] = np.abs((Units[i+1][0] - Units[i][0]) * TwoYear.Price[i] * 0.0001)
        TransactionCost[i][1] = np.abs((Units[i+1][1] - Units[i][1]) * ThreeYear.Price[i] * 0.0001)
        TransactionCost[i][2] = np.abs((Units[i+1][2] - Units[i][2]) * FiveYear.Price[i] * 0.0001)
        TransactionCost[i][3] = np.abs((Units[i+1][3] - Units[i][3]) * TenYear.Price[i] * 0.0001)
    
StatTC = Stat(TransactionCost)

# Q2
# Calcluate Change in Key Rates and YTM
DeltaKeyRate = np.zeros((len(KeyRate)-1,5))
DeltaYTM = np.zeros((len(KeyRate)-1,5))

for i in range(0,len(KeyRate)-1):
        DeltaKeyRate[i][0] = KeyRate['0.50'][i+1] - KeyRate['0.50'][i]
        DeltaKeyRate[i][1] = KeyRate['1.00'][i+1] - KeyRate['1.00'][i]        
        DeltaKeyRate[i][2] = KeyRate['2.00'][i+1] - KeyRate['2.00'][i]
        DeltaKeyRate[i][3] = KeyRate['5.00'][i+1] - KeyRate['5.00'][i]
        DeltaKeyRate[i][4] = KeyRate['10.00'][i+1] - KeyRate['10.00'][i]
        
        DeltaYTM[i][0] = TwoYear.Yield[i+1] - TwoYear.Yield[i]
        DeltaYTM[i][1] = ThreeYear.Yield[i+1] - ThreeYear.Yield[i]
        DeltaYTM[i][2] = FiveYear.Yield[i+1] - FiveYear.Yield[i]
        DeltaYTM[i][3] = SevenYear.Yield[i+1] - SevenYear.Yield[i]
        DeltaYTM[i][4] = TenYear.Yield[i+1] - TenYear.Yield[i]
        
StatKR = Stat(DeltaKeyRate)
StatYTM = Stat(DeltaYTM)
KRCorrelation = np.corrcoef(DeltaKeyRate, y = None, rowvar = False)
YTMCorrelation = np.corrcoef(DeltaYTM, y = None, rowvar = False)

# Q3
# Calculation Beta

def CalBeta( Period , m , n , Delta , PastAll): 
    Beta = np.zeros((len(Delta),1)) 
    if PastAll == False:
        for i in range(Period,len(Delta)):
            ListA = np.zeros(Period)
            ListB = np.zeros(Period)
            for j in range(i-Period,i-1):
                ListA[j-i+Period] =  Delta[j][m] 
                ListB[j-i+Period] =  Delta[j][n]
            cor = stats.pearsonr(ListA,ListB)
            StdA = np.std(ListA, axis = 0)
            StdB = np.std(ListB, axis = 0)
            Beta[i] = cor[0] * StdA / StdB           
    else:
        for i in range(0,len(Delta)):
            ListA = np.zeros(i)
            ListB = np.zeros(i)            
            if i > 5:
                for j in range(0,i-1):
                    ListA[j] =  Delta[j][m] 
                    ListB[j] =  Delta[j][n]
                cor = stats.pearsonr(ListA,ListB)
                StdA = np.std(ListA, axis = 0)
                StdB = np.std(ListB, axis = 0)
                Beta[i] = cor[0] * StdA / StdB
    return Beta

def CalAlpha( Period , m , n , Delta , PastAll): 
    Alpha = np.zeros((len(Delta),1))
    Beta = np.zeros((len(Delta),1)) 
    if PastAll == False:
        for i in range(Period,len(Delta)):
            ListA = np.zeros(Period)
            ListB = np.zeros(Period)
            for j in range(i-Period,i-1):
                ListA[j-i+Period] =  Delta[j][m] 
                ListB[j-i+Period] =  Delta[j][n]
            cor = stats.pearsonr(ListA,ListB)
            StdA = np.std(ListA, axis = 0)
            StdB = np.std(ListB, axis = 0)
            Beta[i] = cor[0] * StdA / StdB
            Alpha[i] = np.mean(ListA)-Beta[i]*np.mean(ListB)
           
    else:
        for i in range(0,len(Delta)):
            ListA = np.zeros(i)
            ListB = np.zeros(i)            
            if i > 5:
                for j in range(0,i-1):
                    ListA[j] =  Delta[j][m] 
                    ListB[j] =  Delta[j][n]
                cor = stats.pearsonr(ListA,ListB)
                StdA = np.std(ListA, axis = 0)
                StdB = np.std(ListB, axis = 0)
                Beta[i] = cor[0] * StdA / StdB
                Alpha[i] = np.mean(ListA)-Beta[i]*np.mean(ListB)
    return Alpha

# Get Different Betas
BetaOneMonth = []
AlphaOneMonth = []

BetaOneYear = []
AlphaOneYear = []

BetaAllBack = []
AlphaAllBack = []

for i in range(0,5):
        BetaOneMonth.append(CalBeta(4,3,i,DeltaYTM,PastAll = False))
        BetaOneMonth.append(Stat(CalBeta(4,3,i,DeltaYTM,PastAll = False)))
        AlphaOneMonth.append(CalAlpha(4,3,i,DeltaYTM,PastAll = False))
        AlphaOneMonth.append(Stat(CalAlpha(4,3,i,DeltaYTM,PastAll = False)))
        
        BetaOneYear.append(CalBeta(52,3,i,DeltaYTM,PastAll = False))
        BetaOneYear.append(Stat(CalBeta(52,3,i,DeltaYTM,PastAll = False)))
        AlphaOneYear.append(CalAlpha(52,3,i,DeltaYTM,PastAll = False))
        AlphaOneYear.append(Stat(CalAlpha(52,3,i,DeltaYTM,PastAll = False)))
        
        BetaAllBack.append(CalBeta(1,3,i,DeltaYTM,PastAll = True))
        BetaAllBack.append(Stat(CalBeta(1,3,i,DeltaYTM,PastAll = True)))
        AlphaAllBack.append(CalAlpha(1,3,i,DeltaYTM,PastAll = True))
        AlphaAllBack.append(Stat(CalAlpha(1,3,i,DeltaYTM,PastAll = True)))

# Calculate Number of Units and Hedging Errors
LHE = np.zeros((NumPeriod-1,4))
LUnits = np.zeros((NumPeriod-1,4))
LTransactionCost = np.zeros((NumPeriod-1,4))

for i in range(0,len(Units)-1):
     LUnits[i][0] = 100000 * SevenYear.DV01[i] / TwoYear.DV01[i] * BetaAllBack[0][i]
     LUnits[i][1] = 100000 * SevenYear.DV01[i] / ThreeYear.DV01[i] * BetaAllBack[2][i]
     LUnits[i][2] = 100000 * SevenYear.DV01[i] / FiveYear.DV01[i] * BetaAllBack[6][i]
     LUnits[i][3] = 100000 * SevenYear.DV01[i] / TenYear.DV01[i] * BetaAllBack[8][i]

for i in range(0,len(HE)-1):     
     LHE[i][0] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-LUnits[i][0] * (TwoYear.Price[i+1]-TwoYear.Price[i])
     LHE[i][1] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-LUnits[i][1] * (ThreeYear.Price[i+1]-ThreeYear.Price[i])
     LHE[i][2] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-LUnits[i][2] * (FiveYear.Price[i+1]-FiveYear.Price[i])
     LHE[i][3] = 100000 * (SevenYear.Price[i+1]-SevenYear.Price[i])-LUnits[i][3] * (TenYear.Price[i+1]-TenYear.Price[i])

StatLHE = Stat(LHE)

# Q4
# Key Rate Hedging
# We got 5 key rates and 5 bonds
KeyRateClean = np.zeros((len(KeyRate)+1,5)) 
KeyRateClean[0][0] = 0.5
KeyRateClean[0][1] = 1
KeyRateClean[0][2] = 2
KeyRateClean[0][3] = 5
KeyRateClean[0][4] = 10

for i in range(1,len(KeyRate)+1):
    for j in range(1,6):
        KeyRateClean[i][0] = KeyRate['0.50'][i-1]
        KeyRateClean[i][1] = KeyRate['1.00'][i-1]       
        KeyRateClean[i][2] = KeyRate['2.00'][i-1]
        KeyRateClean[i][3] = KeyRate['5.00'][i-1]
        KeyRateClean[i][4] = KeyRate['10.00'][i-1]


# Get Key Rate Curve Function
def GetKeyRateCurve( KeyRate , Interval ):
    length = KeyRate.shape[0]
    width = KeyRate.shape[1]
    period = int((KeyRate[0][width-1] - KeyRate[0][0]) / Interval + 1 )
    keyratecurve = np.zeros((length,period))
   
    for i in range(0,length):
        for j in range(0,period):
            index = np.zeros(5)
            for k in range(0,width):
                index[k] = int(KeyRate[0][k]/Interval) - 1
                if j == index[k]:
                    keyratecurve[i][j] = KeyRate[i][k]
            for k in range(0,width-1):
                if index[k] < j < index[k+1]:
                    keyratecurve[i][j] = KeyRate[i][k] + (KeyRate[i][k+1] - KeyRate[i][k])*(j-index[k])/(index[k+1]-index[k])
    return keyratecurve
        
KR = GetKeyRateCurve( KeyRateClean , 0.5 )
KRC = []
ModifiedKRC =[] 

for i in range(0,5):
    KRC.append(copy.deepcopy(KeyRateClean))
    
for j in range(1,len(KeyRate)):
     KRC[0][j][0] = KRC[0][j][0] + 0.0001
     KRC[1][j][1] = KRC[1][j][1] + 0.0001
     KRC[2][j][2] = KRC[2][j][2] + 0.0001
     KRC[3][j][3] = KRC[3][j][3] + 0.0001
     KRC[4][j][4] = KRC[4][j][4] + 0.0001
     
for i in range(0,5):
    ModifiedKRC.append(GetKeyRateCurve(KRC[i],0.5))

# Q5
# function to get price
    
def GetPrice( Coupon, Maturity , Frequency , KeyRateCurve ):
    length = KeyRateCurve.shape[0]
    width = KeyRateCurve.shape[1]
    Discount = np.zeros((length-1,width))  
    
    for i in range(1,length):
        for j in range(0,width):
            Discount[i-1][j] = 1 / ((1 + KeyRateCurve[i][j] / Frequency )**(j+1))
    
    price = np.zeros(length-1)
    DCF = np.zeros((length, int(Maturity * Frequency)))
    
    for i in range(0,length-1):
        for j in range( 0 , int(Maturity * Frequency)):
            if j == Maturity * Frequency-1:
                DCF[i][j] = Discount[i][j] * (100 + Coupon / Frequency)
            else:
                DCF[i][j] = Discount[i][j] * Coupon / Frequency
            price[i] = price[i] + DCF[i][j]
    
    return price

# Get DV01 and Modified Duration from historical data

DV01 = pd.DataFrame(index = ['KR1','KR2','KR3','KR4','KR5'],columns=["2 year bond","3 year bond","5 year bond","7 year bond","10 year bond"])

for i in range(0,5):
    DV01['2 year bond'][i] = - GetPrice( 6 , 2 , 2 , ModifiedKRC[i] ) + GetPrice( 6 , 2 , 2 , KR )
    DV01['3 year bond'][i] = - GetPrice( 12 , 3 , 2 , ModifiedKRC[i] ) + GetPrice( 12 , 3 , 2 , KR )
    DV01['5 year bond'][i] = - GetPrice( 5 , 5 , 2 , ModifiedKRC[i] ) + GetPrice( 5 , 5 , 2 , KR )
    DV01['7 year bond'][i] = - GetPrice( 10 , 7 , 2 , ModifiedKRC[i] ) + GetPrice( 10 , 7 , 2 , KR )
    DV01['10 year bond'][i] = - GetPrice( 8 , 10 , 2 , ModifiedKRC[i] ) + GetPrice( 8 , 10 , 2 , KR )

MKRDuration = pd.DataFrame(index = ['KR1','KR2','KR3','KR4','KR5'],columns=["2 year bond","3 year bond","5 year bond","7 year bond","10 year bond"])

for i in range(0,5):
    MKRDuration['2 year bond'][i] = (GetPrice( 6 , 2 , 2 , ModifiedKRC[i] ) - GetPrice( 6 , 2 , 2 , KR ))/ GetPrice( 6 , 2 , 2 , KR ) * -10000
    MKRDuration['3 year bond'][i] = (GetPrice( 12 , 3 , 2 , ModifiedKRC[i] ) - GetPrice( 12 , 3 , 2 , KR ))/ GetPrice( 12 , 3 , 2 , KR ) * -10000
    MKRDuration['5 year bond'][i] = (GetPrice( 5 , 5 , 2 , ModifiedKRC[i] ) - GetPrice( 5 , 5 , 2 , KR ))/ GetPrice( 5 , 5 , 2 , KR ) * -10000
    MKRDuration['7 year bond'][i] = (GetPrice( 10 , 7 , 2 , ModifiedKRC[i] ) - GetPrice( 10 , 7 , 2 , KR ))/ GetPrice( 10 , 7 , 2 , KR ) * -10000
    MKRDuration['10 year bond'][i] = (GetPrice( 8 , 10 , 2 , ModifiedKRC[i] ) - GetPrice( 8 , 10 , 2 , KR ))/ GetPrice( 8 , 10 , 2 , KR ) * -10000

# Q6
# Key Rate hedging
# Pick up recent date (2018-2-9)
# Use 6 month T-bill, 1 year T-bill, 2, 3, 5 and 10 year coupon bond
    
HedgingDV01 = pd.DataFrame(index = ['KR1','KR2','KR3','KR4','KR5'],columns=["6 month T-bill","1 year T-bill","2 year bond","3 year bond","5 year bond","7 year bond","10 year bond"])
for i in range(0,5):
    HedgingDV01['6 month T-bill'][i] = -GetPrice( 0 , 0.5 , 2 , ModifiedKRC[i] ) + GetPrice( 0 , 0.5 , 2 , KR )
    HedgingDV01['1 year T-bill'][i] = -GetPrice( 0 , 1 , 2 , ModifiedKRC[i] ) + GetPrice( 0 , 1 , 2 , KR )
    HedgingDV01['2 year bond'][i] = -GetPrice( 6 , 2 , 2 , ModifiedKRC[i] ) + GetPrice( 6 , 2 , 2 , KR )
    HedgingDV01['3 year bond'][i] = -GetPrice( 12 , 3 , 2 , ModifiedKRC[i] ) + GetPrice( 12 , 3 , 2 , KR )
    HedgingDV01['5 year bond'][i] = -GetPrice( 5 , 5 , 2 , ModifiedKRC[i] ) + GetPrice( 5 , 5 , 2 , KR )
    HedgingDV01['7 year bond'][i] = -GetPrice( 10 , 7 , 2 , ModifiedKRC[i] ) + GetPrice( 10 , 7 , 2 , KR )
    HedgingDV01['10 year bond'][i] = -GetPrice( 8 , 10 , 2 , ModifiedKRC[i] ) + GetPrice( 8 , 10 , 2 , KR )

Coefficient = np.zeros((5,6))
b = np.zeros((5,1))

for i in range(0,5):
    b[i] = HedgingDV01['7 year bond'][i][1465]
    for j in range(0,5):
        Coefficient[i][j] = HedgingDV01[HedgingDV01.columns[j]][i][1465]
    Coefficient[i][5] = HedgingDV01['10 year bond'][i][1465]

# Solve it
n =  np.zeros(( 6 , 1 ))  
a = Coefficient
b = b * -100000
c = np.zeros((5,1))

n = np.linalg.lstsq(a, b)
m = np.linalg.svd(a)



Portfolio = pd.DataFrame(index = ["6 month T-bill","1 year T-bill","2 year bond","3 year bond","5 year bond","7 year bond","10 year bond","Total Value","Transaction Cost Base"],columns=['Notion','Price'])
Portfolio['Notion']['6 month T-bill'] = n[0][0][0]
Portfolio['Notion']['1 year T-bill'] = n[0][1][0]
Portfolio['Notion']['2 year bond'] = n[0][2][0]
Portfolio['Notion']['3 year bond'] = n[0][3][0]
Portfolio['Notion']['5 year bond'] = n[0][4][0]
Portfolio['Notion']['7 year bond'] = 100000
Portfolio['Notion']['10 year bond'] = n[0][5][0]

Portfolio['Price']['6 month T-bill'] = GetPrice( 0 , 0.5 , 2 , KR )[1466]
Portfolio['Price']['1 year T-bill'] = GetPrice( 0 , 1 , 2 , KR )[1466]
Portfolio['Price']['2 year bond'] = GetPrice( 6 , 2 , 2 , KR )[1466]
Portfolio['Price']['3 year bond'] = GetPrice( 12 , 3 , 2 , KR )[1466]
Portfolio['Price']['5 year bond'] = GetPrice( 5 , 5 , 2 , KR )[1466]
Portfolio['Price']['7 year bond'] = GetPrice( 10 , 7 , 2 , KR )[1466]
Portfolio['Price']['10 year bond'] = GetPrice( 8 , 10 , 2 , KR )[1466]
 
Portfolio['Price']['Total Price'] = 0
for j in range(0,7):
    Portfolio['Price']['Total Price'] = Portfolio['Price']['Total Price'] + Portfolio.iloc[j,1] * Portfolio.iloc[j,0]

Solution = 0

# Calculate Transaction Cost to get best n
def TCost( Portfolio , n , m , Solution):
    Tcost = 0
    Tcost = abs(Portfolio.iloc[0,1] * (Portfolio.iloc[0,0] + m[2][5][0] * Solution)) + \
        abs(Portfolio.iloc[1,1] * (Portfolio.iloc[1,0] + m[2][5][1] * Solution)) +\
        abs(Portfolio.iloc[2,1] * (Portfolio.iloc[2,0] + m[2][5][2] * Solution)) +\
        abs(Portfolio.iloc[3,1] * (Portfolio.iloc[3,0] + m[2][5][3] * Solution)) +\
        abs(Portfolio.iloc[4,1] * (Portfolio.iloc[4,0] + m[2][5][4] * Solution)) +\
        abs(Portfolio.iloc[5,1] * Portfolio.iloc[5,0]) + \
        abs(Portfolio.iloc[6,1] * (Portfolio.iloc[6,0] + m[2][5][5] * Solution))
    return Tcost 

Cost = np.zeros(100000)
MinCost = 0

# Get lowest 
for i in range(0,100000):
    Solution = i - 50000
    if i == 1:
        MinCost = TCost( Portfolio , n , m , Solution)
    else:
        Cost[i] = TCost( Portfolio , n , m , Solution)
        if Cost[i] < MinCost:
            MinCost = Cost[i]
            j = i
        
MinNotion = np.zeros((6,1))
for i in range(0,6):
    MinNotion[i] = j* m[2][5][i] + n[0][i]   
    
Portfolio['Notion']['6 month T-bill'] = MinNotion[0][0]
Portfolio['Notion']['1 year T-bill'] = MinNotion[1][0]
Portfolio['Notion']['2 year bond'] = MinNotion[2][0]
Portfolio['Notion']['3 year bond'] = MinNotion[3][0]
Portfolio['Notion']['5 year bond'] = MinNotion[4][0]
Portfolio['Notion']['7 year bond'] = 100000
Portfolio['Notion']['10 year bond'] = MinNotion[5][0]
 
Portfolio['Price']['Total Value'] = 0
Portfolio['Price']['Transaction Cost Base'] = MinCost

for j in range(0,7):
    Portfolio['Price']['Total Value'] = Portfolio['Price']['Total Value'] + Portfolio.iloc[j,1] * Portfolio.iloc[j,0]
