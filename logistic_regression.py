import numpy as np
import pandas as pd
import statsmodels.api as sm
import math as math

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#added for Logistic regressions lesson 2.4.2
loansData.to_csv('loansData_clean.csv', header=True, index=False)

#cleaning long way
#x = loansData['Interest.Rate'][0:5].values[1]
#x = x.rstrip('%')
#x = float(x)
#x = x/100
#x = round(x, 4)
#print x

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
loansData['Interest.Rate'] = cleanInterestRate
#print loansData['Interest.Rate'][0:5]

cleanLoansLength = loansData['Loan.Length'][0:5].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoansLength
#print loansData['Loan.Length'][0:5]

#cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))
#cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])
#loansData['FICO.Range'] = cleanFICORange
#print loansData['FICO.Range'][0:5]

loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]


intrate = loansData['Interest.Rate'] # dependednt
loansData['intrate'] = intrate #adding intrate
loanamt = loansData['Amount.Requested']
loansData['loanamt'] = loanamt #adding loanamt
fico = loansData['FICO.Score']
loansData['fico'] = fico #adding fico
IR_TF = loansData['Interest.Rate'].map(lambda x: 1 if x >= 0.12 else 0)
loansData['IR_TF'] = IR_TF #adding IR_TF
intercept = loansData['Interest.Rate'].map(lambda x: 1 if x>=0 else 0) #how else could i create a column of 1's?
loansData['intercept'] = intercept #adding intercept to df

ind_vars = ['loanamt', 'fico', 'intercept']

#print loansData[loansData['Interest.Rate'] == 10].head() # should all be True
#print loansData[loansData['Interest.Rate'] == 13].head() # should all be False


#transpose data
#The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

#print intrate
#print loanamt
#print fico
x = np.column_stack([x1,x2])
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print f.summary()


logit = sm.Logit(loansData['IR_TF'], loansData[ind_vars]) #dependent var/indepent vars
result = logit.fit()
coeff = result.params
print(coeff)

#print type(coeff)
#print coeff[0]

#works
#def fun(a, b):
#	k = 1/(1 + math.exp(60.125 - 0.087423*(a) + 0.000174*(b)))
#	print (k)

def fun(a, b, coeff):
		k =1/(1 + math.exp(coeff[2] + (coeff[1]*(a)) + (coeff[0]*(b))))
		print (k)



print "The probability of obtaining a loan at <= 12% interest is"
fun(720, 10000, coeff)





 

