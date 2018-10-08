from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as math
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

Y =  np.array([5, 7, 15, 17, 9, 1])
X-1 =  np.array([0, 0, 10, 10, 20, 20])
X-2 =  np.array([0, 0, 100, 100, 400, 400])

#Observations
n = Y.size, print("\n")
print("Total Number of Observations",n)

#For Mean
Y_mean =  Sum(Y)/Y.size
X-1_mean =  Sum(X-1)/X-1.size
X-2_mean =  Sum(X-2)/X-2.size

print("\n")
print("Mean of Y = ",Y_mean)
print("Mean of X-1 = ",X-1_mean)
print("Mean of X-2 = ",X-2_mean)

# x-1, x-2 ,y Calculations
x-1 = X-1 – X-1_mean
x-2 = X-2 – X-2_mean
y = Y - Y_mean

print("\n")
print("x-1 =", x-1)
print("x-2 =", x-2)
print("y =", y)

#Σ Calculation
Σx-1 = np.Sum(x-1)
Σx-2 = np.Sum(x-2)
Σy = np.Sum(y)

#Square Calculations
x-1_sq = np.Sum((x-1)**2)
x-2_sq = np.Sum((x-2)**2)
y_sq = np.Sum((y)**2)

Σx-1y = np.Sum((x-1*y))
Σx-1_sq = np.Sum((x-1)**2)
Σx-2_sq = np.Sum((x-2)**2)
Σx-2y = np.Sum((x-2*y))
Σx-1x-2 = np.Sum((x-1*x-2))

Σx-1y_Σx-2_sq = Σx-1y*Σx-2_sq
Σx-2y_Σx-1x-2 = Σx-2y*Σx-1x-2
Σx-1_sq_Σx-2_sq = Σx-1_sq*Σx-2_sq
Σx-1x-2_sq = (Σx-1x-2)**2

#B1 Calculation
B-1 = (Σx-1y_Σx-2_sq – Σx-2y_-x-1x-2) / (Σx-1_sq_Σx-2_sq – Σx-1x-2_sq)
print("\nBeta1 = ",B-1)

# B2 Calculation
B-2 = ( (Σx-2y*Σx-1_sq) - (Σx-1y*Σx-1x-2) ) / (Σx-1_sq_Σx-2_sq – Σx-1x-2_sq)
print("\nBeta2 = ",B-2)

# B0 Calculation
B-0 = Y_mean - (B-1*X-1_mean) - (B-2*X-2_mean)
print("\nBeta0 = ",B-0)

#No. of parameters
K = 3
print("\nNumber of Parameters B-0, B-1, B-2 : k = ",K)

#Regression Line
Y-Hat = B-0 + B-1*X-1 + B-2*X-2


print("\nY-Hat = {} + {}X1 + {}X2".format(B-0,B-1,B-2))

#Coefficient of Determination
TSS = np.sum((Y - Y_mean)**2)
SSE = np.sum((Y-Hat - Y_mean)**2)
RSS = np.sum((Y - Y_hat)**2)

print("\nTotal Sum of Squares : TSS = ",TSS)
print("Error Sum of Squares : MSS = ",SSE)
print("Residual Sum of Squares : RSS = ",RSS)

R-sq = MSS/TSS
print("\nR-Square = ",R-sq)

# Mean Square Error
MSE = RSS/(n-K)
print("\nMeans Square Error : MSE = ",MSE)

# Variances
V_B-1 = MSE *( (Σx-2_sq) / (Σx-1_sq_Σx-2_sq – Σx-1x-2_sq) )
SE_B-1 = math.sqrt(V_B-1)
print("\nVariance of Beta1 = ",V_B-1)
print("Standard Error of Beta1 : SE(B-1) = ",SE_B-1)

V_B-2 = MSE *( (Σx-1_sq) / (Σx-1_sq_Σx-2_sq – Σx-1x-2_sq) )
SE_B-2 = math.sqrt(V_B-2)
print("\nVariance of Beta2 = ",V_B-2)
print("Standard Error of Beta2 : SE(B-2) = ",SE_B-2)

X-1_mean_sq = X-1_mean**2
X-2_mean_sq = X-2_mean**2
X-1_mean_X-2_mean = X-1_mean*X-2_mean

V_B-0 = MSE * ((1/n) + ((X-1_mean_sq*Σx-2_sq + X-2_mean_sq*Σx-1_sq - 2*X-1_mean_X-2_mean*(Σx-1x-2))/(Σx-1_sq_Σx-2_sq – Σx-1x-2_sq)))
SE_B-0 = math.sqrt(V_B-0)
print("\nVariance of Beta0 = ",V_B-0)
print("Standard Error of Beta0 : SE(B-0) = ",SE_B-0)

#Plotting graph & ANOVA Table
data_frame = pd.DataFrame(
    {
        "Y": Y
        , "X-1": X-1
        , "X-2": X-2
    }
)

Reg = ols(formula="Y ~ X-1 + X-2", data=data_frame)
Fit-2 = Reg.fit()
print("\n", Fit-2.summary())
print("\n", anova_lm(Fit-2))




fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    data_frame["X-1"]
    , data_frame["X-2"]
    , data_frame["Y"]
    , Color="red"
    , Marker="o"
    , Alpha=5
)
ax.set_xlabel("X-1")
ax.set_ylabel("X-2")
ax.set_zlabel("Y")

plt.show()




