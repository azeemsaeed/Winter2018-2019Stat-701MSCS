from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as math
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

CPU = np.array([500, 700, 150, 175, 980, 180])
No_Of_Processors = np.array([0, 2, 4, 6, 8, 10])
No_Of_Rams = np.array([0, 8, 16, 24, 32, 64])

#Total Observations
n = CPU.size
print("\nNumber of Observations : n = ",n)

#Finding mean
CPU_mean = sum(CPU)/CPU.size
No_Of_Processors_mean = sum(No_Of_Processors)/No_Of_Processors.size
No_Of_Rams_mean = sum(No_Of_Rams)/No_Of_Rams.size

print("\nMean for CPU = ",CPU_mean)
print("Mean for No_Of_Processors = ",No_Of_Processors_mean)
print("Mean for No_Of_Rams = ",No_Of_Rams_mean)

#Calculating x-1, x-2, y
x-1 = No_Of_Processors - No_Of_Processors_mean
x-2 = No_Of_Rams - No_Of_Rams_mean
y = CPU - CPU_mean

print("\nx-1 =",x-1)
print("x-2 =",x-2)
print("Y =",y)

#Calculating Sum
Σx-1 = np.sum(x-1)
Σx-2 = np.sum(x-2)
Σy = np.sum(y)

#Calculating Squares
x-1_sq = np.sum((x-1)**2)
x-2_sq = np.sum((x-2)**2)
y_sq = np.sum((y)**2)

Σx-1y = np.sum((x-1*y))
Σx-1_sq = np.sum((x-1)**2)
Σx-2_sq = np.sum((x-2)**2)
Σx-2y = np.sum((x-2*y))
Σx-1x-2 = np.sum((x-1*x-2))

Σx-1y_Σx-2_sq = Σx-1y*Σx-2_sq
Σx-2y_Σx-1x-2 = Σx-2y*Σx-1x-2
Σx-1_sq_Σx-2_sq = Σx-1_sq*Σx-2_sq
Σx-1x-2_sq = (Σx-1x-2)**2

#Calculating BETA1
B-1 = (Σx-1y_Σx-2_sq - Σx-2y_Σx-1x-2) / (Σx-1_sq_Σx-2_sq - Σx-1x-2_sq)
print("\nBeta1 = ",B-1)

#Calculating BETA2
B-2 = ( (Σx-2y*Σx-1_sq) - (Σx-1y*Σx-1x-2) ) / (Σx-1_sq_Σx-2_sq - Σx-1x-2_sq)
print("Beta2 = ",B-2)

#Calculating BETA0
B-0 = CPU_mean - (B-1*No_Of_Processors_mean) - (B-2*No_Of_Rams_mean)
print("Beta0 = ",B-0)

#Total parameters
k = 3
print("\nNumber of Parameters B0, B1, B2 : k = ",k)

#Calculated Y
Y_hat = B-0 + B-1*No_Of_Processors + B-2*No_Of_Rams


print("\nY_hat = {} + {}X1 + {}X2".format(B-0,B-1,B-2))

#Calculating coefficient of determination
TSS = np.sum((CPU - CPU_mean)**2)
SSE = np.sum((Y_hat - CPU_mean)**2)
RSS = np.sum((CPU - Y_hat)**2)

print("\nTotal Sum of Squares : TSS = ",TSS)
print("ERROR Sum of Squares : SSE = ",SSE)
print("Residual Sum of Squares : RSS = ",RSS)

R_sq = SSE/TSS
print("\nR_square = ",R_sq)

#Calculating mean square error
MSE = RSS/(n-k)
print("\nMeans Square Error : MSE = ",MSE)

#Calculating Variances
V_B1 = MSE *( (Σx2_sq) / (Σx-1_sq_Σx-2_sq - Σx-1x-2_sq) )
SE_B1 = math.sqrt(V_B1)
print("\nVariance of Beta1 = ",V_B1)
print("Standard Error of Beta1 : SE(B1) = ",SE_B1)

V_B2 = MSE *( (Σx-1_sq) / (Σx-1_sq_Σx-2_sq - Σx-1x-2_sq) )
SE_B2 = math.sqrt(V_B2)
print("\nVariance of Beta2 = ",V_B2)
print("Standard Error of Beta2 : SE(B2) = ",SE_B2)

No_Of_Processors_mean_sq = No_Of_Processors_mean**2
No_Of_Rams_mean_sq = No_Of_Rams_mean**2
Number_of_Signals_mean_Number_of_Nodes_mean = No_Of_Processors_mean*No_Of_Rams_mean

V_B0 = MSE * ((1/n) + ((No_Of_Processors_mean_sq*Σx-2_sq + No_Of_Rams_mean_sq*Σx-1_sq - 2*No_Of_Processors_mean_No_Of_Rams_mean*(Σx-1x-2))/(Σx-1_sq_Σx-2_sq - Σx-1x-2_sq)))
SE_B0 = math.sqrt(V_B0)
print("\nVariance of Beta0 = ",V_B0)
print("Standard Error of Beta0 : SE(B0) = ",SE_B0)

#Plotting graph and ANOVA table
data_frame = pd.DataFrame(
    {
        "Y": CPU
        , "X1": No_Of_Processors
        , "X2": No_Of_Rams
    }
)

Reg = ols(formula="CPU ~ No_Of_Processors + No_Of_Rams", data=data_frame)
Fit2 = Reg.fit()
print("\n", Fit2.summary())
print("\n", anova_lm(Fit2))




fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    data_frame["X1"]
    , data_frame["X2"]
    , data_frame["Y"]
    , color="Red"
    , marker="o"
    , alpha=5
)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

plt.show()




