# this python programme plots the stability data generated by compEWM.py
import numpy as np
import matplotlib.pyplot as plt


x = [0, 0.1, 0.25, 0.5, 1.0]

# Stability

y_Re25 = [0.019, 0.020, 0.074, 0.188, 0.550]
y_Re50 = [0.019, 0.059, 0.096, 0.186, 0.372]
y_Re100 = [0.019, 0.054, 0.094, 0.181, 0.366]
y_Re200 = [0.019, 0.1354, 0.376, 0.437, 0.552]


y_Ma0001 = [0.019, 0.059, 0.094, 0.183, 0.366]
y_Ma001 = [0.019, 0.059, 0.097, 0.186, 0.372]
y_Ma005 = [0.024, 0.081, 0.122, 0.214, 0.406]
#y_Ma01 = [0.019, 0.081, 0.144, 0.413]



y_lambda0 = [0.0149, 0.161,  0.595, 0.952, 1.511, 1.854]
y_lambda01 = [0.149, 1.157, 1.599, 7.698, 6.379, 3.211]
y_lambda015 = [0.150,  2.159, 3.179, 10.344, 8.971,  5.261]


# Plot data
"""
plt.figure(0)
plt.plot(x, y_Re25 , 'o--', label=r'$Re = 25$')
plt.plot(x, y_Re50 , 'o--', label=r'$Re = 50$')
plt.plot(x, y_Re100 , 'o--', label=r'$Re = 100$')
plt.plot(x, y_Re200 , 'o--', label=r'$Re = 200$')
plt.legend(loc='best')
plt.xlabel('$We$', fontsize=16)
plt.ylabel('$\chi$', fontsize=16)
plt.savefig("EWMStability_Re.png")
plt.clf()
"""

plt.figure(1)
plt.plot(x, y_Ma0001 , 'o--', label=r'$Ma=0.001$')
plt.plot(x, y_Ma001 , 'o--', label=r'$Ma=0.01$')
plt.plot(x, y_Ma005 , 'o--', label=r'$Ma=0.05$')
#plt.plot(x, y_Ma01 , 'o--', label=r'$Ma=0.1$')
plt.legend(loc='best')
plt.xlabel('$We$', fontsize=16)
plt.ylabel('$\chi$', fontsize=16)
plt.savefig("EWMStability_Ma.png")
plt.clf()


"""
plt.figure(0)
plt.plot(x, y_lambda0 , 'o--', label=r'$\lambda_D=0$')
plt.plot(x, y_lambda01 , 'o--', label=r'$\lambda_D=0.1$')
plt.plot(x, y_lambda015 , 'o--', label=r'$\lambda_D=0.15$')
plt.legend(loc='best')
plt.xlabel('$We$', fontsize=16)
plt.ylabel('$\chi$', fontsize=16)
plt.savefig("FENE-P-MPstability.png")
plt.clf()
"""
