from fenics_fem import *  # Import FEniCS helper functions
import csv
Tf = 20.0
Ra = 10000
betav = 0.5
mm = 40
dt = 0.001
jjj = 1

# Import csv using csv
with open("flow-parameters.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    my_csv_data = list(spamreader)
    re_row = my_csv_data[0]
    we_row = my_csv_data[1]
    ma_row = my_csv_data[2]


data_tag = "comp-flow" 
betav = 0.5
We = float(we_row[4])
Re = float(re_row[4])
 

Ma = float(ma_row[1])
label_1 = "Ma = "+str(Ma) #"Re= "+str(Re)+", We = "+str(We)+", Ma = "+str(Ma) 

Ma = float(ma_row[2])
label_2 = "Ma = "+str(Ma) #"Re= "+str(Re)+", We = "+str(We)+", Ma = "+str(Ma) 

Ma = float(ma_row[3])
label_3 = "Ma = "+str(Ma) #"Re= "+str(Re)+", We = "+str(We)+", Ma = "+str(Ma) 

Ma = float(ma_row[4])
label_4 = "Ma = "+str(Ma) #"Re= "+str(Re)+", We = "+str(We)+", Ma = "+str(Ma) 

start = time.perf_counter_ns()
x1, ek1, ee1 = load_energy_arrays(1, data_tag)
x2, ek2, ee2 = load_energy_arrays(2, data_tag)
x3, ek3, ee3 = load_energy_arrays(3, data_tag)
x4, ek4, ee4 = load_energy_arrays(4, data_tag)



print(f"read data arrays in {(time.perf_counter() - start)} s")

# Kinetic Energy
plt.figure(0)
plt.plot(x1, ek1, 'r-', label=r'%s' % label_1)
plt.plot(x2, ek2, 'b-', label=r'%s' % label_2)
plt.plot(x3, ek3, 'c-', label=r'%s' % label_3)
plt.plot(x4, ek4, 'm-', label=r'%s' % label_4)
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$E_k$')
plt.savefig("plots/energy/MaKineticEnergyRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+"t="+str(Tf)+".png")
plt.show()
plt.clf()

# Elastic Energy
plt.figure(1)
plt.plot(x1, ee1, 'r-', label=r'%s' % label_1)
plt.plot(x2, ee2, 'b-', label=r'%s' % label_2)
plt.plot(x3, ee3, 'c-', label=r'%s' % label_3)
plt.plot(x4, ee4, 'm-', label=r'%s' % label_4)
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$E_e$')
plt.savefig("plots/energy/MaElasticEnergyRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"dt="+str(dt)+"t="+str(Tf)+".png")
plt.show()
plt.clf()