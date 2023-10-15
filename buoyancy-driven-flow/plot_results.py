from fenics_fem import *  # Import FEniCS helper functions
import csv
Tf = 10.0
Ra = 10000
betav = 0.5
mm = 40
dt = 0.001

jjj = 1

# Import csv using csv
with open("flow-parameters.csv", 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    my_csv_data = list(spamreader)
    ra_row = my_csv_data[0]
    we_row = my_csv_data[1]
   
betav = 0.5

data_tag = "incomp-flow-"
      

We = float(we_row[1])
Ra = float(ra_row[3])
label_1 = "Ra="+str(Ra)+",We="+str(We)

We = float(we_row[4])
Ra = float(ra_row[3])
label_2 = "Ra="+str(Ra)+",We="+str(We)

We = float(we_row[5])
Ra = float(ra_row[3])
label_3 = "Ra="+str(Ra)+",We="+str(We)

We = float(we_row[1])
Ra = float(ra_row[5])
label_4 = "Ra="+str(Ra)+",We="+str(We)

We = float(we_row[4])
Ra = float(ra_row[5])
label_5 = "Ra="+str(Ra)+",We="+str(We)

We = float(we_row[5])
Ra = float(ra_row[5])
label_6 = "Ra="+str(Ra)+",We="+str(We)

start = time.perf_counter()
x1, ek1, ee1 = load_energy_arrays(jjj, 1, data_tag)

nus1 = load_data_array(jjj, 1, data_tag)
nus1 = -nus1

x2, ek2, ee2 = load_energy_arrays(jjj, 2, data_tag)


nus2 = load_data_array(jjj, 2, data_tag)
nus2 = -nus2

x3, ek3, ee3 = load_energy_arrays(jjj, 3, data_tag)

nus3 = load_data_array(jjj, 3, data_tag)
nus3 = -nus3

x4, ek4, ee4 = load_energy_arrays(jjj, 4, data_tag)

nus4 = load_data_array(jjj, 4, data_tag)
nus4 = -nus4

x5, ek5, ee5 = load_energy_arrays(jjj, 5, data_tag)


nus5 = load_data_array(jjj, 5, data_tag)
nus5 = -nus5

x6, ek6, ee6 = load_energy_arrays(jjj, 6, data_tag)


nus6 = load_data_array(jjj, 6, data_tag)
nus6 = -nus6


print(f"read data arrays in {(time.perf_counter() - start)} s")

# Kinetic Energy
plt.figure(0)
plt.plot(x1, ek1, 'r-', label=r'%s' % label_1)
plt.plot(x2, ek2, 'b-', label=r'%s' % label_2)
plt.plot(x3, ek3, 'c-', label=r'%s' % label_3)
plt.plot(x4, ek4, 'm-', label=r'%s' % label_4)
plt.plot(x5, ek5, 'r--', label=r'%s' % label_5)
plt.plot(x6, ek6, 'b--', label=r'%s' % label_6)
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$E_k$')
plt.savefig("plots/incompressible-flow/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
plt.show()
plt.clf()

# Elastic Energy
plt.figure(1)
plt.plot(x1, ee1, 'r-', label=r'%s' % label_1)
plt.plot(x2, ee2, 'b-', label=r'%s' % label_2)
plt.plot(x3, ee3, 'c-', label=r'%s' % label_3)
plt.plot(x4, ee4, 'm-', label=r'%s' % label_4)
plt.plot(x5, ee5, 'r--', label=r'%s' % label_5)
plt.plot(x6, ee6, 'b--', label=r'%s' % label_6)
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$E_e$')
plt.savefig("plots/incompressible-flow/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
#plt.show()
plt.clf()


# Kinetic and Elastic Energy
i = 0
while i < 6:
    i += 1
    if i == 1:
        x, ek, ee, label = x1, ek1, ee1, label_1
        color_str_ek, color_str_ee = 'r-', 'r--'
    if i == 2:
        x, ek, ee, label = x2, ek2, ee2, label_2
        color_str_ek, color_str_ee = 'b-', 'b--'
    if i == 3:
        x, ek, ee, label = x3, ek3, ee3, label_3
        color_str_ek, color_str_ee = 'c-', 'c--'
    if i == 4:
        x, ek, ee, label = x4, ek4, ee4, label_4
        color_str_ek, color_str_ee = 'b-', 'b--'
    if i == 5:
        x, ek, ee, label = x5, ek5, ee5, label_5
        color_str_ek, color_str_ee = 'c-', 'c--'
    if i == 6:
        x, ek, ee, label = x6, ek6, ee6, label_6
        color_str_ek, color_str_ee = 'g-', 'g--'
    
    ek_max = max(ek)
    ee_max = max(ee)
    limit = max([ek_max, ee_max])
    plt.figure(i)
    plt.plot(x, ek, color_str_ek, label='${E_k}$' )
    plt.plot(x, ee, color_str_ee, label='${E_e}$' )
    plt.yscale("log")
    plt.ylim([1e-4, 1.2*limit])
    plt.legend(loc='best') #title=label
    plt.xlabel('$t$')
    plt.ylabel('${E_k}$/${E_e}$')

    plt.savefig("plots/incompressible-flow/KineticAndElasticEnergy"+str(label)+"-log"+".png")

    #plt.show()
    plt.clf()


"""
# Kinetic and Elastic Energy
i = 0
while i < 6:
    i += 1
    if i == 1:
        x, ek, ee, label = x1, ek1, ee1, label_1
        color_str_ek, color_str_ee = 'r-', 'r--'
    if i == 2:
        x, ek, ee, label = x2, ek2, ee2, label_2
        color_str_ek, color_str_ee = 'b-', 'b--'
    if i == 3:
        x, ek, ee, label = x3, ek3, ee3, label_3
        color_str_ek, color_str_ee = 'c-', 'c--'
    if i == 4:
        x, ek, ee, label = x4, ek4, ee4, label_4
        color_str_ek, color_str_ee = 'b-', 'b--'
    if i == 5:
        x, ek, ee, label = x5, ek5, ee5, label_5
        color_str_ek, color_str_ee = 'c-', 'c--'
    if i == 6:
        x, ek, ee, label = x6, ek6, ee6, label_6
        color_str_ek, color_str_ee = 'g-', 'g--'
    
    ek_max = max(ek)
    ee_max = max(ee)
    fig, ax1 = plt.subplots()
    ax1.plot(x, ek, color_str_ek, label=r'%s' % label)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$E_k$')
    ax1.set_ylim([0, 3*ek_max])
    ax2 = ax1.twinx() 
    ax2.plot(x, ee, color_str_ee, label=r'%s' % label)
    ax2.set_ylabel('$E_e$')
    ax2.set_ylim([0, 1.2*ee_max])

    fig.tight_layout()  
    plt.legend(loc='best')
    fig.savefig("plots/incompressible-flow/KineticAndElasticEnergy"+str(label)+".png")

    plt.show()
    plt.clf()
"""