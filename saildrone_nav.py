import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from saildrone_hydro import get_vals #hydrodynamic physics for forces and torque

#Initialising physical constants provided
M = 2500                        #Mass (kg)
I = 10000                       #Rotational Inertia (kg·m²)
A = 15                          #Saildone area (m²)
rho = 1.225                     #Air density (kg/m³)
sail_offset = 0.1               #Sail offset from center of rotation (m)
v_wind = np.array([-6.7, 0])    #Wind vector for easerly winds (m/s)

#calculating aerodynamic forces
def saildrone_aero(v_drone, theta, B_sail):
    v_appWind = v_wind - v_drone #calculates apparent wind experienced by the sail drone
    v_appWind_mag = np.linalg.norm(v_appWind)
    if v_appWind_mag == 0:
        return np.array([0.0, 0.0]), 0.0 #if apparent wind is zero, no force is experienced

    #convert apparent wind vector relative to a b axis (drones heading) from x y axis: compute angle of attack of the drone
    Rmatrix = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    v_appWind_body = np.matmul(Rmatrix, v_appWind)

    alpha = B_sail - np.arctan2(v_appWind_body[1], v_appWind_body[0])  #calculating angle of attack (alpha) between sail and direction of apparent wind 

    #calculating aerodynamic coefficients of drag and lift respectively
    CD = 1 - np.cos(2 * alpha)
    CL = 1.5 * np.sin(2 * alpha + 0.5 * np.sin(2 * alpha))

    #calculating pressure forces
    F_mag = 0.5 * rho * A * v_appWind_mag**2
    F_body = F_mag * np.array([CD, CL])
    #CD because drag acts along the flow direction (resisting motion).
    #CL because lift is perpendicular to the flow direction (side force from the sail).

    F_total = np.matmul(np.transpose(Rmatrix),F_body) #transfrom matrix back 
    
    torque = F_body[1] * sail_offset #torque due to sail offset

    return F_total, torque


def saildrone_ODE(t, z):
    
    #control setting at time t
    B_sail, B_rudder = control_stat(t)

    x, vx, y, vy, theta, gamma = z
    v_drone = np.array([vx, vy])

    F_sail, torque_sail = saildrone_aero(v_drone, theta, B_sail)
    F_hydro, torque_hydro = get_vals(v_drone, theta, gamma, B_rudder)

    #using newtons second law to compute accelerations due aero and hydro forces
    ax = (F_sail[0] + F_hydro[0])/M
    ay = (F_sail[1] + F_hydro[1])/M
    alpha = (torque_sail + torque_hydro) / I

    return [vx, ax, vy, ay, gamma, alpha] #returns derivatives of the 6 1st order ODEs

#course sector strategy
def control_stat(t):
    if t < 60:
        return np.deg2rad(-45), np.deg2rad(0)      #sector A
    elif t < 65:
        return np.deg2rad(-22.5), np.deg2rad(2.1)  #sector B
    else:
        return np.deg2rad(-22.5), np.deg2rad(0)    #sector C


#initialising ODE states in order [x, vx, y, vy, theta, gamma]
z0 = [0, 0, 0, 2.9, 0, 0]
t_run = (0, 100)

#solving ODE
sol = solve_ivp(saildrone_ODE, t_run, z0, method='RK45')

#plot
plt.figure()
plt.plot(sol.y[0], sol.y[2], label='Course')

# Sector A (t=0), B (t=60), C (t=65)
pathA = 0
pathB = np.searchsorted(sol.t, 60)
pathC = np.searchsorted(sol.t, 65)

plt.scatter(sol.y[0][pathA], sol.y[2][pathA], c='b', marker='o', label = 'A (Northward Travel)')
plt.scatter(sol.y[0][pathB], sol.y[2][pathB], c='r', marker='o', label = 'B (Clockwise Turn)')
plt.scatter(sol.y[0][pathC], sol.y[2][pathC], c='b', marker='o', label = 'C (Upwind Travel)')

plt.xlabel('x, East (m)')
plt.ylabel('y, North (m)')
plt.title('Saildrone course path')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
