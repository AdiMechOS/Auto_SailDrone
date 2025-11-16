import numpy as np
import matplotlib.pyplot as plt


m = 2.0       #mass (kg)
c = 0.1       #damping coefficient (Ns/m)
k = 0.2       #spring constant (N/m)

def state_deriv_msd(t, z):
    
    x = z[0]
    v = z[1]

    dz1 = v
    dz2 = -(c/m)*v - (k/m)*x

    return np.array([dz1, dz2])


def step_euler(state_deriv, t, dt, z):

    return z + dt * state_deriv(t, z)


def step_rk(state_deriv, t, dt, z):

    k1 = state_deriv(t, z)
    k2 = state_deriv(t + dt/2, z + dt*k1/2)
    k3 = state_deriv(t + dt/2, z + dt*k2/2)
    k4 = state_deriv(t + dt,   z + dt*k3)

    return z + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def solve_ivp(state_deriv, t0, tmax, dt, z0, method='RK'):
    
    t = np.array([t0])
    z = z0.reshape(2,1)

    step = step_rk if method.upper() == 'RK' else step_euler

    n = 0
    while t[n] <= tmax:
        t = np.append(t, t[-1] + dt)
        z_next = step(state_deriv, t[n], dt, z[:, n])
        z = np.append(z, z_next.reshape(2,1), axis=1)
        n += 1

    return t, z


if __name__ == "__main__":

    t0 = 0
    tmax = 100
    dt = 0.1
    z0 = np.array([10, 0])
    
    t, z = solve_ivp(state_deriv_msd, t0, tmax, dt, z0, method='RK')

    plt.figure(figsize=(8,4))
    plt.plot(t, z[0,:], 'blue')
    plt.plot(t, z[1,:], 'red')
    plt.xlabel("Time (s)")
    plt.ylabel("State variables")
    plt.legend(["Displacement x (m)", "Velocity (m/s)"])
    plt.grid()
    plt.tight_layout()
    plt.show()

t_e, z_e = solve_ivp(state_deriv_msd, 0, 100, 1.0, z0, method='Euler')
t_r, z_r = solve_ivp(state_deriv_msd, 0, 100, 1.0, z0, method='RK')

plt.plot(t_e, z_e[0], 'r--', label="Euler")
plt.plot(t_r, z_r[0], 'b-', label="RK4")
plt.legend()
plt.show()
