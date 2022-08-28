# ENGR-E599: Final Project, Ethan Eldridge

import numpy as np
import numpy.linalg as la
import random as rn
import math

# Prior state relation matrix
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Change in time at each step (seconds)
dt = 1

# Implementing system dynamics model
def dynamics(Vl, Vr, dt, lastEst):
    global A
    theta = lastEst[2]
    u = np.array([Vr, Vl])
    B = np.array([[0.5*dt*math.cos(math.radians(theta)), 0.5*dt*math.cos(math.radians(theta))],
                  [0.5*dt*math.sin(math.radians(theta)), 0.5*dt*math.sin(math.radians(theta))],
                  [(1/0.6985)*dt, -(1/0.6985)*dt]])
    state = np.matmul(A, lastEst)+np.matmul(B, u)
    return state

def simPath(steps):
    global dt, thetas
    Vl = 0
    Vr = 0
    actions = ["F", "R", "L"]
    weights = [0.95, 0.025, 0.025]
    estimates = [[0,0,0]]
    positions = [[0,0]]
    encoders = [[Vl, Vr]]
    last = "F"
    # For each step in the path
    for i in range(steps):
        # Adaptively updating weights for added path realism
        if last=="R":
            altered = [0.6, 0.3, 0.1]
            direction = rn.choices(actions, weights=altered, k=1)
        elif last =="L":
            altered = [0.6, 0.1, 0.3]
            direction = rn.choices(actions, weights=altered, k=1)
        else:
            direction = rn.choices(actions, weights=weights, k=1)
        # Generating encoder values according to decision
        if direction[0]=="F":
            last="F"
            Vl = 2
            Vr = 2
        elif direction[0]=="R":
            last="R"
            Vl = 3
            Vr = 1
        else:
            last="L"
            Vl = 1
            Vr = 3
        # Recording generated values
        encoders.append([Vl, Vr])
        results = dynamics(Vl, Vr, dt, estimates[i])
        # Correcting compass heading
        if results[2] < 0: results[2] = results[2] + 360
        estimates.append(results)
        positions.append(results[0:2])
    return positions, encoders


# Generating ground-truth position information
STEPS = 100

pos, enc = simPath(STEPS)

posX = [pos[i][0] for i in range(len(pos))]
posY = [pos[i][1] for i in range(len(pos))]

# Visualizing ground-truth position information
import matplotlib.pyplot as plt
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Simulated Ground Truth Path")
plt.legend()
plt.show()

def add_noise(vals, stdev):
    noise = np.random.normal(0, stdev, len(vals))
    alt = np.array(vals)+noise
    return alt

# Simulating noisy sensor data
GPS_NOISE = 0.25
gps = list(zip(add_noise(posX, GPS_NOISE), add_noise(posY, GPS_NOISE)))

ENC_NOISE = 0.005
VL_noisy = add_noise([enc[i][0] for i in range(len(enc))], ENC_NOISE)
VR_noisy = add_noise([enc[i][1] for i in range(len(enc))], ENC_NOISE)
enc_noisy = list(zip(VL_noisy, VR_noisy))

# Getting path generated from noisy encoders
estimates = [[0,0,0]]
enc_X = []
enc_Y = []
dyn = []
for i in range(len(VR_noisy)):
    res = dynamics(VL_noisy[i], VR_noisy[i], dt, estimates[i])
    estimates.append(res)
    enc_X.append(res[0])
    enc_Y.append(res[1])

# Visualizing noisy encoder data 
import matplotlib.pyplot as plt
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.plot(enc_X, enc_Y, label="Encoder Path (Noisy)", color="blue")
plt.title("Noisy Encoder Data vs Ground Truth Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()

# Visualizing noisy encoder data 
import matplotlib.pyplot as plt
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.scatter([gps[i][0] for i in range(len(gps))], [gps[i][1] for i in range(len(gps))], color = "red", alpha = 0.2, label="GPS (Noisy)")
plt.title("Noisy GPS Data vs Ground Truth Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()

# Measurement signal
z = gps

# State projection input
u = enc_noisy

N = len(gps)


#Initializing remaining matrices

#State-measurement relation matrix
C = np.array([[1, 0, 0],
              [0, 1, 0]])

#Process noise
Q = np.array([[1*(10**-9), 0, 0], 
              [0, 1*(10**-9), 0], 
              [0, 0, 1]])

#Measurement noise
R = np.array([[1*(10**-7), 0],
              [0, 1*(10**-7)]])

I = np.identity(3)


#Error Covariance
P_n = np.array([[1*(10**-9), 0, 0],
                [0, 1*(10**-9), 0],
                [0, 0, 1]])

#Initializing estimates to first GPS position 
X = np.array([[gps[0][0], gps[0][1], 0]])

# For each step
for i in range(N):

    # Making adjustments for initialization case
    if i == 0:
        THETA = X[i][2]
    else:
        THETA = X[i-1][2]
    
    # Correcting compass heading 
    if THETA < 0: 
        THETA = THETA*-1
    else:
        THETA = 360-THETA

    # System dynamics input relation matrix
    B = np.array([[0.5*dt*math.cos(math.radians(THETA)), 0.5*dt*math.cos(math.radians(THETA))],
                  [0.5*dt*math.sin(math.radians(THETA)), 0.5*dt*math.sin(math.radians(THETA))],
                  [(1/0.6985)*dt, -(1/0.6985)*dt]])

    # State Prediction
    if i == 0:
        X[i] = np.matmul(A, X[i])+np.matmul(B, u[i])
    else:
        xhat = np.matmul(A, X[i-1])+np.matmul(B, u[i])
        X = np.append(X, [xhat], axis=0)        
        
    P = np.matmul(np.matmul(A, P_n), A.T)+Q

    # Correction
    K = np.matmul(np.matmul(P, C.T), la.inv(np.matmul(np.matmul(C, P), C.T)+R))
    X[i] = X[i] + np.matmul(K, (z[i] - np.matmul(C, X[i])))
    P = np.matmul((I-np.matmul(K, C)),P)

    P_n = P


import matplotlib.pyplot as plt
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.plot([X[i][0] for i in range(len(X))], [X[i][1] for i in range(len(X))], label="Filter", color="blue")
plt.scatter([gps[i][0] for i in range(len(gps))], [gps[i][1] for i in range(len(gps))], color = "red", alpha = 0.2, label="GPS (Noisy)")
plt.title("Final Simulation of Kalman Filter Along Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()


# DEAD ZONE SIMULATION


# Initializing obstruction noise
OBS_NOISE = GPS_NOISE*10
obs = list(zip(add_noise(posX, OBS_NOISE), add_noise(posY, OBS_NOISE)))

# Recreating path with partial GPS signal obstruction
final_gps = []
obstructed = False
# Keeping track of x-range that is obstructed
toggleX = []
for i in range(len(gps)):
  # Every 1/4 of the path, the signal obstruction is toggled
  point = int(STEPS/4)
  if i%point == 0: 
    obstructed = not(obstructed)
    toggleX.append(posX[i])
  if obstructed:
    final_gps.append(obs[i])
  else:
    final_gps.append(gps[i])


# Visualizing new GPS data
import matplotlib.pyplot as plt
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.scatter([final_gps[i][0] for i in range(len(final_gps))], [final_gps[i][1] for i in range(len(final_gps))], color = "red", alpha = 0.2, label="GPS (Noisy)")
# Plotting dead zones
last = False
for x in range(len(toggleX)):
  if x+1 == len(toggleX):
    plt.axvspan(toggleX[x], toggleX[-1], color="red", alpha=0.1)
    break
  else:
    if last==False:
      plt.axvspan(toggleX[x], toggleX[x+1], color="red", alpha=0.1)
      last=True
    else:
      last = False
plt.title("Noisy GPS Data vs Partially Obstructed \nGround Truth Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()

# Resetting measurement signal to newly compiled GPS data 
z = final_gps

X = filter()

# Visualizing new filtering process with obstructed signals 
plt.plot(posX, posY, color="black", label="Ground Truth")
plt.plot([X[i][0] for i in range(len(X))], [X[i][1] for i in range(len(X))], label="Filter", color="blue")
plt.scatter([final_gps[i][0] for i in range(len(final_gps))], [final_gps[i][1] for i in range(len(final_gps))], color = "red", alpha = 0.2, label="GPS (Noisy)")
# Plotting dead zones
last = False
for x in range(len(toggleX)):
  if x+1 == len(toggleX):
    plt.axvspan(toggleX[x], toggleX[-1], color="red", alpha=0.1)
  else:
    if last==False:
      plt.axvspan(toggleX[x], toggleX[x+1], color="red", alpha=0.1)
      last=True
    else:
      last = False
plt.title("Final Simulation of Kalman Filter Along \nPartially Obstructed Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.show()
  