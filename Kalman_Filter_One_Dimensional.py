import numpy as np
n = 4 #number of iterations
true_val = 72
estimate = np.zeros((n+1,1)).reshape(n+1,1)
estimate[0] = 68
#print(estimate)
estimate_error = np.zeros((n+1,1)).reshape(n+1,1)
estimate_error[0] = 2
#print(estimate_error)
measured = np.array([0, 75, 71, 70, 74]).reshape(n+1,1)
measured_error = np.array([0, 4, 4, 4, 4]).reshape(n+1,1)


kalman_gain = np.zeros((n+1,1)).reshape(n+1,1)


for i in range(1,n+1):
    
    
    kalman_gain[i] = (estimate_error[i-1])/(estimate_error[i-1] + measured_error[i])
    
    estimate[i] = estimate[i-1] + ( kalman_gain[i] * (measured[i] - estimate[i-1]) )
    
    estimate_error[i] = (1 - kalman_gain[i]) * estimate_error[i-1]
    

print(estimate)
print(kalman_gain)
print(estimate_error)
