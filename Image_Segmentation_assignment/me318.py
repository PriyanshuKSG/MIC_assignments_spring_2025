import numpy as np

pd = np.array([0, 0.4, 1.2, 2, 2.6, 3])
t = np.array([13.24, 13.14, 13.31, 13.44, 13.43, 14.06])
te = np.array([32.33, 30.36, 25.88, 22.39, 20.26, 19.44])

hd = pd*11.36
hd += 0.61

hs = 1.24
hf = 0.2

ht = hd + hf + hs

V = 0.08*0.1
Q = V/t
Q *= 1e4

Pout = 0.088 * 9.81 * Q * ht
Pout /= 1000

Pelec = 10*9/8
Pelec /= te

Pin = Pelec * 0.8

efficiency = Pout / Pin
efficiency *= 100

print("\nhd = ", hd)
print("\nht = ", ht)
print("\nQ = ", Q)
print("\nPelec = ", Pelec)
print("\nPin = ", Pin)
print("\nPout = ", Pout)
print("\neta = ", efficiency)