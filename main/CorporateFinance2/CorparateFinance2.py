
# Calculate the present value (PV) of your savings for the period that starts today and ends when you retire
PV1 =  3000 / (1 + 0.055)**2

PV2 = (3000 / (0.055- 0.0275)) * (1 -(((1+0.0275)**14) /((1+0.055)**14)))

PV3 = 3000 / (1 + 0.055) + 3000 / (1 + 0.055)**2 + 3000 / (1 + 0.055)**3 + 3000 / (1 + 0.055)**4 + 3000 / (1 + 0.055)**5

PV4 = 3000/(0.055+0.01) * (1- ((1-0.01)**15 / (1+0.055)**15))


PV = PV1 + PV2 + PV3 + PV4

FV = PV * (1+0.055) ** 36

Rr = (1+0.055)/(1+0.0125) - 1

PVd = (533274.16*0.042) / (1-(1/((1+0.042)**20)))

RCF = 39937.30*((1+0.0125)**20)

print(PV1)
print(PV2)
print(PV3)
print(PV4)
print('PV')
print(PV)
print(FV)
print(Rr)
print(PVd)
print(RCF)