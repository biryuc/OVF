import numpy as np
import sys
def floatEpsilon(eps=np.float32):
    counter=0
    machine_epsilon = eps(1)
    while eps(1)+eps(machine_epsilon) != eps(1) :
        machine_epsilon_last = machine_epsilon
        machine_epsilon = eps(machine_epsilon) / eps(2)
        counter += 1
    return [machine_epsilon_last,counter-1]

def doubleEpsilon(eps=np.float64):
    counter=0
    machine_epsilon = eps(1)
    while (eps(1)+eps(machine_epsilon) != eps(1)):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = eps(machine_epsilon) / eps(2)
        counter+=1
    return [machine_epsilon_last,counter-1]

def maxfloatPowerE():
    maxpower=[np.float32(1.0)]
    i=0
    while maxpower[i] != np.float32('inf'):
      maxpower.append(np.float32(maxpower[i]*2))
      i+=1
    return maxpower[i-1]

def maxdoublePowerE():
    maxpower=[np.float64(1.0)]
    i=0
    while maxpower[i] != float('inf'):
      maxpower.append(float(maxpower[i]*2))
      i+=1
    return maxpower[i-1]
def minfloatPowerE():
    minpower = [np.float32(1.0)]
    i = 0
    while minpower[i] != np.float32(0.0):
        minpower.append(np.float32(minpower[i]/2))
        i += 1
    return minpower[i - 1 - floatResult[1]]
def mindoublePowerE():
    minpower = [np.float64(1.0)]
    i = 0
    while minpower[i] != np.float64(0.0):
        minpower.append(np.float64(minpower[i]/2))
        i += 1
    return minpower[i - 1 - doubleResult[1]]

###################################
floatResult = floatEpsilon()
doubleResult = doubleEpsilon()
maxpowerfloatResult = maxfloatPowerE()
maxpowerdoubleResult = maxdoublePowerE()
minpowerfloatResult = minfloatPowerE()
minpowerdoubleResult = mindoublePowerE()
###################################

comparefloatList=[1.0,1.0+floatResult[0]/2,1.0+floatResult[0],1.0+floatResult[0]+floatResult[0]/2]
comparedoubleList=[1.0,1.0+doubleResult[0]/2,1.0+doubleResult[0],1.0+doubleResult[0]+doubleResult[0]/2]
def compareFunc(list):
    for startIndex in range(len(list) -1 ):
        smallestIndex = startIndex
        currentIndex=startIndex+1
        for currentIndex in range(len(list)-1):
            currentIndex += 1
            if list[currentIndex]<list[smallestIndex]:
                smallestIndex = currentIndex
    print(list)

print("##############  EPSILON AND MANTISSA FLOAT #########################")
print("EPSILON FLOAT = ",floatResult[0],"        MANTISSA FLOAT (BITS)= ",floatResult[1])
print("############## EPSILON AND MANTISSA DOUBLE ######################")
print("EPSILON DOUBLE = ",doubleResult[0],"MANTISSA DOUBLE (BITS)= ",doubleResult[1])
print("############# MAX POWER  ######################")
print("MAX POWER FLOAT = ",maxpowerfloatResult)
print("MAX POWER DOUBLE = ",maxpowerdoubleResult)
print("############## MIN POWER  ######################")
print("MIN POWER FLOAT = ",minpowerfloatResult)
print("MIN POWER DOUBLE = ",minpowerdoubleResult)
print("############### COMPARE FLOAT####################")
compareFunc(comparefloatList)
print("############### COMPARE DOUBLE####################")
compareFunc(comparedoubleList)








