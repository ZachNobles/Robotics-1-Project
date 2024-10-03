from ForwardKinematics import fk_Dofbot as fk
import numpy as np

#check if angles go past the limits
def validJointAngles(q):
    for i in range(4): # range is 180 for joints 1-4
        if q[i] < 0 or q[i] > 180:
            return False
        
    if q[5] < 0 or q[5] > 270:
        return False
    
    return True


# joint angle array
q = np.array([[910, 45, 135, 45, 135]]).T



if not validJointAngles(q):
    print("invalid joint angles input")
    exit(0)
R2,P2 = fk(q)
print('Rotation matrix (R_ {0T}) for test case 2: ')
print(R2. as_matrix ())
print('Position vector (P_ {0T}) for test case 2: ')
print(P2)