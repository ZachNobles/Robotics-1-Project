import numpy as np
from scipy.spatial.transform import Rotation as R
from sys import exit

def fk_Dofbot (q):
    # FWDKIN_DOFBOT Computes the end effector position and orientation relative to the base frame for Yahboom's Dofbot manipulator 
    #     using the product of exponentials approach
    # Input :
    # q: 5x1 vector of joint angles in degrees
    #
    # Output :
    # Rot: The 3x3 rotation matrix describing the relative orientation of the end effector frame to the base frame (R_ {0T})
    # Pot: The 3x1 vector describing the position of the end effector relative to the base, 
    #      where the first element is the position along the base frame x-axis,
    #      the second element is the position along the base frame y-axis,
    #      and the third element is the position along the base frame z- axis (P_ {0T})

    #set up the basis unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # define the link lengths in meters
    l0 = 0.061 # base to servo 1
    l1 = 0.0435 # servo 1 to servo 2
    l2 = 0.08285 # servo 2 to servo 3
    l3 = 0.08285 # servo 3 to servo 4
    l4 = 0.07385 # servo 4 to servo 5
    l5 = 0.05457 # servo 5 to gripper

    #set up the rotation matrices between subsequent frames
    R01 = rotz(q[0]) # rotation between base frame and 1 frame
    R12 = roty(-q[1]) # rotation between 1 and 2 frames
    R23 = roty(-q[2]) # rotation between 2 and 3 frames
    R34 = roty(-q[3]) # rotation between 3 and 4 frames
    R45 = rotx(-q[4]) # rotation between 4 and 5 frames
    R5T = roty(0) #the tool frame is defined to be the same as frame 5

    #set up the position vectors between subsequent frames
    P01 = (l0+l1)*ez # translation between base frame and 1 frame in base frame
    P12 = np.zeros(3,) # translation between 1 and 2 frame in 1 frame
    P23 = l2*ex # translation between 2 and 3 frame in 2 frame
    P34 = -l3*ez # translation between 3 and 4 frame in 3 frame
    P45 = np.zeros(3,) # translation between 4 and 5 frame in 4 frame
    P5T = -(l4+l5)*ex # translation between 5 and tool frame in 5 frame

    # calculate Rot and Pot
    #Rot is a sequence of rotations
    Rot = R01*R12*R23*R34*R45*R5T
    #Pot is a combination of the position vectors. 
    #    Each vector must be represented in the base frame before addition. 
    #    This is achieved using the rotation matrices.
    Pot = P01 + R01.apply(P12 + R12.apply(P23 + R23.apply(P34 + R34.apply(P45+ R45.apply(P5T)))))

    return Rot, Pot






def rotx(theta):
    # return the principal axis rotation matrix for a rotation about the Xaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rx = R.from_euler('x',theta , degrees = True )
    return Rx

def roty(theta):
    # return the principal axis rotation matrix for a rotation about the Yaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Ry = R.from_euler('y',theta , degrees = True )
    return Ry

def rotz(theta):
    # return the principal axis rotation matrix for a rotation about the Zaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rz = R.from_euler('z', theta, degrees = True )
    return Rz


def validJointAngles(q):
    if len(q) < 5:
        return False
    for i in range(4): # range is 180 for joints 1-4
        if q[i] < 0 or q[i] > 180:
            return False
        
    if q[4] < 0 or q[4] > 270:
        return False
    
    return True



def runFK(q):

    if not validJointAngles(q):
        print("invalid joint angles input")
        return False

    R2,P2 = fk_Dofbot(q)
    print('Rotation matrix (R_ {0T}) for input: ')
    print(np.round(R2.as_matrix(), 2))
    print('Position vector (P_ {0T}) for input: ')
    print(np.round(P2, 2))
    return True