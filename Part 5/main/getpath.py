import numpy as np
from scipy.spatial.transform import Rotation as R

def rotx(theta):
    return R.from_euler('x', theta, degrees=True).as_matrix()

def roty(theta):
    return R.from_euler('y', theta, degrees=True).as_matrix()

def rotz(theta):
    return R.from_euler('z', theta, degrees=True).as_matrix()

def fwdkin_Dofbot(q):
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    l0 = 0.061  # base to servo 1
    l1 = 0.0435  # servo 1 to servo 2
    l2 = 0.08285  # servo 2 to servo 3
    l3 = 0.08285  # servo 3 to servo 4
    l4 = 0.07385  # servo 4 to servo 5
    l5 = 0.05457  # servo 5 to gripper

    R01 = rotz(q[0])  # rotation between base frame and 1 frame
    R12 = roty(-q[1])  # rotation between 1 and 2 frames
    R23 = roty(-q[2])  # rotation between 2 and 3 frames
    R34 = roty(-q[3])  # rotation between 3 and 4 frames
    R45 = rotx(-q[4])  # rotation between 4 and 5 frames
    R5T = roty(0)  # the tool frame is defined to be the same as frame 5

    # Set up the position vectors between subsequent frames
    P01 = (l0 + l1) * ez  # translation between base frame and 1 frame in base frame
    P12 = np.zeros(3)  # translation between 1 and 2 frame in 1 frame
    P23 = l2 * ex  # translation between 2 and 3 frame in 2 frame
    P34 = -l3 * ez  # translation between 3 and 4 frame in 3 frame
    P45 = np.zeros(3)  # translation between 4 and 5 frame in 4 frame
    P5T = -(l4 + l5) * ex  # translation between 5 and tool frame in 5 frame

    # Calculate Rot and Pot
    # Rot is a sequence of rotations
    Rot = R01 @ R12 @ R23 @ R34 @ R45 @ R5T  # Use @ for matrix multiplication
    # Pot is a combination of the position vectors.
    Pot = P01 + R01 @ (P12 + R12 @ (P23 + R23 @ (P34 + R34 @ (P45 + R45 @ P5T))))

    return Rot, Pot

def rotm2eul(R):
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix.")

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6  # Check for singularity
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def wrap_to_180(angle):
    return (angle + 180) % 360 - 180


def getPath(start, end):
    # Initial joint configuration in degrees
    qstart = np.array(start)
    # Final joint configuration in degrees
    qend = np.array(end)

    N = 250  # Number of sample points along the path
    lambda_vals = np.linspace(0, 1, N)  # Path variable from 0 to 1

    # Pre-allocate space for variables
    q = np.zeros((5, N))  # q(lambda)
    qset = list()
    Rot = np.zeros((3, 3, N))  # Rot(lambda)
    eulerot = np.zeros((3, N))  # Rot as Euler angles
    Pot = np.zeros((3, N))  # Pot(lambda)

    for ii in range(N):
        q[:, ii] = (1 - lambda_vals[ii]) * qstart + lambda_vals[ii] * qend  # Create q(lambda)
        Rot[:, :, ii], Pot[:, ii] = fwdkin_Dofbot(q[:, ii])
        eulerot[:, ii] = wrap_to_180(rotm2eul(Rot[:, :, ii]) * 180 / np.pi)  # Convert to degrees

    # Print the joint positions
    for i in range(N):
        qcurrent=[round(angle) for angle in q[:, i]]
        #print(f'λ={lambda_vals[i]:.3f}, q={qcurrent}')
        qset.append(qcurrent)


    # remove duplicates so the motion is smooth
    qset = [arr for i, arr in enumerate(qset) if arr not in qset[:i]]
    qset = [[int(round(x)) for x in s] for s in qset]
    return qset