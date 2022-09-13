import math_utility
import poses_quat
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats

""" 
    Operations and functions related to SE(3) poses / poses pdf expressed as 6d vector euler + 3d
    
    Notes
    -----

    Implementation based on [1]. The last updated version of the paper can be found here :
    https://github.com/jlblancoc/tutorial-se3-manifold

    Pose pdf are represented are dict with keys "pose_mean" and "pose_cov".
    Point pdf are are represented are dict with keys "mean" and "cov".

    References
    ----------
    .. [1] "J.-L. Blanco. A tutorial on se (3) transformation parameterizations and
            on-manifold optimization. University of Malaga, Tech. Rep, 3, 2010." (Last update : 11/05/2020)
    .. [2] "Y. Breux, MpIC_oldVersion_noApprox" (in doc folder)
    .. [3] "Breux, Yohan, André Mas, and Lionel Lapierre.
            On-manifold Probabilistic ICP: Application to Underwater Karst Exploration." (2021)
"""

def composePoseEuler(q1, q2):
    """ Composition of two poses expressed in Euler + 3D (6D vector)

        Parameters
        ----------
        q1 : array
             first pose (Euler + 3D)
        q2 : array 
             second pose (Euler + 3D)

        Returns
        -------
        array
                Pose resulting from the composition 

        Notes
        -----
        See [1], section 5.1
    """

    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX',q1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX',q2[0:3])
    t1 = q1[3:6]
    t2 = q2[3:6]

    rot_12 = scipy.spatial.transform.Rotation.from_matrix(np.matmul(rot1.as_matrix(), rot2.as_matrix()))
    t_12 = rot1.apply(t2) + t1

    return np.block([rot_12.as_euler('ZYX'), t_12])


def composePoseEuler_array(q1, q2_array):
    """ 
        Composition of one pose with an array of poses in Euler + 3D (6D vector)
        
        Parameters
        ----------
        q1 : array
             first pose (Euler + 3D)
        q2_array : 2D array  
             array of second poses (Euler + 3D)

        Returns
        -------
        2D array
                Poses resulting from the composition of q1 with each pose in q2_array 

        Notes
        -----
        See [1], section 5.1
    """

    res = np.empty((q2_array.shape[0],6))
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX',q1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX',q2_array[:,0:3])
    t1 = q1[3:6]
    t2 = q2_array[:,3:6]

    rot_12 = scipy.spatial.transform.Rotation.from_matrix(np.einsum('ij,kjl->kil', rot1.as_matrix(), rot2.as_matrix())) #np.matmul(rot1.as_matrix(), rot2.as_matrix()))
    res[:,0:3] = scipy.spatial.transform.Rotation.from_matrix(np.einsum('ij,kjl->kil', rot1.as_matrix(), rot2.as_matrix())).as_euler('ZYX')
    res[:,3:6] = rot1.apply(t2) + t1

    return res

def composePoseEulerPoint(q_poseEuler, x):
    """ 
        Compose a pose (Euler + 3D) with a 3D point

        Parameters
        ----------
        q_poseEuler : array
                      pose (Euler + 3D)
        x :  array  
             3D point

        Returns
        -------
        array
              Resulting point from the pose-point composition

        Notes
        -----
        See [1], section 3.1
    """

    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3])
    t = q_poseEuler[3:6]
    x_composed = rot.apply(x) + t
    return x_composed

def inverseComposePoseEuler(q_poseEuler1, q_poseEuler2):
    """ 
        Inverse pose composition in Euler + 3D (q_poseEuler1 - q_poseEuler2)

        Parameters
        ----------
        q_poseEuler1 : array
                       first pose (Euler + 3D)
        q_poseEuler2 : array  
                       second pose (Euler + 3D)

        Returns
        -------
        array
              Result of the inverse composition, corresponding to 
              the relative coords of q_poseEuler1 wrt q_poseEuler2

        Notes
        -----
        See [1], section 5.1 / 6.1. Note that q1 - q2 = -q2 + q1
    """

    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler1[0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler2[0:3]).inv()
    t1 = q_poseEuler1[3:6]
    t2 = -rot2.apply(q_poseEuler2[3:6])

    rot_12 = scipy.spatial.transform.Rotation.from_matrix(np.matmul(rot1.as_matrix(), rot2.as_matrix()))
    t_12 = rot1.apply(t2) + t1

    return np.block([rot_12.as_euler('ZYX'), t_12])

def inverseComposePoseEuler_array(q_poseEuler1_array, q_poseEuler2):
    """ 
        Inverse pose composition of an array of poses with a pose (q_poseEuler1_array - q_poseEuler2)

        Parameters
        ----------
        q_poseEuler1_array : 2D array
                             first poses (Euler + 3D)
        q_poseEuler2 : array  
                       second pose (Euler + 3D)

        Returns
        -------
        2D array
              Result of the inverse composition, corresponding to 
              the relative coords of each pose in q_poseEuler1 wrt q_poseEuler2

        Notes
        -----
        See [1], section 5.1 / 6.1. Note that q1 - q2 = -q2 + q1
    """

    res = np.empty((q_poseEuler1_array.shape[0],6))
    rot1 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler1_array[:,0:3])
    rot2 = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler2[0:3]).inv()
    t1 = q_poseEuler1_array[:,3:6]
    t2 = -rot2.apply(q_poseEuler2[3:6])

    rot1_mat = rot1.as_matrix()
    res[:,0:3] = scipy.spatial.transform.Rotation.from_matrix(np.einsum('...ij,jk->...ik', rot1_mat, rot2.as_matrix())).as_euler('ZYX')
    res[:,3:6] = np.einsum('...ij,j->...i', rot1_mat, t2) + t1

    return res

def inverseComposePoseEulerPoint(q_poseEuler, x):
    """
        Inverse pose-point composition (x - q_poseEuler) in Euler + 3D

        Parameters
        ----------
        q_poseEuler : array
                      pose (Euler + 3D angles)
        x : array  
            3D point 

        Returns
        -------
        array
              Resulting 3D point, corresponding to the relative coords of x wrt q_poseEuler

        Notes
        -----
        See [1], section 4.1.
    """

    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3]).inv()
    t = q_poseEuler[3:6]
    x_composed = rot_T.apply(x - t)
    return x_composed

def inverseComposePoseEulerPoint_opt(q_poseEuler_array, x):
    """
        Inverse pose-point composition of a point x with each pose in q_poseEuler_arra (in Euler + 3D)

        Parameters
        ----------
        q_poseEuler_array : 2D array
                            array of poses (Euler + 3D)
        x : array  
            3D point 

        Returns
        -------
        2D array
              Resulting array 3D point, corresponding to the relative coords of x wrt poses in q_poseEuler_array

        Notes
        -----
        See [1], section 4.1.
    """
    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler_array[:,0:3]).inv()
    t = q_poseEuler_array[:,3:6]
    x_composed = rot_T.apply(x - t)
    return x_composed

def inverseComposePoseEulerPoint_opt_array(q_poseEuler_tensor, x_array):
    """
        Inverse pose-point composition of an array of points x_array 
        with a 2D array of poses in Euler + 3D angles. 

        Parameters
        ----------
        q_poseEuler_tensor : 3D array
                             Tensor of poses (samples X points X pose) (Euler + 3D)
        x_array : 2D array  
                  3D points 

        Returns
        -------
        3D array
              Resulting 3D points, corresponding to the relative coords of each x in x_array  
              wrt to each pose in q_poseEuler_tensor

        Notes
        -----
        Functions specifically implemented for mypicp algo.
        The 3D array input q_poseEuler_array has dimensions #samples X #points X 6D. 
        The samples correspond to the samples of poses in the robot trajectory from the CUT algorithm.
        The points correspond to the point clouds.

        See [1], section 4.1. and [2], section 5.3
    """

    # Reshape the n_samples X n_points X 6 3D array of q_poseEuler to n_samples * n_points X 6 2D array
    n_samples = q_poseEuler_tensor.shape[0]
    n_points = q_poseEuler_tensor.shape[1]
    q_posesEuler_reshaped = q_poseEuler_tensor.reshape((n_samples * n_points, 6), order='F')

    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posesEuler_reshaped[:, 0:3]).inv()
    t = q_poseEuler_tensor[:,:,3:6]

    # Dimension : n_samples * n_points X 3 X 1 
    diff = (x_array - t).reshape((n_samples*n_points,3,1),order='F')

    # Multiply slice by slice ie R_{i,j}(x_i - t_{i,j}) 
    # Reshape the results back to a 3D array with samples as the depth dimension 
    x_composed_array = np.einsum('...ij,...jh->...ih', rot_T.as_matrix(), diff,optimize=True).reshape((n_samples, n_points, 3),order='F')
    return x_composed_array

def inverseComposePoseEulerPoint_opt_array_association(q_poseEuler_tensor, x_array):
    """
        Inverse pose-point composition of an array of points x_array 
        with a 2D array of poses in Euler + 3D. 

        Parameters
        ----------
        q_poseEuler_tensor : array
                             Tensor of poses (samples X points X pose) (Euler + 3D)
        x : array  
            3D point 

        Returns
        -------
        array
              Resulting 3D point, corresponding to the relative coords of x wrt q_poseEuler

        Notes
        -----
        Functions specifically implemented for mypicp algo (for association step).
        The 3D array input q_poseEuler_array has dimensions #samples X #points X 6D. 
        The samples correspond to the samples of poses in the robot trajectory from the CUT algorithm.
        The points correspond to the point clouds.

        See [1], section 4.1. and [2], section 5.3
    """
    
    # Reshape the n_samples X n_points X 6 3D array of q_poseEuler to n_samples * n_points X 6 2D array 
    n_samples = q_poseEuler_tensor.shape[0]
    n_points_q = q_poseEuler_tensor.shape[1]
    n_points_x = x_array.shape[0]
    q_posesEuler_reshaped = q_poseEuler_tensor.reshape((n_samples * n_points_q, 6), order='C')

    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posesEuler_reshaped[:, 0:3]).inv()
    t = q_posesEuler_reshaped[:,3:6] 

    # Dimension of diff : n_samples X n_points_q X n_points_x (x_array) X 3 
    # We have diff[k][i][j] = x_array[j] - t[k][i] 
    diff = (x_array - t[:,None]).reshape((n_samples,n_points_q,n_points_x, 3))

    # Multiply slice by slice ie R_{i,k}(x_j - t_{i,k}) 
    x_composed_array = np.einsum('kilm,kijm->kijl', rot_T.as_matrix().reshape((n_samples,n_points_q,3,3)), diff,optimize=True)
    return x_composed_array

def fromPoseEulerToPoseQuat(q_poseEuler):
    """
        Convert a pose in Euler + 3D to a pose in quaternion + 3D

        Parameters
        ----------
        q_poseEuler : array
                      pose in Euler + 3D

        Returns
        -------
        array
             pose in quaternion + 3D

        Notes
        -----
        See [1], section 2.1
    """

    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3])
    quat = rot.as_quat()
    
    # numpy quaternion -> [qx qy qz qr] , mrpt quat is [qr qx qy qz]
    quat_arr = poses_quat.fromqxyzrToqrxyz(quat) 
    q_poseQuat = np.block([quat_arr, np.array(q_poseEuler[3:6])])
    return q_poseQuat

def fromPoseEulerToPoseQuat_array(q_poseEuler_array):
    """
        Convert an array of poses in Euler + 3D to an array of poses in quaternion + 3D 

        Parameters
        ----------
        q_poseEuler_array : 2D array
                            array of poses (euler + 3D)

        Returns
        -------
        2D array
                array of poses (quaternion + 3D)
        
        Notes
        -----
        See [1], section 2.1
    """

    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler_array[:,0:3])
    # numpy quaternion -> [qx qy qz qr] , mrpt quat is [qr qx qy qz] 
    q = rot.as_quat()

    q_poseQuat_array = np.empty((q_poseEuler_array.shape[0],7))
    q_poseQuat_array[:, 0] = q[:, 3]
    q_poseQuat_array[:, 1] = q[:, 0]
    q_poseQuat_array[:, 2] = q[:, 1]
    q_poseQuat_array[:, 3] = q[:, 2]
    q_poseQuat_array[:, 4:7] = q_poseEuler_array[:, 3:6]
    return q_poseQuat_array

''' Convert Pose in Quaternion+3D to Pose in Euler+3D '''
def fromPoseQuatToPoseEuler(q_poseQuat):
    """
        Convert a pose in quaternion + 3D to pose in euler + 3D

        Parameters
        ----------
        q_poseQuat : array
                     pose (quaternion + 3D)
        
        Returns
        -------
        array
             pose (euler + 3D)

        Notes
        -----
        See [1], section 2.2
    """

    quat = poses_quat.fromqrxyzToqxyzr(q_poseQuat) 
    
    # Quaternion is normalized here 
    quat[0:4] /= np.linalg.norm(quat[0:4])
    rot = scipy.spatial.transform.Rotation.from_quat(quat)
    q_poseEuler = np.block([np.array(rot.as_euler('ZYX')), np.array(q_poseQuat[4:7])])
    return q_poseEuler

def fromPoseQuatToPoseEuler_array(q_poseQuat_array):
    """
        Convert an array of poses in quaternion + 3D to an array of poses in euler + 3D

        Parameters
        ----------
        q_poseQuat_array : 2D array
                           array of poses (quaternion + 3D)
        
        Returns
        -------
        2D array
             array of poses (euler + 3D)

        Notes
        -----
        See [1], section 2.2
    """
    
    quat = poses_quat.fromqrxyzToqxyzr_array(q_poseQuat_array)
    
    # Quaternion is normalized here 
    rot = scipy.spatial.transform.Rotation.from_quat(quat)

    q_poseEuler_array = np.empty((q_poseQuat_array.shape[0], 6))
    q_poseEuler_array[:,0:3] = rot.as_euler('ZYX')
    q_poseEuler_array[:,3:6] = q_poseQuat_array[:,4:7]

    return q_poseEuler_array

def computeJacobian_quatToEuler(q_poseQuat):
    """
        Compute the jacobian of the quaternion + 3D pose to euler + 3D pose convertion.

        Parameters
        ----------
        q_poseQuat : array
                     quaternion + 3D pose at which the jacobian is computed 
        
        Results
        -------
        2D array
                 Returns a 6 X 7 matrix
        
        Notes
        -----
        See [1], section 2.2.2
    """

    # Compute the jacobian of euler angles wrt to quaternion
    determinant = q_poseQuat[0]*q_poseQuat[2] - q_poseQuat[1]*q_poseQuat[3]
    jacobian_euler_quat = np.zeros((3, 4))
    if(determinant > 0.49999):
        num = 2./(q_poseQuat[0]**2 + q_poseQuat[1]**2)
        jacobian_euler_quat[0][0] = num*q_poseQuat[1]
        jacobian_euler_quat[0][2] = -num*q_poseQuat[0]
    elif(determinant < -0.49999):
        num = 2. / (q_poseQuat[0] ** 2 + q_poseQuat[1] ** 2)
        jacobian_euler_quat[0][0] = -num * q_poseQuat[1]
        jacobian_euler_quat[0][2] = num * q_poseQuat[0]
    else:
        x_sqr = q_poseQuat[1]**2
        y_sqr = q_poseQuat[2]**2
        z_sqr = q_poseQuat[3]**2
        r_z = q_poseQuat[0]*q_poseQuat[3]
        r_x = q_poseQuat[0]*q_poseQuat[1]
        x_y = q_poseQuat[1]*q_poseQuat[2]
        y_z = q_poseQuat[2]*q_poseQuat[3]
        a = 1. - 2.*(y_sqr + z_sqr)
        a_sqr = a**2
        b = 2.*(r_z + x_y)
        c = 1. - 2.*(x_sqr + y_sqr)
        c_sqr = c**2
        d = 2.*(r_x + y_z)

        atan_prime_yaw = 1./(1. + (b/a)**2)
        atan_prime_roll = 1./(1. + (d/c)**2)
        asin_prime = 1./np.sqrt(1. - 4.*determinant**2)

        jacobian_euler_quat[0][0] = 2.*q_poseQuat[3]*atan_prime_yaw/a
        jacobian_euler_quat[0][1] = 2.*q_poseQuat[2]*atan_prime_yaw/a
        jacobian_euler_quat[0][2] = 2.*((q_poseQuat[1]*a + 2.*q_poseQuat[2]*b)/a_sqr)*atan_prime_yaw
        jacobian_euler_quat[0][3] = 2.*((q_poseQuat[0]*a + 2.*q_poseQuat[3]*b)/a_sqr)*atan_prime_yaw

        jacobian_euler_quat[1][0] = 2.*q_poseQuat[2]*asin_prime
        jacobian_euler_quat[1][1] = -2. * q_poseQuat[3] * asin_prime
        jacobian_euler_quat[1][2] = 2. * q_poseQuat[0] * asin_prime
        jacobian_euler_quat[1][3] = -2. * q_poseQuat[1] * asin_prime

        jacobian_euler_quat[2][0] = 2.*(q_poseQuat[1]/c)*atan_prime_roll
        jacobian_euler_quat[2][1] = 2. * ((q_poseQuat[0]*c + 2.*q_poseQuat[1]*d)/c_sqr) * atan_prime_roll
        jacobian_euler_quat[2][2] = 2. * ((q_poseQuat[3]*c + 2.*q_poseQuat[2]*d)/c_sqr) * atan_prime_roll
        jacobian_euler_quat[2][3] = 2. * (q_poseQuat[2] / c) * atan_prime_roll

        J_norm = poses_quat.jacobianQuatNormalization(q_poseQuat[0:4])


    jacobian = np.block([[jacobian_euler_quat@J_norm , np.zeros((3,3))],
                         [np.zeros((3,4)), np.eye(3)]
                        ])

    return jacobian

def computeJacobian_quatToEuler_array(q_poseQuat_array):
    """
        Compute the jacobian of the quaternion + 3D pose to euler + 3D pose convertion 
        at each pose in q_poseQuat_array

        Parameters
        ----------
        q_poseQuat_array : 2D array
                           Array of quaternion + 3D poses at which the jacobian is computed 
        
        Results
        -------
        3D array
                 Returns a q_poseQuat_array.shape[0] X 6 X 7 tensor
        
        Notes
        -----
        See [1], section 2.2.2
    """

    # Compute the jacobian of euler angles wrt to quaternion
    determinant = q_poseQuat_array[:,0]*q_poseQuat_array[:,2] - q_poseQuat_array[:,1]*q_poseQuat_array[:,3]
    jacobian_euler_quat = np.zeros((q_poseQuat_array.shape[0],3, 4))

    num = 2./(np.power(q_poseQuat_array[:,0],2) + np.power(q_poseQuat_array[:,1],2))
    num_poseQuat0 = num*q_poseQuat_array[:,0]
    num_poseQuat1 = num * q_poseQuat_array[:, 1]

    x_sqr = np.power(q_poseQuat_array[:,1],2)
    y_sqr = np.power(q_poseQuat_array[:,2],2)
    z_sqr = np.power(q_poseQuat_array[:,3],2)
    r_z = q_poseQuat_array[:,0] * q_poseQuat_array[:,3]
    r_x = q_poseQuat_array[:,0] * q_poseQuat_array[:,1]
    x_y = q_poseQuat_array[:,1] * q_poseQuat_array[:,2]
    y_z = q_poseQuat_array[:,2] * q_poseQuat_array[:,3]
    a = 1. - 2. * (y_sqr + z_sqr)
    a_inv = 1./a
    a_sqr = np.power(a,2)
    a_sqr_inv = 1./a_sqr
    b = 2. * (r_z + x_y)
    c = 1. - 2. * (x_sqr + y_sqr)
    c_inv = 1./c
    c_sqr_inv = 1./np.power(c,2)
    d = 2. * (r_x + y_z)

    atan_prime_yaw = 1. / (1. + np.power(b*a_inv,2))
    atan_prime_roll = 1. / (1. + np.power(d*c_inv,2))
    asin_prime = 1. / np.sqrt(1. - 4. * np.power(determinant,2))

    a00 = 2.*q_poseQuat_array[:,3]*atan_prime_yaw*a_inv
    a01 = 2.*q_poseQuat_array[:,2]*atan_prime_yaw*a_inv
    a02 = 2.*((q_poseQuat_array[:,1]*a + 2.*q_poseQuat_array[:,2]*b)*a_sqr_inv)*atan_prime_yaw
    a03 = 2. * ((q_poseQuat_array[:,0] * a + 2. * q_poseQuat_array[:,3] * b)* a_sqr_inv) * atan_prime_yaw
    a10 = 2.*q_poseQuat_array[:,2]*asin_prime
    a11 = -2. * q_poseQuat_array[:,3] * asin_prime
    a12 = 2. * q_poseQuat_array[:,0] * asin_prime
    a13 = -2. * q_poseQuat_array[:,1] * asin_prime
    a20 = 2.*(q_poseQuat_array[:,1]*c_inv)*atan_prime_roll
    a21 = 2. * ((q_poseQuat_array[:,0]*c + 2.*q_poseQuat_array[:,1]*d)*c_sqr_inv) * atan_prime_roll
    a22 = 2. * ((q_poseQuat_array[:,3]*c + 2.*q_poseQuat_array[:,2]*d)*c_sqr_inv) * atan_prime_roll
    a23 = 2. * (q_poseQuat_array[:,2]*c_inv) * atan_prime_roll
    for k in range(0,determinant.shape[0]):
        det = determinant[k]
        if(det> 0.49999):
            jacobian_euler_quat[k,0,0] = num_poseQuat1[k]
            jacobian_euler_quat[k,0,2] = -num_poseQuat0[k]
        elif(det< -0.49999):
            jacobian_euler_quat[k,0,0] = -num_poseQuat1[k]
            jacobian_euler_quat[k,0,2] = num_poseQuat0[k]
        else:
            jacobian_euler_quat[k,0,0] = a00[k]
            jacobian_euler_quat[k,0,1] = a01[k]
            jacobian_euler_quat[k,0,2] = a02[k]
            jacobian_euler_quat[k,0,3] = a03[k]

            jacobian_euler_quat[k,1,0] = a10[k]
            jacobian_euler_quat[k,1,1] = a11[k]
            jacobian_euler_quat[k,1,2] = a12[k]
            jacobian_euler_quat[k,1,3] = a13[k]

            jacobian_euler_quat[k,2,0] = a20[k]
            jacobian_euler_quat[k,2,1] = a21[k]
            jacobian_euler_quat[k,2,2] = a22[k]
            jacobian_euler_quat[k,2,3] = a23[k]

    J_norm = poses_quat.jacobianQuatNormalization_array(q_poseQuat_array[:,0:4])

    jacobian = np.zeros((q_poseQuat_array.shape[0],6,7))
    jacobian[:,0:3,0:4] = np.einsum('kij,kjl->kil',jacobian_euler_quat,J_norm,optimize=True)
    ones = np.full(q_poseQuat_array.shape[0],1.)
    jacobian[:,3,4] = ones
    jacobian[:,4,5] = ones
    jacobian[:,5,6] = ones

    return jacobian

def computeJacobian_eulerToQuat(q_poseEuler):
    """
        Compute the jacobian of the euler + 3D pose to quaternion + 3D pose convertion.

        Parameters
        ----------
        q_poseEuler : array
                     euler + 3D pose at which the jacobian is computed 
        
        Results
        -------
        2D array
                 Returns a 7 X 6 matrix
        
        Notes
        -----
        See [1], section 2.1.2
    """

    half_yaw = 0.5*q_poseEuler[0]
    half_pitch = 0.5*q_poseEuler[1]
    half_roll = 0.5*q_poseEuler[2]
    cos_yaw = np.cos(half_yaw)
    sin_yaw = np.sin(half_yaw)
    cos_pitch = np.cos(half_pitch)
    sin_pitch = np.sin(half_pitch)
    cos_roll = np.cos(half_roll)
    sin_roll = np.sin(half_roll)
    ccc = cos_roll*cos_pitch*cos_yaw
    ccs = cos_roll*cos_pitch*sin_yaw
    csc = cos_roll*sin_pitch*cos_yaw
    css = cos_roll*sin_pitch*sin_yaw
    scs = sin_roll*cos_pitch*sin_yaw
    scc = sin_roll*cos_pitch*cos_yaw
    ssc = sin_roll*sin_pitch*cos_yaw
    sss = sin_roll*sin_pitch*sin_yaw

    jacobian_quat_euler = 0.5*np.array([[ssc - ccs, scs - csc, css - scc],
                               [-(csc + scs), -(ssc + ccs), ccc + sss],
                               [scc - css, ccc - sss, ccs - ssc],
                               [ccc + sss, -(css + scc), -(csc + scs)]])


    jacobian = np.block([[jacobian_quat_euler, np.zeros((4,3))],
                         [np.zeros((3,3)), np.eye(3)]
                         ])

    return jacobian

def computeJacobian_eulerToQuat_array(q_poseEuler_array):
    """
        Compute the jacobian of the euler + 3D pose to quaternion + 3D pose convertion 
        at each pose in q_poseEuler_array.

        Parameters
        ----------
        q_poseEuler_array : 2D array
                            euler + 3D array of poses at which the jacobian is computed 
        
        Results
        -------
        3D array
                 Returns a q_poseEuler_array.shape[0] X 7 X 6 tensor
        
        Notes
        -----
        See [1], section 2.1.2
    """

    half_yaw = 0.5*q_poseEuler_array[:,0]
    half_pitch = 0.5*q_poseEuler_array[:,1]
    half_roll = 0.5*q_poseEuler_array[:,2]
    cos_yaw = np.cos(half_yaw)
    sin_yaw = np.sin(half_yaw)
    cos_pitch = np.cos(half_pitch)
    sin_pitch = np.sin(half_pitch)
    cos_roll = np.cos(half_roll)
    sin_roll = np.sin(half_roll)
    ccc = cos_roll*cos_pitch*cos_yaw
    ccs = cos_roll*cos_pitch*sin_yaw
    csc = cos_roll*sin_pitch*cos_yaw
    css = cos_roll*sin_pitch*sin_yaw
    scs = sin_roll*cos_pitch*sin_yaw
    scc = sin_roll*cos_pitch*cos_yaw
    ssc = sin_roll*sin_pitch*cos_yaw
    sss = sin_roll*sin_pitch*sin_yaw

    a11 = ssc - ccs
    a12 = scs - csc
    a13 = css - scc
    a21 = -(csc + scs)
    a22 = -(ssc + ccs)
    a23 = ccc + sss
    a32 = ccc - sss
    a42 = -(css + scc)

    jacobian_array = np.zeros((q_poseEuler_array.shape[0],7,6))
    jacobian_array[:, 0, 0] = 0.5*a11
    jacobian_array[:, 0, 1] = 0.5*a12
    jacobian_array[:, 0, 2] = 0.5*a13
    jacobian_array[:, 1, 0] = 0.5*a21
    jacobian_array[:, 1, 1] = 0.5*a22
    jacobian_array[:, 1, 2] = 0.5*a23
    jacobian_array[:, 2, 0] = -0.5*a13
    jacobian_array[:, 2, 1] = 0.5*a32
    jacobian_array[:, 2, 2] = -0.5*a11
    jacobian_array[:, 3, 0] = 0.5*a23
    jacobian_array[:, 3, 1] = 0.5*a42
    jacobian_array[:, 3, 2] = 0.5*a21
    ones = np.full(q_poseEuler_array.shape[0], 1.)
    jacobian_array[:, 4, 3] = ones
    jacobian_array[:, 5, 4] = ones
    jacobian_array[:, 6, 5] = ones

    return jacobian_array

def computeJacobianEuler_composePose(q_poseEuler1, q_poseEuler2, q_poseEuler_compose_mean):
    """
        Compute the jacobian of euler + 3D pose composition

        Parameters
        ----------
        q_poseEuler1 : array
                       first euler + 3D pose at which the jacobian is computed
        q_poseEuler2 : array
                       second euler + 3D pose at which the jacobian is computed
        q_poseEuler_compose_mean : array
                                   pre-computed composition q_poseEuler1 + q_poseEuler2

        Returns
        -------
        jacobian_q1 : 2D array
                      Jacobian relative to the first pose (6 X 6 matrix) 
        jacobian_q2 : 2D array
                      Jacobian relative to the second pose (6 X 6 matrix)

        Notes
        -----
        See [1], section 5.1.2
    """

    q_poseQuat1 = fromPoseEulerToPoseQuat(q_poseEuler1)
    q_poseQuat2 = fromPoseEulerToPoseQuat(q_poseEuler2)
    q_poseQuat_compose = fromPoseEulerToPoseQuat(q_poseEuler_compose_mean)

    jacobian_quat_q1, jacobian_quat_q2 = poses_quat.computeJacobianQuat_composePose(q_poseQuat1, q_poseQuat2)

    jacobian_quatToEuler_q_compose = computeJacobian_quatToEuler(q_poseQuat_compose)
    jacobian_eulerToQuat_q1 = computeJacobian_eulerToQuat(q_poseEuler1)
    jacobian_eulerToQuat_q2 = computeJacobian_eulerToQuat(q_poseEuler2)

    jacobian_q1 = np.matmul(jacobian_quatToEuler_q_compose, np.matmul(jacobian_quat_q1, jacobian_eulerToQuat_q1))
    jacobian_q2 = np.matmul(jacobian_quatToEuler_q_compose, np.matmul(jacobian_quat_q2, jacobian_eulerToQuat_q2))

    return jacobian_q1, jacobian_q2

def computeJacobianEuler_composePose_array(q_poseEuler1, q_poseEuler2_array, q_poseEuler_compose_mean_array):
    """
        Compute the jacobian of euler + 3D pose composition 
        at pose q_poseEuler1 and at each pose in q_poseEuler2_array

        Parameters
        ----------
        q_poseEuler1 : array
                       first euler + 3D pose at which the jacobian is computed
        q_poseEuler2_array : 2D array
                             array of second euler + 3D poses at which the jacobian is computed
        q_poseEuler_compose_mean_array : 2D array
                                         pre-computed composition of q_poseEuler1 + q_poseEuler2
                                         for each q_poseEuler2 in q_poseEuler2_array

        Returns
        -------
        jacobian_q1 : 3D array
                      Jacobian relative to the first pose (q_poseEuler2_array.shape[0] X 6 X 6 tensor) 
        jacobian_q2 : 3D array
                      Jacobian relative to the second pose (q_poseEuler2_array.shape[0] X 6 X 6 tensor)

        Notes
        -----
        See [1], section 5.1.2
    """

    q_poseQuat1 = fromPoseEulerToPoseQuat(q_poseEuler1)
    q_poseQuat2_array = fromPoseEulerToPoseQuat_array(q_poseEuler2_array)
    q_poseQuat_compose_array = fromPoseEulerToPoseQuat_array(q_poseEuler_compose_mean_array)

    jacobian_quat_q1, jacobian_quat_q2 = poses_quat.computeJacobianQuat_composePose_array(q_poseQuat1, q_poseQuat2_array)

    jacobian_quatToEuler_q_compose = computeJacobian_quatToEuler_array(q_poseQuat_compose_array)
    jacobian_eulerToQuat_q1 = computeJacobian_eulerToQuat(q_poseEuler1)
    jacobian_eulerToQuat_q2 = computeJacobian_eulerToQuat_array(q_poseEuler2_array)

    return np.einsum('kij,kjl->kil',jacobian_quatToEuler_q_compose, np.einsum('kij,jl->kil',jacobian_quat_q1, jacobian_eulerToQuat_q1)), \
           np.einsum('kij,kjl->kil',jacobian_quatToEuler_q_compose, np.einsum('kij,kjl->kil',jacobian_quat_q2, jacobian_eulerToQuat_q2))

def computeJacobianEuler_composePosePDFPoint_pose(q_mean, point_mean):
    """
        Compute jacobian of the pose-point composition relative to the pose (in euler + 3D)

        Parameters
        ----------
        q_mean : array 
                 euler + 3D pose at which the jacobian is computed
        point_mean : array
                     3D point at which the jacobian is computed

        Returns
        -------
        2D array
                 jacobian relative to the pose (3 X 6 matrix) 

        Notes
        -----
        See [1], section 3.1.2           
    """

    cos_yaw = np.cos(q_mean[0])
    sin_yaw = np.sin(q_mean[0])
    cos_pitch = np.cos(q_mean[1])
    sin_pitch = np.sin(q_mean[1])
    cos_roll = np.cos(q_mean[2])
    sin_roll = np.sin(q_mean[2])

    a11 = -point_mean[0]*sin_yaw*cos_pitch - point_mean[1]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll) + point_mean[2]*(-sin_yaw*sin_pitch*cos_roll + cos_yaw*sin_roll)
    a12 = -point_mean[0]*cos_yaw*sin_pitch + point_mean[1]*cos_yaw*cos_pitch*sin_roll + point_mean[2]*cos_yaw*cos_pitch*cos_roll
    a13 = point_mean[1]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll) + point_mean[2]*(-cos_yaw*sin_pitch*sin_roll + sin_yaw*cos_roll)
    a21 = point_mean[0]*cos_yaw*cos_pitch + point_mean[1]*(cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll) + point_mean[2]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll)
    a22 = -point_mean[0]*sin_yaw*sin_pitch + point_mean[1]*sin_yaw*cos_pitch*sin_roll + point_mean[2]*sin_yaw*cos_pitch*cos_roll
    a23 = point_mean[1]*(sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll) - point_mean[2]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll)
    a31 = 0.
    a32 = -(point_mean[0]*cos_pitch + point_mean[1]*sin_pitch*sin_roll + point_mean[2]*sin_pitch*cos_roll)
    a33 = point_mean[1]*cos_pitch*cos_roll - point_mean[2]*cos_pitch*sin_roll

    return np.block([np.array([[a11,a12,a13],
                     [a21,a22,a23],
                     [a31,a32,a33]]),np.eye(3)])

def computeJacobianEuler_composePosePDFPoint_pose_array(q_mean, point_mean_array):
    """
        Compute jacobian of the pose-point composition relative to the pose (in euler + 3D)
        evaluated at each point in point_mean_array

        Parameters
        ----------
        q_mean : array 
                 euler + 3D pose at which the jacobian is computed
        point_mean_array : 2D array
                           array of 3D points at which the jacobian is computed

        Returns
        -------
        3D array
                 jacobians relative to the pose (point_mean_array.shape[0] X 3 X 6 tensor) 

        Notes
        -----
        See [1], section 3.1.2           
    """

    cos_yaw = np.cos(q_mean[0])
    sin_yaw = np.sin(q_mean[0])
    cos_pitch = np.cos(q_mean[1])
    sin_pitch = np.sin(q_mean[1])
    cos_roll = np.cos(q_mean[2])
    sin_roll = np.sin(q_mean[2])

    jacobian = np.zeros((point_mean_array.shape[0], 3, 6))

    jacobian[:,0,0] = -point_mean_array[:,0]*sin_yaw*cos_pitch - point_mean_array[:,1]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll) + point_mean_array[:,2]*(-sin_yaw*sin_pitch*cos_roll + cos_yaw*sin_roll)
    jacobian[:,0,1] = -point_mean_array[:,0]*cos_yaw*sin_pitch + point_mean_array[:,1]*cos_yaw*cos_pitch*sin_roll + point_mean_array[:,2]*cos_yaw*cos_pitch*cos_roll
    jacobian[:,0,2] = point_mean_array[:,1]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll) + point_mean_array[:,2]*(-cos_yaw*sin_pitch*sin_roll + sin_yaw*cos_roll)
    jacobian[:,1,0] = point_mean_array[:,0]*cos_yaw*cos_pitch + point_mean_array[:,1]*(cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll) + point_mean_array[:,2]*(cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll)
    jacobian[:,1,1] = -point_mean_array[:,0]*sin_yaw*sin_pitch + point_mean_array[:,1]*sin_yaw*cos_pitch*sin_roll + point_mean_array[:,2]*sin_yaw*cos_pitch*cos_roll
    jacobian[:,1,2] = point_mean_array[:,1]*(sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll) - point_mean_array[:,2]*(sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll)
    jacobian[:,2,0] = 0.
    jacobian[:,2,1] = -(point_mean_array[:,0]*cos_pitch + point_mean_array[:,1]*sin_pitch*sin_roll + point_mean_array[:,2]*sin_pitch*cos_roll)
    jacobian[:,2,2] = point_mean_array[:,1]*cos_pitch*cos_roll - point_mean_array[:,2]*cos_pitch*sin_roll
    ones = np.full(point_mean_array.shape[0],1.)
    jacobian[:,0,3] = ones
    jacobian[:,1,4] = ones
    jacobian[:,2,5] = ones

    return jacobian

def composePosePDFEuler(q_posePDFEuler1, q_posePDFEuler2):
    """
        Compute pose pdf (mean + covariance) composition in euler + 3D 

        Parameters
        ----------
        q_posePDFEuler1 : dict
                          first pose pdf
        q_posePDFEuler1 : dict
                          second pose pdf
        
        Results
        -------
        dict
            pose pdf of q_posePDFEuler1 + q_posePDFEuler2

        Notes
        -----
        See [1], section 5.1
    """

    q_poseEuler_compose_mean = composePoseEuler(q_posePDFEuler1['pose_mean'], q_posePDFEuler2['pose_mean'])

    jacobian_q1, jacobian_q2 = computeJacobianEuler_composePose(q_posePDFEuler1['pose_mean'], q_posePDFEuler2['pose_mean'], q_poseEuler_compose_mean)

    q_poseEuler_compose_cov = np.matmul(jacobian_q1, np.matmul(q_posePDFEuler1['pose_cov'], np.transpose(jacobian_q1))) +\
          np.matmul(jacobian_q2, np.matmul(q_posePDFEuler2['pose_cov'], np.transpose(jacobian_q2)))

    q_posePDFEuler_compose = {'pose_mean' : q_poseEuler_compose_mean, 'pose_cov' : q_poseEuler_compose_cov}
    return q_posePDFEuler_compose

def composePosePDFEuler_array(q_posePDFEuler1, q_posePDFEuler2_array):
    """
        Compute pose pdf (mean + covariance) composition in euler + 3D for each
        pose in q_posePDFEuler2_array

        Parameters
        ----------
        q_posePDFEuler1 : dict
                          first pose pdf
        q_posePDFEuler2_array : array of dict
                                second poses pdf
        
        Returns
        -------
        array of dict
                      poses pdf of q_posePDFEuler1 + q2 for each q2 in q_posePDFEuler2_array

        Notes
        -----
        See [1], section 5.1
    """

    q_poseEuler_compose_mean = composePoseEuler_array(q_posePDFEuler1['pose_mean'], q_posePDFEuler2_array['pose_mean'])

    jacobian_q1, jacobian_q2 = computeJacobianEuler_composePose_array(q_posePDFEuler1['pose_mean'], q_posePDFEuler2_array['pose_mean'], q_poseEuler_compose_mean)

    q_poseEuler_compose_cov =  np.einsum('kij,kjl->kil', jacobian_q1, np.einsum('ij,klj->kil',q_posePDFEuler1['pose_cov'],jacobian_q1,optimize=True),optimize=True) + \
                               np.einsum('kij,kjl->kil', jacobian_q2,
                                         np.einsum('kij,klj->kil', q_posePDFEuler2_array['pose_cov'], jacobian_q2,
                                                   optimize=True), optimize=True)

    q_posePDFEuler_compose = {'pose_mean' : q_poseEuler_compose_mean, 'pose_cov' : q_poseEuler_compose_cov}
    return q_posePDFEuler_compose

def composePosePDFEulerPoint(q_posePDFEuler, point):
    """
        Compute composition of pose pdf - point pdf 

        Parameters
        ----------
        q_posePDFEuler : dict
                         pose pdf in euler + 3D

        point : dict
                3D point pdf 

        Returns
        -------
        dict
            the point pdf q_posePDFEuler + point

        Notes
        -----
        See [1], section 3.1.1
    """

    jacobian_pose = computeJacobianEuler_composePosePDFPoint_pose(q_posePDFEuler['pose_mean'], point['mean'])
    jacobian_point = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posePDFEuler['pose_mean'][0:3]).as_matrix()

    cov = jacobian_pose@q_posePDFEuler['pose_cov']@jacobian_pose.T + jacobian_point@point['cov']@jacobian_point.T

    mean = composePoseEulerPoint(q_posePDFEuler['pose_mean'], point['mean'])
    return {'mean': mean, 'cov': cov}

def composePosePDFEulerPoint_array(q_posePDFEuler, point_array):
    """
        Compute composition of pose pdf - point pdf for each point in point_array

        Parameters
        ----------
        q_posePDFEuler : dict
                         pose pdf in euler + 3D
        point_array: array of dict
                     array of 3D point pdf 

        Returns
        -------
        array of dict
                      the points pdf q_posePDFEuler + point for each point in point_array 

        Notes
        -----
        See [1], section 3.1.1
    """

    jacobian_pose_array = computeJacobianEuler_composePosePDFPoint_pose_array(q_posePDFEuler['pose_mean'], point_array['mean'])
    jacobian_point = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posePDFEuler['pose_mean'][0:3]).as_matrix()

    cov = np.einsum('kij,kjl->kil', jacobian_pose_array, np.einsum('ij,klj->kil',q_posePDFEuler['pose_cov'],jacobian_pose_array,optimize=True),optimize=True) + \
          np.einsum('ij,kjl->kil', jacobian_point,
                    np.einsum('kij,lj->kil', point_array['cov'], jacobian_point,
                              optimize=True), optimize=True)

    mean = composePoseEulerPoint(q_posePDFEuler['pose_mean'], point_array['mean'])
    return {'mean': mean, 'cov': cov}

def inversePosePDFEuler(q_posePDFEuler):
    """
        Compute the inverse pose pdf in euler + 3D

        Parameters
        ----------
        q_posePDFEuler : dict
                         pose pdf
        
        Returns
        -------
        dict
            inverse pose pdf
        
        Notes
        -----
        See [1], section 6.1
    """

    q_posePDFQuat = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler)
    q_posePDFQuat_inv = poses_quat.inversePosePDFQuat(q_posePDFQuat)
    return fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_inv)

def inverseComposePosePDFEuler(q_posePDFEuler1, q_posePDFEuler2):
    """
        Compute the inverse pose pdf composition in euler + 3D 

        Parameters
        ----------
        q_posePDFEuler1 : dict
                          first pose pdf
        q_posePDFEuler2 : dict
                          second pose pdf
        
        Returns
        -------
        dict
            inverse pose pdf composition q_posePDFEuler1 - q_posePDFEuler2
        
        Notes
        -----
        See [1], section 3.1.1
    """

    q_posePDFQuat1 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler1)
    q_posePDFQuat2 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler2)
    q_posePDFQuat_composed = poses_quat.inverseComposePosePDFQuat(q_posePDFQuat1, q_posePDFQuat2)
    return fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_composed)

def inverseComposePosePDFEuler_array(q_posePDFEuler1_array, q_posePDFEuler2):
    """
        Compute the inverse pose pdf composition in euler + 3D  for each pose in q_posePDFEuler1_array

        Parameters
        ----------
        q_posePDFEuler1_array : array of dict
                                array of first poses pdf
        q_posePDFEuler2 : dict
                          second pose pdf
        
        Returns
        -------
        array of dict
                      inverse poses pdf composition q1 - q_posePDFEuler2 for each q1 in q_posePDFEuler1_array
        
        Notes
        -----
        See [1], section 3.1.1
    """

    q_posePDFQuat1_array = fromPosePDFEulerToPosePDFQuat_array(q_posePDFEuler1_array)
    q_posePDFQuat2 = fromPosePDFEulerToPosePDFQuat(q_posePDFEuler2)
    q_posePDFQuat_composed_array = poses_quat.inverseComposePosePDFQuat_array(q_posePDFQuat1_array, q_posePDFQuat2)
    return fromPosePDFQuatToPosePDFEuler_array(q_posePDFQuat_composed_array)

def inverseComposePosePDFEulerPoint_array(q_posePDFEuler, x_array):
    """
        Compute the inverse pose pdf- point pdf composition in euler + 3D  for each point in x_array

        Parameters
        ----------
        q_posePDFEuler : dict
                         first pose pdf
        x_array : array of dict
                  array of 3D points pdf
        
        Returns
        -------
        array of dict
                      inverse pose pdf - point pdf composition x - q_posePDFEuler for each x in x_array
        
        Notes
        -----
        See [1], section 4.1.1
    """

    q_pdf_inv = inversePosePDFEuler(q_posePDFEuler)
    return composePosePDFEulerPoint_array(q_pdf_inv, x_array)

def fromPosePDFEulerToPosePDFQuat(q_posePDFEuler):
    """
        Convert euler + 3D pose pdf to quaternion + 3D pose pdf

        Parameters
        ----------
        q_posePDFEuler : dict
                         pose pdf in euler + 3D
        
        Returns
        -------
        dict 
            pose pdf in quaternion + 3D
        
        Notes
        -----
        See [1], section 2.1 
    """

    q_posePDFQuat_mean = fromPoseEulerToPoseQuat(q_posePDFEuler['pose_mean'])
    J = computeJacobian_eulerToQuat(q_posePDFEuler['pose_mean'])
    q_posePDFQuat_cov = np.matmul(J, np.matmul(q_posePDFEuler['pose_cov'], np.transpose(J)))
    return {'pose_mean' : q_posePDFQuat_mean, 'pose_cov': q_posePDFQuat_cov}

def fromPosePDFEulerToPosePDFQuat_array(q_posePDFEuler_array):
    """
        Convert euler + 3D poses pdf array to quaternion + 3D poses pdf array

        Parameters
        ----------
        q_posePDFEuler_array : array of dict
                               array of poses pdf in euler + 3D
        
        Returns
        -------
        array of dict 
                      array of poses pdf in quaternion + 3D
        
        Notes
        -----
        See [1], section 2.1 
    """

    q_posePDFQuat_mean_array = fromPoseEulerToPoseQuat_array(q_posePDFEuler_array['pose_mean'])
    J = computeJacobian_eulerToQuat_array(q_posePDFEuler_array['pose_mean'])
    q_posePDFQuat_cov_array = np.einsum('kij,kjl->kil', J, np.einsum('kij,klj->kil',q_posePDFEuler_array['pose_cov'],J,optimize=True),optimize=True) 
    return {'pose_mean' : q_posePDFQuat_mean_array, 'pose_cov': q_posePDFQuat_cov_array}

def fromPosePDFQuatToPosePDFEuler(q_posePDFQuat):
    """
        Convert quaternion + 3D pose pdf to euler + 3D pose pdf

        Parameters
        ----------
        q_posePDFQuat : dict
                        pose pdf in quaternion + 3D
        
        Returns
        -------
        dict 
            pose pdf in euler + 3D
        
        Notes
        -----
        See [1], section 2.2 
    """

    q_posePDFEuler_mean = fromPoseQuatToPoseEuler(q_posePDFQuat['pose_mean'])
    J = computeJacobian_quatToEuler(q_posePDFQuat['pose_mean'])
    q_posePDFEuler_cov = np.matmul(J, np.matmul(q_posePDFQuat['pose_cov'], np.transpose(J)))
    return {'pose_mean':q_posePDFEuler_mean, 'pose_cov':q_posePDFEuler_cov}

def fromPosePDFQuatToPosePDFEuler_array(q_posePDFQuat_array):
    """
        Convert quaternion + 3D poses pdf array to euler + 3D poses pdf array

        Parameters
        ----------
        q_posePDFQuat_array : array of dict
                              array of poses pdf in quaternion + 3D
        
        Returns
        -------
        array of dict 
                      array of poses pdf in euler + 3D
        
        Notes
        -----
        See [1], section 2.2
    """

    q_posePDFEuler_mean = fromPoseQuatToPoseEuler_array(q_posePDFQuat_array['pose_mean'])
    J = computeJacobian_quatToEuler_array(q_posePDFQuat_array['pose_mean'])
    q_posePDFEuler_cov = np.einsum('kij,kjl->kil', J, np.einsum('kij,klj->kil',q_posePDFQuat_array['pose_cov'],J,optimize=True),optimize=True)
    return {'pose_mean':q_posePDFEuler_mean, 'pose_cov':q_posePDFEuler_cov}

def distanceSE3(p1, p2):
    """
        Compute the distance (relative to the metric in SE(3)) between two poses euler + 3D

        Parameters
        ----------
        p1 : array
             first pose in euler + 3D

        p2 : array
             second pose in euler + 3D

        Returns
        -------
        float
             Distance between p1 and p2 in SE(3)

        Notes
        -----
        See [3], section 3.1 equation (6) / equation (104) for an example of usage
    """
    
    return math_utility.distanceSE3(p1, p2)