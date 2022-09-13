import math_utility
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import copy

""" 
    Operations and functions related to SE(2) poses 
    Poses are supposed expressed as q = [x, y, theta] 
"""

def fromPosePDF3DEuler(posePDF3D):
    """
        Extract the projected 2d pose (x,y,yaw) of a 3D pose

        Parameters
        ----------
        posePDF3D : dict
                    3D pose pdf in euler + 3d
        
        Returns
        -------
        dict 
            2D pose pdf
    """

    mean = np.empty((3,))
    mean[0:2] = posePDF3D["pose_mean"][3:5]
    mean[2] = posePDF3D["pose_mean"][0]
    cov = conditionalCovariance2D(posePDF3D["pose_cov"]) 
    return {"pose_mean": mean, "pose_cov": cov}

def conditionalCovariance2D(cov3D):
    """
        Get the covariance of a projected 2D pose from a 3D pose  
        ie. Cov(x,y,yaw | z, pitch, roll)

        Parameters
        ----------
        cov3D : 2D array
                3D pose covariance matrix in euler + 3d
        
        Returns
        -------
        2D array
                2D pose covariance matrix
    """

    count_nz = np.count_nonzero(cov3D)
    if(count_nz == 0):
        return np.zeros((3,3))
    else:
        # Submatrix related to x,y,yaw (S_11)
        cov_xyyaw = np.empty((3,3))
        cov_xyyaw[0:2,0:2] = cov3D[3:5,3:5]
        cov_xyyaw[2,2] = cov3D[0,0]
        cov_xyyaw[0,2] = cov3D[0,3]
        cov_xyyaw[2,0] = cov_xyyaw[0,2]
        cov_xyyaw[1,2] = cov3D[0,4]
        cov_xyyaw[2,1] = cov_xyyaw[1,2]

        # Submatrix related to z, pitch, roll (S_22)
        cov_zpitchroll = np.empty((3,3))
        cov_zpitchroll[1:3,1:3] = cov3D[1:3, 1:3]
        cov_zpitchroll[0,0] = cov3D[5,5]
        cov_zpitchroll[0,1] = cov3D[1,5]
        cov_zpitchroll[1,0] = cov_zpitchroll[0,1]
        cov_zpitchroll[0,2] = cov3D[2,5]
        cov_zpitchroll[2,0] = cov_zpitchroll[0,2]

        cov_zpitchroll_inv = np.linalg.inv(cov_zpitchroll)

        # Submatrix off diagonal (S_12)
        cov_offDiagBlock = np.empty((3,3))
        cov_offDiagBlock[0,0] = cov3D[3,5]
        cov_offDiagBlock[0,1] = cov3D[3,1]
        cov_offDiagBlock[0,2] = cov3D[3,2]
        cov_offDiagBlock[1,0] = cov3D[4,5]
        cov_offDiagBlock[1,1] = cov3D[4,1]
        cov_offDiagBlock[1,2] = cov3D[4,2]
        cov_offDiagBlock[2,0] = cov3D[0,5]
        cov_offDiagBlock[2,1] = cov3D[0,1]
        cov_offDiagBlock[2,2] = cov3D[0,2]

        # Covariance of the conditional distribution for a subset of variables from a multivariate normal 
        return cov_xyyaw - cov_offDiagBlock@cov_zpitchroll_inv@cov_offDiagBlock.T

def conditionalCovarianceXY(cov2D):
    """
        Get the covariance Cov(x,y) from the 2D pose ovariance Cov(x,y,yaw)

        Parameters
        ----------
        cov2D : 2D array
                2D pose covariance matrix (3 X 3)
        
        Results
        -------
        2D array
                (X,Y) 2 X 2 covariance matrix (no angle)
    """

    count_nz = np.count_nonzero(cov2D)
    if(count_nz==0):
        return np.zeros((2,2))
    else:
        cov_xy = np.empty((2,2))
        cov_xy[0:2,0:2] = cov2D[0:2,0:2]
        cov_offdiag = cov2D[0:2,2]
        inv_var_yaw = 1./cov2D[2,2]
        return cov_xy - inv_var_yaw*np.outer(cov_offdiag, cov_offdiag)

''' Convert a 2D pose to a 3D pose '''
def fromPosePDF2DTo3D(posePDF2D, posePDF3D):
    """
        Convert a 2D pose pdf to a 3D pose pdf (euler + 3D)

        Parameters
        ----------
        posePDF2D : dict
                    2D pose pdf
        posePDF3D : dict
                    3D pose pdf containing the (z,pitch,roll) values
        
        Returns
        -------
        dict
            3D pose pdf  
    """

    pose3D = copy.deepcopy(posePDF3D)
    pose3D["pose_mean"][0] = posePDF2D["pose_mean"][2]
    pose3D["pose_mean"][3] = posePDF2D["pose_mean"][0]
    pose3D["pose_mean"][4] = posePDF2D["pose_mean"][1]

    pose3D["pose_cov"][0, 0] = posePDF2D["pose_cov"][2, 2]
    pose3D["pose_cov"][3:5, 3:5] = posePDF2D["pose_cov"][0:2, 0:2]
    pose3D["pose_cov"][0, 3] = posePDF2D["pose_cov"][0, 2]
    pose3D["pose_cov"][3, 0] = pose3D["pose_cov"][0, 3]
    pose3D["pose_cov"][0, 4] = posePDF2D["pose_cov"][1, 2]
    pose3D["pose_cov"][4, 0] = pose3D["pose_cov"][0, 4]

    return pose3D

def rotationMatrix(q):
    """
        Get the rotation matrix of a 2D pose

        Parameters
        ----------
        q : array
           2D pose

        Returns
        -------
        2D array
                2 X 2 rotation matrix
    """

    cos = np.cos(q[2])
    sin = np.sin(q[2])
    return np.array([[cos, -sin],
                     [sin, cos]])
                 
def composePose(q1, q2):
    """
        Composition of 2D poses q1 + q2

        Parameters
        ----------
        q1: array
            first 2D pose
        q2 : array
            second 2D pose

        Returns
        -------
        array
             composed 2D pose
    """

    R2 = rotationMatrix(q2)
    t_comp = R2@q1[0:2] + q2[0:2]
    return np.array([t_comp[0], t_comp[1], q1[2] + q2[2]])

def composePosePoint(q, point):
    """
        Compose pose-point in 2D

        Parameters
        ----------
        q : array 
            2D pose
        point : array
                2D point

        Returns
        -------
        array
             composed 2D point 
    """

    R = rotationMatrix(q)
    return R@point + q[0:2]

def composePosePoint_array(q, point_array):
    """
        Composition pose-point q + point for each point in point_array

        Parameters
        ----------
        q : array
            2D pose
        point_array : 2D array
                      array of 2D points

        Returns
        -------
        2D array
                array of composed pose-point 
    """

    R = rotationMatrix(q)
    return np.einsum('ij,kj->ki', R,  point_array) + q[0:2]

def computeJacobian_composePose(q1, q2):
    """
        Jacobian of pose-pose composition in 2D

        Parameters
        ----------
        q1 : array
             first 2D pose
        q2 : array
             second 2D pose

        Returns 
        -------
        jacobian_q1 : 2D array
                      3 X 3 jacobian matrix wrt the first pose
        jacobian_q2 : 2D array
                      3 X 3 jacobian matrix wrt the second pose        
    """

    cos_2 = np.cos(q2[2])
    sin_2 = np.sin(q2[2])
    jacobian_q2 = np.array([[1., 0., -(sin_2*q1[0] +  cos_2*q1[1])],
                             [0., 1., cos_2*q1[0] - sin_2*q1[1]],
                             [0., 0., 1.]])
    jacobian_q1 = np.zeros((3, 3))
    jacobian_q1[0:2,0:2] = rotationMatrix(q2)
    jacobian_q1[2,2] = 1.

    return jacobian_q1, jacobian_q2

def composePosePDF(q1, q2):
    """
        Composition of 2D poses pdf

        Parameters
        ----------
        q1 : dict
             first 2D pose pdf
        
        q2 : dict
             second 2D pose pdf

        Returns
        -------
        dict
            composed 2D pose pdf
    """

    jacobian_q1, jacobian_q2 = composeJacobian_composePose(q1, q2)
    return {"pose_mean": composePose(q1["pose_mean"], q2["pose_mean"]), "pose_cov": jacobian_q1@q1["pose_cov"]@jacobian_q1.t +
             jacobian_q2@q2["pose_cov"]@jacobian_q2.t}

def jacobian_composePosePoint(q, point):
    """
        Jacobian of pose-point composition

        Parameters
        ----------
        q : array
            2D pose
        point : array
                2D point    

        Returns 
        -------
        jacobian_q : 2D array
                     2 X 3 jacobian matrix wrt pose
        jacobian_point : 2D array
                        2 X 2 jacobian matrix wrt point 
    """
    cos = np.cos(q[2])
    sin = np.sin(q[2])
    jacobian_q = np.array([[1., 0., -sin*point[0] - cos*point[1]],
                           [0., 1., cos*point[0] - sin*point[1]]])
    jacobian_point = rotationMatrix(q)

    return jacobian_q, jacobian_point

def jacobian_composePosePoint_array(q, point_array):
    """
        Jacobian of pose-point composition evaluated at an array of point

        Parameters
        ----------
        q : array
            2D pose
        point_array : 2D array
                      array of points

        Returns
        -------
        jacobian_q : 3D array
                     array of 2 X 3 jacobian matrices wrt pose
        jacobian_point  : 2D array
                          2 X 2 jacobian matrices wrt point
        
        Notes
        -----
        The jacobian wrt point is the rotation matrix of the pose q.
        It is therefore independant of the points. This is why only one matrix jacobian_point is returned.    
    """

    cos = np.cos(q[2])
    sin = np.sin(q[2])
    jacobian_q = np.empty((point_array.shape[0], 2, 3))
    jacobian_q[:,0:2,0:2] = np.eye(2)
    jacobian_q[:, 0, 2] = -sin*point_array[:,0] - cos*point_array[:,1]
    jacobian_q[:, 1, 2] = cos*point_array[:,0] - sin*point_array[:,1]

    jacobian_point = rotationMatrix(q)
    return jacobian_q, jacobian_point

def composePosePDFPoint(q, point):
    """
        Compose pose-point pdf in 2D

        Parameters
        ----------
        q : dict
            2D pose pdf
        point : dict
                2D point pdf

        Returns 
        -------
        dict
            composed 2D point pdf
    """

    jacobian_q, jacobian_point = jacobian_composePosePoint(q["pose_mean"], point["mean"])
    return {"mean": composePosePoint(q["pose_mean"], point["mean"]), "cov": jacobian_q@q["pose_cov"]@jacobian_q.T +
                                                                            jacobian_point@point["cov"]@jacobian_point.T}

def composePosePDFPoint_array(q, point_array):
    """
        Compose a pose with an array of 2D points pdf

        Parameters
        ----------
        q : dict
            2D pose pdf
        point_array : dict
                      Each attribute ("mean" and "cov") are arrays
        
        Returns
        -------
        dict
            results of the composition. Each attribute ("mean" and "cov") are arrays
    """
    
    jacobian_q, jacobian_point = jacobian_composePosePoint_array(q["pose_mean"], point_array["mean"])
    cov = np.einsum('kij,kjl->kil', jacobian_q, np.einsum('ij,klj->kil',q['pose_cov'],jacobian_q,optimize=True),optimize=True) + \
          np.einsum('ij,kjl->kil', jacobian_point,
                    np.einsum('kij,lj->kil', point_array['cov'], jacobian_point,
                              optimize=True), optimize=True)

    return {"mean" : composePosePoint_array(q["pose_mean"], point_array["mean"]), "cov" : cov}