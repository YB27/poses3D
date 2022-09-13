import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import time
import poses_euler
import multiprocessing as mp

""" 
    Operations and functions related to SE(3) poses

    References
    ----------
    .. [1] Yohan Breux and André Mas and Lionel Lapierre, "On-manifold Probabilistic ICP : Application to Underwater Karst
    Exploration"
    .. [2] Blanco JL (2010) A tutorial on se (3) transformation parameterizations and on-manifold optimization. University of Malaga,
           Tech. Rep 3.
    .. [3] METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS, K. Madsen, H.B. Nielsen, O. Tingleff, 2004
"""

def fromSphericalToCart(x):
    """ 
        Spherical to Cartesian coords

        Parameters
        ----------
        x : (3) array
            spherical coords [rho, theta, psi]

        Returns
        -------
        x_cart : (3) array
                 cartesian coords [x,y,z] 
    """

    cos_theta = np.cos(x[1])
    sin_theta = np.sin(x[1])
    cos_phi = np.cos(x[2])
    sin_phi = np.sin(x[2])

    return np.array([x[0] * cos_theta * cos_phi, x[0] * cos_theta * sin_phi, x[0] * sin_theta])

def fromSphericalToCart_array(x_array):
    """ 
        Spherical to Cartesian coords for an array of points

        Parameters
        ----------
        x_array : (n,3) array
                  spherical coords [rho_i, theta_i, psi_i] of each point

        Returns
        ------- 
        x_cart : (n,3) array
                 cartesian coords [x_i,y_i,z_i] of each point
    """
    
    n = x_array.shape[0]
    cos_thetas = np.cos(x_array[:,1])
    sin_thetas = np.sin(x_array[:,1])
    cos_phis = np.cos(x_array[:,2])
    sin_phis = np.sin(x_array[:, 2])

    res = np.empty((n,3))
    res[:,0] = x_array[:,0] * cos_thetas * cos_phis
    res[:,1] = x_array[:,0] * cos_thetas * sin_phis
    res[:,2] = x_array[:,0] * sin_thetas
    return res

def fromCartToSpherical(x):
    """ 
        Cartesian to Spherical coords 

        Parameters
        ----------
        x : (3) array
            cartesian coords [x,y,z] 
        
        Returns
        -------
        x_sphe : (3) array
                 spherical coords [rho, theta, psi]
                  
    """

    rho = np.linalg.norm(x)
    return np.array([rho, np.arcsin(x[2] / rho), np.arctan2(x[1], x[0])])

def vectToSkewMatrix(x):
    """ 
        Generate a skew-symmetric matrix from a 3D vector
        Generally used to express the cross product as a matrix vector operation (a X b = [a]_x b) 

        Parameters
        ----------
        x : (3) array
            vector
        
        Returns
        -------
        m : (3,3) array
            skew-symetric matrix 
        
        Notes
        -----
        Defined in [1], Appendix A definition 1
    """
    
    return np.array([[0., -x[2], x[1]],
                    [x[2], 0., -x[0]],
                    [-x[1], x[0], 0.]], dtype=float)

def vectToSkewMatrixArray(x_array, res):
    """ 
        Generate skew matrices from an array of 3D vector

        Parameters
        ----------
        x_array : (n,3) ndarray
                  array of vectors
        res : (n,3,3) ndarray
              resulting array of skew-symmetric matrices 

        Notes
        -----
        Defined in [1], Appendix A definition 1
    """
    
    #TODO : Faster way to do it ?
    res[:,0,1] = -x_array[:,2]
    res[:,0,2] = x_array[:,1]
    res[:,1,0] = x_array[:,2]
    res[:,1,2] = -x_array[:,0]
    res[:,2,0] = -x_array[:,1]
    res[:,2,1] = x_array[:,0]

def exp_SE3(epsilon):
    """ 
        Exponential map for SE(3) returning Rotation matrix and translation vector 

        Parameters
        ----------
        epsilon : (6) array
                  element of Lie algebra se(3)

        Returns
        -------
        R : (3,3) ndarray
            Rotation matrix of exp(epsilon)
        t : (3) array
            Translation of exp(epsilon)

        Notes
        -----
        See [2], section 9.4
    """
    
    I_3 = np.eye(3)
    w = epsilon[0:3]
    tau = epsilon[3:6]
    norm_w = np.linalg.norm(w)
    if(norm_w < 1e-8):
        R = I_3
        t = tau
    else:
        norm_w_sqr_inv = 1. / norm_w ** 2
        sin = np.sin(norm_w)
        a = sin/norm_w
        b = (1 - np.cos(norm_w))*norm_w_sqr_inv
        c = (norm_w - sin)/(norm_w**3)

        w_x = vectToSkewMatrix(w)
        w_x_sqr = np.matmul(w_x,w_x)

        R = I_3 + a*w_x + b*w_x_sqr

        V = I_3 + b*w_x + c*w_x_sqr
        t = np.matmul(V,tau)

    return R,t

def exp_SE3_euler(epsilon):
    """ 
        Exponential map for SE(3) returning pose in euler + 3d 

        Parameters
        ----------
        epsilon : (6) array
                  element of Lie algebra se(3)

        Returns
        -------
        q_euler : (6) array
                  pose in euler + 3d

        Notes
        -----
        See [2], section 9.4
    """

    R, t = exp_SE3(epsilon)
    q_euler = np.empty((6,))
    q_euler[0:3] = scipy.spatial.transform.Rotation.from_matrix(R).as_euler("ZYX")
    q_euler[3:6] = t
    return q_euler

def log_SO3(p):
    """ 
        Logarithm map for SO(3) with input pose in euler + 3d

        Parameters
        ----------
        p : (6) array
            SE(3) pose in euler + 3d

        Returns
        -------
        xi : (6) array
             element of Lie algebra se(3)

        Notes
        -----
        See [2], section 9.4 
    """

    q = scipy.spatial.transform.Rotation.from_euler('ZYX',p).as_quat()
    squared_n = q[0]*q[0] + q[1]*q[1] + q[2]*q[2]
    n = np.sqrt(squared_n)
    if (n < 1e-7):
        two_atan = 2./q[3] - 2.*squared_n/(q[3]**3)
    elif (np.abs(q[3]) < 1e-7):
        if q[3] > 0:
            two_atan = np.pi / n
        else:
            two_atan = -np.pi / n
    else:
        two_atan = 2.*np.arctan(n / q[3]) / n

    return two_atan*q[0:3]

def log_SE3(p):
    """ 
        Logarithm map for SE(3) with input pose in euler + 3d

        Parameters
        ----------
        p : (6) array
            SE(3) pose in euler + 3d
        
        Returns
        -------
        xi : (6) array
             element of Lie algebra se(3)
    """
    
    log_R = log_SO3(p[0:3])

    norm_w = np.linalg.norm(log_R)
    norm_w_sqr_inv = 1. / norm_w ** 2
    sin = np.sin(norm_w)
    w_x = vectToSkewMatrix(log_R)
    w_x_sqr = np.matmul(w_x, w_x)
    b = (1 - np.cos(norm_w)) * norm_w_sqr_inv
    c = (norm_w - sin) / (norm_w ** 3)
    V = np.eye(3) + b * w_x + c * w_x_sqr

    return np.concatenate((log_R, np.linalg.solve(V,p[3:6]))) 

def distanceSE3(p1, p2):
    """ 
        Compute the distance between two poses in euler + 3d

        Parameters
        ----------
        p1 : (6) array
             first pose
        p2 : (6) array
             second pose

        Returns
        -------
        dist : float
               distances between the two poses (in SE(3)) 
    """

    G = np.block([[2.*np.eye(3), np.zeros((3,3))],
                  [np.zeros((3,3)), np.eye(3)]])
    log = log_SE3(poses_euler.inverseComposePoseEuler(p1, p2))
    return np.sqrt(np.dot(log, G@log))

def numericalJacobian(func, output_dim, x, increments, *args, **kargs):
    """ 
        Numerically compute a jacobian 
        Used principally to test the closed form jacobian 

        Parameters
        ----------
        func : function
               function for which the jacobian is computed
        output_dim : int
                     output dimension of the function 
        x : (n) array
            point at which the jacobian is evaluated
        increments : (n) array
                     increments for computing the numerical derivatives

        Returns
        -------
        jacobian : (output_dim, n) ndarray
                   jacobian matrix
    """
    
    i = 0
    m = len(x)
    jacobian = np.zeros((output_dim, m))
    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i + incr_i
        f_plus = func(x_mod, *args, **kargs)

        x_mod[i] = x_i - incr_i
        f_minus = func(x_mod, *args, **kargs)

        denum = 0.5/incr_i
        if(output_dim == 1):
            jacobian[0][i] = denum*(f_plus - f_minus)
        else:
            for j in range(0, output_dim):
                jacobian[j][i] = denum*(f_plus[j] - f_minus[j])

        i = i+1

    return jacobian

def numericalJacobian_pool(func, output_dim, x, increments, *args, **kargs):
    """ 
        Numerically compute a jacobian (parallelized)

        Parameters
        ----------
        func : function
               function for which the jacobian is computed
        output_dim : int
                     output dimension of the function 
        x : (n) array
            point at which the jacobian is evaluated
        increments : (n) array
                     increments for computing the numerical derivatives

        Returns
        -------
        jacobian : (output_dim, n)
                   jacobian matrix
    """
    
    pool = mp.Pool(mp.cpu_count())

    i = 0
    m = len(x)
    jacobian = np.zeros((output_dim, m))
    args_list = []
    denum = []
    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i + incr_i
        args_list.append(x_mod)

    for x_i, incr_i in zip(x, increments):
        x_mod = x.copy()
        x_mod[i] = x_i - incr_i
        args_list.append(x_mod)
        denum.append(0.5/incr_i)

    func_vals = pool.map(func, args_list)
    pool.close()
    pool.join()

    jacobian[0] = denum * (func_vals[:m] - func_vals[m:])

    return jacobian

def numericalJacobian_pdf(func, output_dim, x, increments):
    """
        Numerically compute a jacobian with input being random variable 
        (ie dict with "pose_mean" and "pose_cov" keys)
        Used principally to test the closed form jacobian 

        Parameters
        ----------
        func : function
               function for which the jacobian is computed
        output_dim : int
                     output dimension of the function 
        x : (n) array
            point at which the jacobian is evaluated
        increments : (n) array
                     increments for computing the numerical derivatives

        Returns
        -------
        jacobian : (output_dim, n)
                   jacobian matrix
    """
    
    i = 0
    m = len(x["pose_mean"])
    jacobian = np.zeros((output_dim, m))
    x_mod = {"pose_mean": None, "pose_cov":x["pose_cov"]}
    for x_i, incr_i in zip(x["pose_mean"], increments):
        x_mod["pose_mean"] = x["pose_mean"].copy()
        x_mod["pose_mean"][i] = x_i + incr_i
        f_plus = func(x_mod)

        x_mod["pose_mean"][i] = x_i - incr_i
        f_minus = func(x_mod)

        denum = 0.5/incr_i

        if(output_dim == 1):
            jacobian[0][i] = denum*(f_plus - f_minus)
        else:
            for j in range(0, output_dim):
                jacobian[j][i] = denum*(f_plus[j] - f_minus[j])

        i = i+1

    return jacobian

def numericalHessian(func, output_dim, x, increments, *args):
    """ 
        Compute the hessian numerically

        TODO : Here implicitly supposed that output_dim = 1 (ie scalar function)
        if output_dim  > 1, then we obtain a tensor for the Hessian

        Parameters
        ----------
        func : function
               function for which the jacobian is computed
        output_dim : int
                     output dimension of the function 
        x : (n) array
            point at which the jacobian is evaluated
        increments : (n) array
                     increments for computing the numerical derivatives

        Returns
        -------
        hessian : (n,n) ndarray
                  Hessian matrix
    """

    output_dim_jacobian = len(x)
    return numericalJacobian(lambda y: numericalJacobian(func, output_dim, y, increments, *args).T, output_dim_jacobian,
                             x, increments, *args)

def numericalHessian_pdf(func, output_dim, x, increments):
    """ 
        Compute the hessian numerically with input being random variable 
        (ie dict with "pose_mean" and "pose_cov" keys) 

        Compute the hessian numerically

        TODO : Here implicitly supposed that output_dim = 1 (ie scalar function)
        if output_dim  > 1, then we obtain a tensor for the Hessian

        Parameters
        ----------
        func : function
               function for which the jacobian is computed
        output_dim : int
                     output dimension of the function 
        x : (n) array
            point at which the jacobian is evaluated
        increments : (n) array
                     increments for computing the numerical derivatives

        Returns
        -------
        hessian : (n,n) ndarray
                  Hessian matrix
    """

    output_dim_jacobian = len(increments)
    return numericalJacobian_pdf(lambda y: numericalJacobian_pdf(func, output_dim, y, increments).T, output_dim_jacobian,
                                 x, increments)

def LevenbergMarquardt(x0, maxIter, updateArgsFunc, func, jacobFunc, incrFunc,
                       n, tau=1., e1=1e-8, e2=1e-8, **kargsInit):
    """
        Generic LM algorithm implementation as in [3]

        TODO: should use "value" in the path "dict" (instead of "pose")
        Parameters
        ----------
        x0 : (n) array
             initial value
        maxIter : int
                  maximum number of iterations
        updateArgsFunc : function
                         function which updates intermediate values
        func : function
               function to be optimized
        jacobFunc : function
                    function computing the jacobian of func
        incrFunc : function
                   function incrementing the value to be optimized
        n : int
            dimension of x
        tau : float
              internal parameter related to the initial value of the damping parameter lambda
        e1 : float
             termination threshold on the L-inf norm of the gradient 
        e2 : float
             termination threshold on the change in x

        Returns
        -------
        x : (n) array
            optimized value
        Fval : float
               final cost function value
        path : dict with 'time', 'cost', 'pose'
               info at each iteration
    """
    x = x0

    identity = np.eye(n)

    kargs = updateArgsFunc(x, **kargsInit)

    fval = func(x, **kargs)
    J_f = jacobFunc(x, fval, **kargs)

    Fval = np.linalg.norm(fval)**2

    J_fT = np.transpose(J_f)
    H = np.matmul(J_fT, J_f)
    if(isinstance(fval,float)):
        g = fval * J_fT
    else:
        g = J_fT@fval

    #nCheck g norm 
    hasConverged = (np.linalg.norm(g, np.inf) <= e1)
    if (hasConverged):
        print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

    lambda_val = tau * np.amax(np.diagonal(H))
    iter = 0
    v = 2.

    # Keep data at each iterations
    path = {'time': [0], 'pose': [x], 'cost': [Fval]}

    while (not hasConverged and iter < maxIter):
        start = time.time()
        iter += 1
        H_inv = np.linalg.inv(H + lambda_val * identity)
        epsilon_mean = -np.matmul(H_inv, g).flatten()
        epsilon_mean_norm = np.linalg.norm(epsilon_mean)
        epsilon = {"pose_mean": epsilon_mean, "pose_cov":  H_inv}
        if (epsilon_mean_norm < (e2 ** 2)):
            hasConverged = True
            print("Ending condition on epsilon norm: {} < {}".format(epsilon_mean, e2 ** 2))
        else:
            # Increment 
            xnew = incrFunc(x, epsilon, **kargs)
            kargs = updateArgsFunc(xnew, **kargsInit)

            fval = func(xnew, **kargs)
            Fval_new = np.linalg.norm(fval)**2

            tmp = lambda_val * epsilon["pose_mean"] - g.flatten()
            denom = 0.5 * np.dot(tmp.flatten(), epsilon["pose_mean"])
            l = (Fval - Fval_new) / denom
            if (l > 0):
                x = xnew

                Fval = Fval_new
                J_f = jacobFunc(x, fval, **kargs)
                J_fT = np.transpose(J_f)
                H = np.matmul(J_fT, J_f)
                if (isinstance(fval, float)):
                    g = fval * J_fT
                else:
                    g = J_fT @ fval

                # Check g norm 
                norm_g = np.linalg.norm(g, np.inf)
                hasConverged = (norm_g <= e1)
                if (hasConverged):
                    print(" Ending condition : norm_inf(g) = {} <= e1".format(g))

                lambda_val *= np.max([0.33, 1. - (2. * l - 1.) ** 3.])
                v = 2.
                compTime = time.time() - start
                path['time'].append(path['time'][len(path['time']) - 1] + compTime)
                path['cost'].append(Fval)
                path['pose'].append(x)
            else:
                lambda_val *= v
                v *= 2.

   # print("-- LM finished --")
   # print("Final sqr error : {}".format(Fval))
   # print("Iterations : {}".format(iter))
   # print("Final pose : {}".format(x))
    return x, Fval, path