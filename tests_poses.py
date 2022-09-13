''' Operations and functions related to SE(3) poses
'''

import math_utility
import poses_quat
import poses_euler
import poses2D
import time
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats

def testNumericalJacobian():
    ''' GT : [[2, -4],
              [-2, 12]] '''
    jacobian_gt = np.array([[2, -4], [-2, 12]])
    
    jacobian = math_utility.numericalJacobian(lambda x: [2 * x[0] - x[1] ** 2, 3 * x[1] ** 2 - x[0] ** 2], 2, [1., 2.],
                                 [0.001, 0.001])
    print("Test numerical Jacobian : {}".format(jacobian))

    assert np.allclose(jacobian_gt, jacobian, atol=1e-5)

def testJacobian_composePose2D():
    q1 = [1.,2.,0.1]
    q2 = [-2.,3.,-0.2]
    incr = np.full((3,), 0.0001)
    j_num_q1 = math_utility.numericalJacobian(lambda x: poses2D.composePose(x, q2), 3, q1, incr)
    j_num_q2 = math_utility.numericalJacobian(lambda x: poses2D.composePose(q1, x), 3, q2, incr)
    j_computed_q1, j_computed_q2 = poses2D.computeJacobian_composePose(q1,q2)

    print("Test Jacobian composePose 2D")
    print("J_num_q1 :")
    print(j_num_q1)
    print("J_computed_q1 :")
    print(j_computed_q1)
    print("J_num_q2 :")
    print(j_num_q2)
    print("J_computed_q2 :")
    print(j_computed_q2)

    assert np.allclose(j_num_q1, j_computed_q1, atol=1e-5) and \
           np.allclose(j_num_q2, j_computed_q2, atol=1e-5)

def testJacobian_composePosePoint_2D():
    q = [1., 2., 0.1]
    point = [-2.3, 3.6]
    incr = np.full((3,), 0.0001)
    j_num_q     = math_utility.numericalJacobian(lambda x: poses2D.composePosePoint(x, point), 2, q, incr)
    j_num_point = math_utility.numericalJacobian(lambda x: poses2D.composePosePoint(q, x), 2, point, incr[0:2])
    j_computed_q, j_computed_point = poses2D.jacobian_composePosePoint(q, point)

    print("Test Jacobian composePosePoint 2D")
    print("J_num_q :")
    print(j_num_q)
    print("J_computed_q :")
    print(j_computed_q)
    print("J_num_point :")
    print(j_num_point)
    print("J_computed_point :")
    print(j_computed_point)

    assert np.allclose(j_num_q, j_computed_q, atol=1e-5) and \
           np.allclose(j_num_point, j_computed_point, atol=1e-5)

def testJacobian_composePosePoint_array_2D():
    q = [1., 2., 0.1]
    point_array = np.array([[-2.3, 3.6],
                            [5.3, 2.2],
                            [-1.2, 0.1]])
    incr = np.full((3,), 0.0001)

    j_num_point_array = np.empty((3,2,2))
    j_num_q = np.empty((3, 2, 3))
    for i in range(0,3):
        j_num_q[i,:,:] = math_utility.numericalJacobian(lambda x: poses2D.composePosePoint(x, point_array[i]), 2, q, incr)
        j_num_point_array[i,:, : ] = math_utility.numericalJacobian(lambda x: poses2D.composePosePoint(q, x), 2, point_array[i], incr[0:2])
    j_computed_q, j_computed_point_array = poses2D.jacobian_composePosePoint_array(q, point_array)

    print("Test Jacobian composePosePoint 2D")
    print("J_num_q :")
    print(j_num_q)
    print("J_computed_q :")
    print(j_computed_q)
    print("J_num_point_array :")
    print(j_num_point_array)
    print("J_computed_point_array :")
    print(j_computed_point_array)

    assert np.allclose(j_num_q, j_computed_q, atol=1e-5) and \
           np.allclose(j_num_point_array, j_computed_point_array, atol=1e-5)

def testJacobianNormalization():
    q_poseEuler = [0.5, -0.3, 0.5, 1.5, -2.2, 1.6]
    q_poseQuat = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)
    J_closedForm = poses_quat.jacobianQuatNormalization(q_poseQuat)
    J_num = math_utility.numericalJacobian(lambda x: (1./np.linalg.norm(x))*x, 4, q_poseQuat[0:4],
                                           np.full((4,),0.001))
    print("Test JacobianNormalization")
    print("J_closedForm : ")
    print(J_closedForm)
    print("J_num : ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5)

def testComputeJacobian_eulerToQuat():
    q_poseEuler = [0.5,-0.2,0.4,1.5,-2.2,1.6]
    J_closedForm = poses_euler.computeJacobian_eulerToQuat(q_poseEuler)
    J_num = math_utility.numericalJacobian(lambda x : poses_euler.fromPoseEulerToPoseQuat(x), 7, q_poseEuler, np.full((6,),0.001))
    print("Test ComputeJacobian_eulerToQuat")
    print("J_closedForm : ")
    print(J_closedForm)
    print("J_num : ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5)

def testComputeJacobian_quatToEuler():
    q_poseEuler = [0.5,-0.2,0.4,1.5,-2.2,1.6]
    q_poseQuat = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)

    J_closedForm = poses_euler.computeJacobian_quatToEuler(q_poseQuat)
    J_num = math_utility.numericalJacobian(lambda x: poses_euler.fromPoseQuatToPoseEuler(x), 6, q_poseQuat,
                                           np.full((7,),1e-8))
    print("Test ComputeJacobian_quatToEuler")
    print("J_closedForm : ")
    print(J_closedForm)
    print("J_num : ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5)

def testComputeJacobianQuat_composePosePoint():
    q_poseEuler = [0.5, -0.3, 0.5, 1.5, -2.2, 1.6]
    q_poseQuat  = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)
    print("q_poseQuat: {}".format(q_poseQuat))

    point = [1.,2.,3.]
    J_closedForm_pose, J_closedForm_point = poses_quat.computeJacobianQuat_composePosePoint(q_poseQuat,point)
    J_num_pose = math_utility.numericalJacobian(lambda x: poses_quat.composePoseQuatPoint(np.block([(x[0:4] / np.linalg.norm(x[0:4])), x[4:7]]),point), 3, q_poseQuat,
                                   np.full((7,), 1e-8))
    J_num_point = math_utility.numericalJacobian(lambda x: poses_quat.composePoseQuatPoint(q_poseQuat,x), 3, point,
                                   np.full((3,), 1e-8))
    print("Test ComputeJacobianQuat_composePosePoint")
    print("J_closedForm_pose : ")
    print(J_closedForm_pose)
    print("J_num_pose : ")
    print(J_num_pose)
    print("J_closedForm_point : ")
    print(J_closedForm_point)
    print("J_num_point : ")
    print(J_num_point)

    assert np.allclose(J_closedForm_pose, J_num_pose, atol=1e-5) and \
    np.allclose(J_closedForm_point, J_num_point, atol=1e-5)

def testComputeJacobianQuat_composePose():
    q_poseEuler1 = [0.5, -0.2, 0.4, 1.5, -2.2, 1.6]
    q_poseEuler2 = [0.2,0.2,-0.1, 1, 2, -2]
    q_poseQuad1 = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler1)
    q_poseQuad2 = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler2)

    #q_poseQuad1 = [0.137515,-0.860573,-0.28688,-0.397748,96.5692,21.1987,-18.7762]
    #q_poseQuad2 = [0.391368,-0.631567,0.66872,-0.0277322,-67.5089, 19.9077,-44.0226]

    q_poseEuler_compose = poses_euler.composePoseEuler(q_poseEuler1, q_poseEuler2)
    q_poseQuat_compose = poses_quat.composePoseQuat(q_poseQuad1, q_poseQuad2)
    print("q_poseQuat_compose : {}".format(q_poseQuat_compose))

    q_poseEuler_test = poses_euler.fromPoseQuatToPoseEuler(q_poseQuat_compose)
    print("poseEuler gt : {}".format(q_poseEuler_compose))
    print("poseEuler test : {}".format(q_poseEuler_test))

    print("q_poseQuad1 : {}".format(q_poseQuad1))
    print("q_poseQuad2 : {}".format(q_poseQuad2))
    J_closedForm_q1, J_closedForm_q2 = poses_quat.computeJacobianQuat_composePose(q_poseQuad1, q_poseQuad2)
    J_num_q1 = math_utility.numericalJacobian(lambda x: poses_quat.composePoseQuat(np.block([(x[0:4] / np.linalg.norm(x[0:4])), x[4:7]]), q_poseQuad2), 7, q_poseQuad1,
                              np.full((7,), 1e-8))
    J_num_q2 = math_utility.numericalJacobian(lambda x: poses_quat.composePoseQuat(q_poseQuad1, np.block([x[0:4] / np.linalg.norm(x[0:4]), x[4:7]])), 7, q_poseQuad2,
                              np.full((7,), 1e-8))
    
    print("Test ComputeJacobianQuat_composePose : ")
    print("J_closedForm_q1 : ")
    print(J_closedForm_q1)
    print("J_num_q1 : ")
    print(J_num_q1)
    print("J_closedForm_q2 : ")
    print(J_closedForm_q2)
    print("J_num_q2 : ")
    print(J_num_q2)

    assert np.allclose(J_closedForm_q1, J_num_q1, atol=1e-5) and \
        np.allclose(J_closedForm_q2, J_num_q2, atol=1e-5)


def testComputeJacobianEuler_composePose():
    q_poseEuler1 = [0.5, -0.2, 0.4, 1.5, -2.2, 1.6]
    q_poseEuler2 = [0.2, 0.2, -0.1, 1, 2, -2]
    q_poseEuler_compose = poses_euler.composePoseEuler(q_poseEuler1, q_poseEuler2)
    J_closedForm_q1, J_closedForm_q2 = poses_euler.computeJacobianEuler_composePose(q_poseEuler1, q_poseEuler2, q_poseEuler_compose)
    J_num_q1 = math_utility.numericalJacobian(lambda x: poses_euler.composePoseEuler(x, q_poseEuler2), 6, q_poseEuler1,
                                 np.full((7,), 0.001))
    J_num_q2 = math_utility.numericalJacobian(lambda x: poses_euler.composePoseEuler(q_poseEuler1, x), 6, q_poseEuler2,
                                 np.full((7,), 0.001))
    print("Test ComputeJacobianEuler_composePose")
    print("J_closedForm_q1 : ")
    print(J_closedForm_q1)
    print("J_num_q1 : ")
    print(J_num_q1)
    print("J_closedForm_q2 : ")
    print(J_closedForm_q2)
    print("J_num_q2 : ")
    print(J_num_q2)

    assert np.allclose(J_closedForm_q1, J_num_q1, atol=1e-5) and \
        np.allclose(J_closedForm_q2, J_num_q2, atol=1e-5)

def testComputeJacobian_inversePoseQuatPoint_pose():
    q_poseEuler = [0.5, -0.3, 0.5, 1.5, -2.2, 1.6]
    q_poseQuat = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)
    print("q_poseQuat: {}".format(q_poseQuat))
    point = [1., 2., 3.]
    print("a_compose : {}".format(poses_quat.inversePoseQuat_point(q_poseQuat,point)))
    J_closedForm = poses_quat.computeJacobian_inversePoseQuatPoint_pose(q_poseQuat, point)

    ''' We expect unit quaternion but here need to explicitly normalize the quaternion '''
    J_num = math_utility.numericalJacobian(lambda x: poses_quat.inversePoseQuat_point(np.block([(1./np.linalg.norm(x[0:4]))*x[0:4],x[4:7]]),point),
                                           3, q_poseQuat,
                                          np.full((7,), 0.001))
    print("Test testComputeJacobian_inversePoseQuatPoint_pose")
    print("J_closedForm: ")
    print(J_closedForm)
    print("J_num: ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5) 

def testComputeJacobian_inversePoseQuat():
    q_poseEuler = [0.5, -0.3, 0.5, 1.5, -2.2, 1.6]
    q_poseQuat = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)
    print("q_poseQuat: {}".format(q_poseQuat))
    J_closedForm = poses_quat.computeJacobian_inversePoseQuat(q_poseQuat)

    ''' We expect unit quaternion but here need to explicitly normalize the quaternion '''
    J_num = math_utility.numericalJacobian(
        lambda x: poses_quat.inversePoseQuat(np.block([(1. / np.linalg.norm(x[0:4])) * x[0:4], x[4:7]])),
        7, q_poseQuat,
        np.full((7,), 0.001))
    print("Test testComputeJacobian_inversePoseQuat")
    print("J_closedForm: ")
    print(J_closedForm)
    print("J_num: ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5) 

def testComposesAndInverseComposes():
    q_poseEuler_1 = [0.,0.,0.,1,2,3]
    q_poseEuler_2 = [0.,0.,0.,3,2,1]
    q_poseQuat_1 = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler_1)
    q_poseQuat_2 = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler_2)

    q_poseQuat_composed  = poses_quat.composePoseQuat(q_poseQuat_1, q_poseQuat_2)
    q_poseEuler_composed = poses_euler.composePoseEuler(q_poseEuler_1, q_poseEuler_2)

    q_poseEuler_inverse = poses_euler.inverseComposePoseEuler(q_poseEuler_1, q_poseEuler_2)

    q_poseEuler_composed_converted = poses_euler.fromPoseQuatToPoseEuler(q_poseQuat_composed)

    print("----> Test compose poses ")
    print(" q_poseEuler_composed : {}".format(q_poseEuler_composed))
    print(" q_poseEuler_composed_converted : {}".format(q_poseEuler_composed_converted))
    print(" q_poseEuler_inverse : {}".format(q_poseEuler_inverse))

    assert np.allclose(q_poseEuler_composed_converted, q_poseEuler_composed, atol=1e-5) 

def testComposePoseQuatPoint():
    q_poseEuler_1 = [0.5*np.pi, 0., 0., 1, 2, 3]
    point = [1,1,1]
    point_composed_gt = np.array([0,3,4])
    q_poseQuat_1 = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler_1)

    point_computed = poses_quat.composePoseQuatPoint(q_poseQuat_1, point)
    print("test ComposePoseQuatPoint")
    print("pt_composed : {}".format(point_computed))

    assert np.allclose(point_computed, point_composed_gt, atol=1e-5)

def testComposePosePDFQuatPoint():
    cov = np.zeros((7, 7))
    cov[4][4] = 1.
    cov_pt = np.zeros((3, 3))
    cov_pt[0][0] = 1.
    q_posePDFQuat = {'pose_mean': np.array([0., 0., 0.,1., 1, 2, 3]), 'pose_cov': cov}
    pointPDF = {'mean': np.array([[1, 1, 1]]), 'cov': np.array([cov_pt])}

    pointPDF_compose = poses_quat.composePosePDFQuatPoint(q_posePDFQuat, pointPDF)
    print("Test ComposePosePDFQuatPoint")
    print(pointPDF_compose)

def testComputeJacobianEuler_composePosePDFPoint_pose():
    mat_cov = np.random.rand(6,6)
    mat_cov_pt = np.random.rand(3,3)
    cov = mat_cov.T@mat_cov
    cov_pt = mat_cov_pt.T@mat_cov_pt
    q_posePDFEuler = {'pose_mean' : np.array([0.3, 0.4, -0.3, 1, 2, 3]), 'pose_cov' : cov}
    pointPDF = {'mean': np.array([1, 1, 1]), 'cov': cov_pt}

    J_closedForm = poses_euler.computeJacobianEuler_composePosePDFPoint_pose(q_posePDFEuler['pose_mean'], pointPDF['mean'])

    ''' We expect unit quaternion but here need to explicitly normalize the quaternion '''
    J_num = math_utility.numericalJacobian(
        lambda x: poses_euler.composePoseEulerPoint(x,pointPDF["mean"]),
        3, q_posePDFEuler["pose_mean"],
        np.full((6,), 0.001))
    print("Test computeJacobianEuler_composePosePDFPoint_pose")
    print("J_closedForm: ")
    print(J_closedForm)
    print("J_num: ")
    print(J_num)

    assert np.allclose(J_closedForm, J_num, atol=1e-5)

def testComposePosePDFEulerPoint():
    cov = np.zeros((6,6))
    cov[3][3] = 1.
    cov_pt = np.zeros((3,3))
    cov_pt[0][0] = 1.
    q_posePDFEuler = {'pose_mean': np.array([0., 0., 0., 1, 2, 3]), 'pose_cov': cov}
    pointPDF = {'mean': np.array([1, 1, 1]), 'cov': cov_pt}

    pointPDF_compose = poses_euler.composePosePDFEulerPoint(q_posePDFEuler, pointPDF)
    print("Test ComposePosePDFEulerPoint")
    print(pointPDF_compose)

def testComposePDF():
    cov = np.eye(6)
    q_posePDFEuler_1 = {'pose_mean' : np.array([0.5, 0.2, -0.1, 1, 2, 3]), 'pose_cov': cov}
    q_posePDFEuler_2 = {'pose_mean' : np.array([0.2, -0.1, 0.2, 3, 2, 1]), 'pose_cov': cov}

    q_posePDFEuler_1 = {'pose_mean': np.array([np.pi*0.5,0,0,0,0,3]), 'pose_cov': cov}
    q_posePDFEuler_2 = {'pose_mean': np.array([0,0,0,0,0,1]), 'pose_cov': cov}
    q_posePDFEuler_composed = poses_euler.composePosePDFEuler(q_posePDFEuler_1, q_posePDFEuler_2)

    q_posePDFQuat_1 = poses_euler.fromPosePDFEulerToPosePDFQuat(q_posePDFEuler_1)
    q_posePDFQuat_2 = poses_euler.fromPosePDFEulerToPosePDFQuat(q_posePDFEuler_2)
    q_posePDFQuat_composed = poses_quat.composePosePDFQuat(q_posePDFQuat_1, q_posePDFQuat_2)

    q_posePDFEuler_composed_converted = poses_euler.fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_composed)

    print("----> Test compose poses ")
    print(" q_posePDFEuler_composed : {}".format(q_posePDFEuler_composed))
    print(" q_posePDFEuler_composed_converted : {}".format(q_posePDFEuler_composed_converted))

    assert np.allclose(q_posePDFEuler_composed['pose_mean'], q_posePDFEuler_composed_converted['pose_mean'], atol=1e-5) and np.allclose(q_posePDFEuler_composed['pose_cov'], q_posePDFEuler_composed_converted['pose_cov'], atol=1e-5)

def testComposePosearray():
    cov = np.eye(6)
    q_poseEuler = np.array([0.2,0.3,0.4,1,-2,3])
    q_poseEuler_array = np.array([[0.5, 0.2, -0.1, 1, 2, 3],
                                  [0.2,-0.2,0.5, 2,4,3]])
    q_composePoseEuler = poses_euler.composePoseEuler_array(q_poseEuler, q_poseEuler_array)
    q_composePoseEuler_gt = np.empty((2,6))
    q_composePoseEuler_gt[0] = poses_euler.composePoseEuler(q_poseEuler, q_poseEuler_array[0])
    q_composePoseEuler_gt[1] = poses_euler.composePoseEuler(q_poseEuler, q_poseEuler_array[1])

    q_poseQuat = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler)
    q_poseQuat_array = np.empty((2, 7))
    q_poseQuat_array[0] = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler_array[0])
    q_poseQuat_array[1] = poses_euler.fromPoseEulerToPoseQuat(q_poseEuler_array[1])
    q_composePoseQuat_q1q2array = poses_quat.composePoseQuat_array(q_poseQuat, q_poseQuat_array)
    q_composePoseQuat_q1arrayq2 = poses_quat.composePoseQuat_array(q_poseQuat_array, q_poseQuat)
    q_composePoseQuat_q1q2array_gt = np.empty((2,7))
    q_composePoseQuat_q1arrayq2_gt = np.empty((2,7))
    q_composePoseQuat_q1q2array_gt[0] = poses_quat.composePoseQuat(q_poseQuat, q_poseQuat_array[0])
    q_composePoseQuat_q1q2array_gt[1] = poses_quat.composePoseQuat(q_poseQuat, q_poseQuat_array[1])
    q_composePoseQuat_q1arrayq2_gt[0] = poses_quat.composePoseQuat(q_poseQuat_array[0], q_poseQuat)
    q_composePoseQuat_q1arrayq2_gt[1] = poses_quat.composePoseQuat(q_poseQuat_array[1], q_poseQuat)

    print("----> Test compose poses array")
    print(" q_composePoseEuler : {}".format(q_composePoseEuler))
    print(" q_composePoseEuler_gt : {}".format(q_composePoseEuler_gt))
    print(" q_composePoseQuat_q1q2array :{}".format(q_composePoseQuat_q1q2array))
    print(" q_composePoseQuat_q1q2array_gt: {}".format(q_composePoseQuat_q1q2array_gt))
    print(" q_composePoseQuat_q1arrayq2 :{}".format(q_composePoseQuat_q1arrayq2))
    print(" q_composePoseQuat_q1arrayq2_gt: {}".format(q_composePoseQuat_q1arrayq2_gt))

    assert np.allclose(q_composePoseEuler, q_composePoseEuler_gt, atol=1e-5) and \
        np.allclose(q_composePoseQuat_q1q2array, q_composePoseQuat_q1q2array_gt, atol=1e-5) and \
        np.allclose(q_composePoseQuat_q1arrayq2, q_composePoseQuat_q1arrayq2_gt, atol=1e-5)        

def testInverseComposePosePDF():
    cov1 = np.eye(6)
    cov2 = 1.5*np.eye(6)
    q_posePDFEuler_1 = {'pose_mean': np.array([0.1, 0, 0., 1, 2, 3]), 'pose_cov': cov1}
    q_posePDFEuler_2 = {'pose_mean': np.array([0.2, 0, 0, 3, 2, 1]), 'pose_cov': cov2}
    q_posePDFEuler_inverseComposed = poses_euler.inverseComposePosePDFEuler(q_posePDFEuler_1, q_posePDFEuler_2)

    q_inverseCompose = poses_euler.inverseComposePoseEuler(q_posePDFEuler_1["pose_mean"], q_posePDFEuler_2["pose_mean"])

    q_posePDFQuat_1 = poses_euler.fromPosePDFEulerToPosePDFQuat(q_posePDFEuler_1)
    q_posePDFQuat_2 = poses_euler.fromPosePDFEulerToPosePDFQuat(q_posePDFEuler_2)

    q_inverseQuat = poses_quat.inversePoseQuat(q_posePDFQuat_1["pose_mean"])
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX', q_posePDFEuler_1["pose_mean"][0:3])
    rot_inv = rot.inv()
    q_inverseQuat_ = poses_quat.fromqxyzrToqrxyz(rot_inv.as_quat())
    print("Q_inverseQuat actuel : {}".format(q_inverseQuat))
    print("Q_inverse avec rot : {}".format(q_inverseQuat_))

    q_posePDFQuat_inverseComposed = poses_quat.inverseComposePosePDFQuat(q_posePDFQuat_1, q_posePDFQuat_2)

    q_posePDFEuler_inverseComposed_converted = poses_euler.fromPosePDFQuatToPosePDFEuler(q_posePDFQuat_inverseComposed)

    print("----> Test compose poses ")
    print(" q_inverseCompose : {}".format(q_inverseCompose))
    print(" q_posePDFEuler_inverseComposed : {}".format(q_posePDFEuler_inverseComposed))
    print(" q_posePDFEuler_inverseComposed_converted : {}".format(q_posePDFEuler_inverseComposed_converted))

    assert np.allclose(q_posePDFEuler_inverseComposed['pose_mean'], q_posePDFEuler_inverseComposed_converted['pose_mean'], atol=1e-5) and\
    np.allclose(q_posePDFEuler_inverseComposed['pose_cov'], q_posePDFEuler_inverseComposed_converted['pose_cov'], atol=1e-5)

def testTime():
    q_poseEuler = np.array([[0.1,0.2,0.3,1.,1.,1.]] * 2)
    x = np.array([1.,2.,3.])
    composed_pt_ = poses_euler.inverseComposePoseEulerPoint(q_poseEuler[0], x)
    composed_pt = poses_euler.inverseComposePoseEulerPoint_opt(q_poseEuler, x)
    print(composed_pt_)
    print(composed_pt)

    '''start = time.time()
    composed_pt = poses_euler.inverseComposePoseEulerPoint(q_poseEuler,x)
    comp_time = time.time() - start
    print("Composed pt : {}".format(composed_pt))
    print("Comp time old : {}".format(comp_time))

    R = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3]).as_matrix()
    composed_pt_opt = poses_euler.inverseComposePoseEulerPoint_opt(R, q_poseEuler[3:6], x)

    start = time.time()
    R = scipy.spatial.transform.Rotation.from_euler('ZYX', q_poseEuler[0:3]).as_matrix()
    composed_pt_opt = poses_euler.inverseComposePoseEulerPoint_opt(R,q_poseEuler[3:6],x)
    comp_time_opt = time.time() - start
    print("Composed pt : {}".format(composed_pt_opt))
    print("Comp time opt : {}".format(comp_time_opt))
'''

def testArray():
    cov = np.full((6,6),0.2)
    q_posePDFeuler_array1 = {'pose_mean' : None, 'pose_cov': None}
    q_posePDFeuler_2 = {'pose_mean': None, 'pose_cov': None}
    q_posePDFeuler_array1['pose_mean'] = np.array([[0.1, 0.2, 0.3,1,2,3],
                                                  [0.1, 0.2, 0.3,1,2,3]])
    q_posePDFeuler_2['pose_mean'] = np.array([-0.1, -0.2, -0.3, 3, 2, 1])
    q_posePDFeuler_array1['pose_cov'] = np.empty((2,6,6))
    q_posePDFeuler_array1['pose_cov'][0,:,:] = cov
    q_posePDFeuler_array1['pose_cov'][1,:,:] = cov
    q_posePDFeuler_2['pose_cov'] = cov

    q_posePDFQuat_array1 = poses_euler.fromPosePDFEulerToPosePDFQuat_array(q_posePDFeuler_array1)
    q_posePDFQuat2 = poses_euler.fromPosePDFEulerToPosePDFQuat(q_posePDFeuler_2)

    q_poseeuler1_ = poses_euler.fromPosePDFQuatToPosePDFEuler_array(q_posePDFQuat_array1)

    x_array = {'mean' : None, 'cov':None}
    x_array['mean'] = np.random.rand(2,3)
    x_array['cov'] = np.random.rand(2,3,3)
    print("Array version :")
    start = time.time()
    res = poses_euler.composePosePDFEulerPoint_array(q_posePDFeuler_2, x_array)
    #res = poses_euler.composePosePDFEuler_array(q_posePDFeuler_2,q_posePDFeuler_array1)
    #res = poses_quat.inverseComposePosePDFQuat_array(q_posePDFQuat_array1, q_posePDFQuat2)
    print("Comp time : {}".format(time.time() - start))
    print(res)

    print("Old values : ")
    start = time.time()
    res1 = poses_euler.composePosePDFEulerPoint(q_posePDFeuler_2, {'mean': x_array['mean'][0], 'cov':x_array['cov'][0]})
    res2 = poses_euler.composePosePDFEulerPoint(q_posePDFeuler_2, {'mean': x_array['mean'][1], 'cov':x_array['cov'][1]})
    '''res1 = poses_quat.inverseComposePosePDFQuat({'pose_mean': q_posePDFQuat_array1['pose_mean'][0],
                                                 'pose_cov': q_posePDFQuat_array1['pose_cov'][0]}, q_posePDFQuat2)
    res2 = poses_quat.inverseComposePosePDFQuat({'pose_mean': q_posePDFQuat_array1['pose_mean'][1],
                                                      'pose_cov': q_posePDFQuat_array1['pose_cov'][1]}, q_posePDFQuat2)'''
    print("Comp time : {}".format(time.time() - start))
    print(res1)
    print(res2)

#------------- MAIN ---------------------
if __name__ == "__main__":
    testComputeJacobian_quatToEuler()

