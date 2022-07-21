import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def get_resized_bbox(bbox, img_shape, boundary_factor=None):
    if boundary_factor is None:
        boundary_factor = np.random.random(4)
                
    width = abs(bbox[1] - bbox[0])
    height = abs(bbox[3] - bbox[2])
    
    return [
        max(bbox[0] - boundary_factor[0] * width, 0),
        min(bbox[1] + boundary_factor[1] * width, img_shape[1] - 1),
        max(bbox[2] - boundary_factor[2] * height, 0),
        min(bbox[3] + boundary_factor[3] * height, img_shape[0] - 1)
    ]
        
        
def crop_img(img, bbox, img_height, img_width):
    crop_img = img[int(bbox[2]): int(bbox[3]), int(bbox[0]): int(bbox[1])]
    crop_img = cv2.resize(crop_img, (img_height, img_width))
    return np.asarray(crop_img)


def get_rmatrix_tvec_from_gt(gt_vec):
    return Rotation.from_quat(gt_vec[:4]).as_matrix(), np.array(gt_vec[4:])


def get_pose_dict(r_matrix, t_vec):
    return {'R': r_matrix, 't': t_vec}


def project_to_2d(R, t, model, camera_matrix):
    camera_coords = R.dot(model.T) + t.reshape((3, 1))
    img_coords_ = camera_matrix.dot(camera_coords)
    return img_coords_[:2] / img_coords_[2]


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def add_error(pose_est, pose_gt, model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param pose_est: Estimated pose given by a dictionary:
    {'R': 3x3 rotation matrix, 't': 3x1 translation vector}.
    :param pose_gt: The ground truth pose given by a dictionary (as pose_est).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_gt = transform_pts_Rt(model, pose_gt['R'], pose_gt['t'])
    pts_est = transform_pts_Rt(model, pose_est['R'], pose_est['t'])
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e