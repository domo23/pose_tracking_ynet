import numpy as np
import tensorflow as tf
import cv2
from scipy.spatial.transform import Rotation
from utils import get_resized_bbox, crop_img
from y_net import YNet

class PoseTracker:
    def __init__(self, ynet: YNet, model_3d, keypoints_3d, camera_matrix, bounding_factor = 0.3):
        self.ynet = ynet
        self.model_3d = model_3d
        self.keypoints_3d = keypoints_3d
        self.camera_matrix = camera_matrix
        self.pose = np.zeros(4)
        self.R = None
        self.t = None
        self.bounding_factor = bounding_factor
        
    
    def get_ave_xy(self, hmi, n_points = 10, thresh=0.3):
        '''
        hmi      : heatmap np array of size (height,width)
        n_points : x,y coordinates corresponding to the top  densities to calculate average (x,y) coordinates
        
        
        convert heatmap to (x,y) coordinate
        x,y coordinates corresponding to the top  densities 
        are used to calculate weighted average of (x,y) coordinates
        the weights are used using heatmap
        
        if the heatmap does not contain the probability > 
        then we assume there is no predicted landmark, and 
        x = -1 and y = -1 are recorded as predicted landmark.
        '''

        ind = np.argsort(hmi, axis=None)[-n_points:] ## pick the largest n_points
        topind = np.unravel_index(ind, hmi.shape)
        i0, i1, hsum = 0, 0, 0
        for ind in zip(topind[0],topind[1]):
            h  = hmi[ind[0],ind[1]]
            if h == 0:
                continue
            
            hsum += h
            i0   += ind[0]*h
            i1   += ind[1]*h

        if hsum == 0 or hsum/n_points <= thresh:
            i0, i1 = -1, -1
        else:
            i0 /= hsum
            i1 /= hsum
        
        return([i1,i0])
    
    
    def move_keypoints_coords_back(self, keypoints, bbox, shape):
        return tf.stack([tf.stack(
                [(x / shape[1]) * abs(bbox[0] - bbox[1]) + bbox[0],
                (y / shape[0]) * abs(bbox[2] - bbox[3]) + bbox[2]])
                if x != -1 and y != -1 
                else [-1, -1] 
                for x, y in keypoints]).numpy()
    
    
    def get_best_approx(self, rvecs, R_prev):
        if R_prev is None:
            return cv2.Rodrigues(rvecs[0])[0]

        evalueated = []
        for rvec in rvecs:
            r = cv2.Rodrigues(rvec)[0]
            err = np.mean(r - R_prev) ** 2
            evalueated.append([err, r])
        
        evalueated.sort(key=lambda x: x[0])
        return evalueated[0][1]
    
    
    def get_rotation_translation(self, keypoint_hms, bbox, keypoints_3d, camera_matrix, rvec=None, tvec=None):
        keypoints_coords = [self.get_ave_xy(keypoint_hms[:, :, i]) for i in range(keypoint_hms.shape[2])]
        keypoints_2d = self.move_keypoints_coords_back(keypoints_coords, bbox, keypoint_hms.shape)
        filtered = list(filter(lambda x: (x[0] > 0).all(), list(zip(keypoints_2d, keypoints_3d))))
        
        if len(filtered) < 4:
            return np.zeros((3,3)), np.zeros(3)

        kp_2d, kp_3d = list(zip(*filtered))

        success, rvecs, tvecs, err = cv2.solvePnPGeneric(
            np.stack(kp_3d[:4]), 
            np.stack(kp_2d[:4]), 
            camera_matrix, None, 
            flags=cv2.SOLVEPNP_SQPNP, 
            rvec=rvec, 
            tvec=tvec)

        
        rmatrix = self.get_best_approx(rvecs, rvec)
        
        return rmatrix, tvecs[0].flatten()
        
    
    def get_bbox_from_pose(self, R, t, model, camera_matrix):
        if R is None and t is None:
            return 0, 640, 0, 480
        camera_coords = R.dot(model.T) + t.reshape((3, 1))
        img_coords_ = camera_matrix.dot(camera_coords)
        img_coords = img_coords_[:2] / img_coords_[2]
        return img_coords[0].min(), img_coords[0].max(), img_coords[1].min(), img_coords[1].max()
    
    
    def predict(self, image):
        bbox = self.get_bbox_from_pose(self.R, self.t, self.model_3d, self.camera_matrix)
        resized_bbox = get_resized_bbox(bbox, image.shape, np.ones(4) * self.bounding_factor)
        cropped_image = crop_img(image, resized_bbox, self.ynet.img_width, self.ynet.img_height)
        pred_keypoints = self.ynet.model.predict((np.expand_dims(cropped_image, 0), np.expand_dims(self.pose, 0)), verbose=0)
        r_matrix, t_vec = self.get_rotation_translation(pred_keypoints[0], resized_bbox, self.keypoints_3d, self.camera_matrix, self.R, self.t)

        if r_matrix.any():
            self.R = r_matrix
            self.t = t_vec
            self.pose = Rotation.from_matrix(r_matrix).as_quat()
        else:
            self.pose = np.zeros(4)
            self.R = None
            self.t = None
        

        # hm = np.zeros(pred_keypoints[0].shape, dtype = np.float32)
        # for i in range(9):
        #     x, y = get_ave_xy(pred_keypoints[0, :, :, i])
        #     hm[:, :, i] = gaussian_k(x, y, 3, cropped_image.shape[0], cropped_image.shape[1])

        # clear_output(wait=True)
        # Canvas().display(cropped_image, self.pose, pred_keypoints, hm)

        self.meta = {
            'img': cropped_image,
            'pred_keypoints': pred_keypoints,
            'pose': self.pose
        }

        if self.R is not None and self.t is not None:
            return self.R.copy(), self.t.copy()
        else: 
            return np.zeros((3,3)), np.zeros(3)
    
    def init_pose(self, pose):
        self.pose = np.array(pose[:4])
        self.R = Rotation.from_quat(pose[:4]).as_matrix()
        self.t = np.array(pose[4:])