from typing import List

import logging

import numpy as np
import torch
import yacs.config

from gaze_estimation.gaze_estimator.common import Camera, Face, FacePartsName, MODEL3D
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from gaze_estimation import (GazeEstimationMethod, create_model,
                             create_transform)

import pdb
import time
logger = logging.getLogger(__name__)

SHIFT_PIXELS = 0
PRINT_MODEL_PARAMS = 1
class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: yacs.config.CfgNode, AVG_LANDMARKS=0, num_frames=None):
        self._config = config
        #pdb.set_trace()
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config, AVG_LANDMARKS, num_frames)
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)
        self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device(self._config.device))
        model.eval()
        if PRINT_MODEL_PARAMS:
            num_params = sum(x.numel() for x in model.parameters())
            num_train = sum(x.numel() for x in model.parameters() if x.requires_grad)
            print('TOTAL nr of params = ', num_params)
            #print('TOTAL nr of trainable params = ', num_train)

        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        # Estimation of the head pose rotation matrix (3x3) and the head pose (3D coords)
        pose_time = time.time()
        MODEL3D.estimate_head_pose(face, self.camera)
        if 0:
            print('Pose faces: ', time.time() - pose_time, ' seconds.')
        # Fits the 3D landmark model with head pose rotation matrix and the head pose vector 
        pose3d_time = time.time()
        MODEL3D.compute_3d_pose(face)
        if 0:
            print('3D Pose faces: ', time.time() - pose3d_time, ' seconds.')
        # Compute the face center (left right eye and mouth indicies)
        # Compute eye centers using the (corresponding eye indicies)
        center_time = time.time()
        MODEL3D.compute_face_eye_centers(face)
        if 0:
            print('Face center: ', time.time() - center_time, ' seconds.')

        # Image normalization
        if self._config.mode == GazeEstimationMethod.MPIIGaze.name:
            norm_time = time.time()
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                # head pose normalizer!
                self._head_pose_normalizer.normalize(image, eye)
            if 0:
                print('Normalization: ', time.time() - norm_time, ' seconds.')
                model_time = time.time()
            self._run_mpiigaze_model(face)
            if 0:
                print('Prediction: ', time.time() - model_time, ' seconds.')
        elif self._config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        # OWND STuFF
        shift_images = []

        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            #pdb.set_trace()
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
                
        if SHIFT_PIXELS:
            shift_images = torch.stack(shift_images)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            if SHIFT_PIXELS:
                pdb.set_trace()
                import matplotlib.pyplot as plt
                plt.ion()
                test_img = np.squeeze(image.numpy())
                shift_img = np.zeros((test_img.shape))
                # SHIFT UP
                shift_img[0:-5][:] = test_img[5:][:]
                self._gaze_estimation_model(shift_images, head_poses)
                #torch.Tensor(shift_img).unsqueeze_(0)

            # INPUT IS CONCATENATION OF LEFT AND RIGHT EYE PATCH
            predictions = self._gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()
            #

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

    def _run_mpiifacegaze_model(self, face: Face) -> None:
        # pdb.set_trace()
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        with torch.no_grad():
            image = image.to(device)
            prediction = self._gaze_estimation_model(image)
            prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
