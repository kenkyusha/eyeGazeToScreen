from typing import List

import dlib
import numpy as np
import yacs.config
import pdb

from gaze_estimation.gaze_estimator.common import Face


class LandmarkEstimator:
    def __init__(self, config: yacs.config.CfgNode, AVG_LANDMARKS=0, num_frames=3):
        self.mode = config.face_detector.mode
        self.AVG_LANDMARKS = AVG_LANDMARKS
        self.num_frames = num_frames
        if self.mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                config.face_detector.dlib.model)
            self.landmark_holder = []
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        else:
            raise ValueError

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        #pdb.set_trace()
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            # THIS IS FOR AVERAGING OVER LANDMARK FRAMES!
            if self.AVG_LANDMARKS:
                self.landmark_holder.append(landmarks)
                if len(self.landmark_holder) >= self.num_frames:
                    #pdb.set_trace()
                    landmarks = np.mean(self.landmark_holder[-self.num_frames:],axis=0)
            # --------------------------------------
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected
