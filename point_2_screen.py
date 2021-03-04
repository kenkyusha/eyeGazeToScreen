#!/usr/bin/env python

from typing import Optional

import datetime
import logging
import pathlib

import cv2
import numpy as np
import yacs.config

from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from gaze_estimation.utils import load_config
from gaze_estimation import GazeEstimationMethod, GazeEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pdb
import pickle
import time
import imutils
import sys
import os

import draw_utils
from helper_fn import point_to_screen, dump_dict, calc_metrics, round_tup
from screen_conf import *
# FLAGS PARAMETERS
#------------------------------------------------
fpath = 'recs/'
rgb_fp = 'det/'

# AVERAGING OVER GAZE VALUES TOGGLE
#------------------------------------------------
GAZE_AVG_FLAG = 0
num_frames = 3 # num of frames to average over
#------------------------------------------------
# AVERAGING OVER LANDMARKS TOGGLE
AVG_LANDMARKS = 0
num_avg_frames = 3 # num of frames to average over


# GLOBAL VARIABLES
#------------------------------------------------
img = np.zeros((adj_H, W_px,3))
mid_point = (0,0)
rng_pos = (np.random.randint(0, W_px),np.random.randint(0, H_px))
focus = 0
avg_pos = []
#------------------------------------------------
DEBUG = 0 #'EYE' #  'EYE' DEBUG INDIVIDUAL VALUES
if DEBUG:
    try:
        print('Creating dirs')
        os.mkdir(fpath)
        os.mkdirs(fpath+rgb_fp)
    except:
        print('dirs already exist')

#------------------------------------------------
class Demo:
    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):
        self.config = config
        self.gaze_estimator = GazeEstimator(config, AVG_LANDMARKS=AVG_LANDMARKS, num_frames=num_avg_frames)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        # Turn writer on and off.
        if SAVE_VIDEO:
            self.writer = self._create_video_writer()

        else:
            self.writer = 0

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = NORM_EYEZ # self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        # FRAME COUNTER
        self.i = 0
        self.pts = []
        self.cur_pos = []
        self.true_pos = []
        self.dist = []

        self.left_eye_cent = []
        self.right_eye_cent = []
        self.right_eye_gaze = []
        self.left_eye_gaze = []
        self.face_gaze = []
        self.face_cent = []

    def run(self) -> None:
        while True:

            if DEMO:
                pts = draw_utils.display_canv(CANV_MODE=CANV_MODE, cur_pos=mid_point) #cur_pos=cur_pos
                self.pts.append(pts)
                self.true_pos.append(pts[0])
                self.cur_pos.append(pts[1])
            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()
            if not ok:
                break

            if CUST_VIDEO:
                frame = imutils.resize(frame, width=self.gaze_estimator.camera.width, height=self.gaze_estimator.camera.height)

            calib_time = time.time()
            # FIRST WE UNDISTORT THE IMAGE!
            undistorted = cv2.undistort(
                frame, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)
            if RUNTIME:
                print('Image calibration: ', time.time()-calib_time, ' seconds.')

            self.visualizer.set_image(frame.copy())

            dlib_time = time.time()
            faces = self.gaze_estimator.detect_faces(undistorted)
            if RUNTIME:
                print('DLIB faces: ', time.time() - dlib_time, ' seconds.')

            for face in faces:
                self.gaze_estimator.estimate_gaze(undistorted, face)
                self._draw_face_bbox(face)
                self._draw_head_pose(face)
                self._draw_landmarks(face)
                self._draw_face_template_model(face)
                self._draw_gaze_vector(face)
                self._display_normalized_image(face)

            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            if self.writer:
                self.writer.write(self.visualizer.image)
                #self.write_eyes.write(self.visualizer.image)
            if self.config.demo.display_on_screen:
                self.visualizer.image = cv2.resize(self.visualizer.image, (0, 0), fy=IMG_SCALE, fx=IMG_SCALE)
                cv2.imshow('frame', self.visualizer.image)
                # MOVE TO TOP LEFT CORNER
                cv2.moveWindow("frame", 0,0)
                if DEBUG:
                    cv2.imwrite(fpath+rgb_fp+'rgb_'+str(self.i).zfill(5)+'.png', self.visualizer.image)
            # INCREMENT COUNTER
            self.i += 1
        self.cap.release()
        if self.writer:
            self.writer.release()

    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            # use recording or the custom video
            if CUST_VIDEO:
                cap = cv2.VideoCapture(vid_file)
            else:
                cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        # pdb.set_trace()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, FPS,
                                 (VID_W,
                                  VID_H))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> None:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
                    f'roll: {roll:.2f}, distance: {face.distance:.2f}')

        self.dist.append(face.distance)

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else:
            raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        normalized = cv2.resize(normalized, (0, 0), fy=5, fx=5)
        if PRINT_VALS:
            H, W = normalized.shape
            left_edge = W - 50
            left_edge_H = 20
            cv2.putText(normalized, 
                        str(self.i),  #'cur frame = '
                        (left_edge, left_edge_H), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 1)
            save_str = 'norm_eyes_fix/img_'+str(self.i).zfill(5)+'.png'
            if NORM_EYEZ:
                cv2.imwrite(save_str, normalized[:,300:])

        cv2.imshow('normalized', normalized)

    def avg_frames(self):
        if 0:
            r_avg_cent = [np.array([x[0] for x in self.right_eye_cent[-num_frames:]]).mean(),
                        np.array([x[1] for x in self.right_eye_cent[-num_frames:]]).mean(),
                        np.array([x[2] for x in self.right_eye_cent[-num_frames:]]).mean()]
            l_avg_cent = [np.array([x[0] for x in self.left_eye_cent[-num_frames:]]).mean(),
                        np.array([x[1] for x in self.left_eye_cent[-num_frames:]]).mean(),
                        np.array([x[2] for x in self.left_eye_cent[-num_frames:]]).mean()]
        else:
            r_avg_cent = self.right_eye_cent[-1]
            l_avg_cent = self.left_eye_cent[-1]
        r_avg_gaze = [np.array([x[0] for x in self.right_eye_gaze[-num_frames:]]).mean(),
                    np.array([x[1] for x in self.right_eye_gaze[-num_frames:]]).mean(),
                    np.array([x[2] for x in self.right_eye_gaze[-num_frames:]]).mean()]
        l_avg_gaze = [np.array([x[0] for x in self.left_eye_gaze[-num_frames:]]).mean(),
                    np.array([x[1] for x in self.left_eye_gaze[-num_frames:]]).mean(),
                    np.array([x[2] for x in self.left_eye_gaze[-num_frames:]]).mean()]
        
        right_eye_XY = point_to_screen(r_avg_cent, r_avg_gaze)
        left_eye_XY = point_to_screen(l_avg_cent, l_avg_gaze)
        mid_x = np.mean([right_eye_XY[0], left_eye_XY[0]])
        mid_y = np.mean([right_eye_XY[1], left_eye_XY[1]])

        if PRINT_VALS:
            self.draw_vals(r_avg_gaze, r_avg_cent, l_avg_gaze,l_avg_cent)
        return mid_x, mid_y

    def draw_vals(self, r_gaze, r_cent, l_gaze, l_cent):
        H, W, _ = self.visualizer.image.shape
        left_edge = W - 350
        left_edge_H = 40
        flip_img = cv2.flip(self.visualizer.image, 1)
        r_gaze = round_tup(r_gaze)
        r_cent = round_tup(r_cent)
        l_gaze = round_tup(l_gaze)
        l_cent = round_tup(l_cent)
        print('frame no ', self.i)
        print('right_gaze, ', r_gaze)
        print('left_gaze , ', l_gaze)
        print('right_cent, ', r_cent)
        print('left_cent , ', l_cent)

        cv2.putText(flip_img, 
                    'cur frame = '+ str(self.i), 
                    (left_edge, left_edge_H-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 1)
        cv2.putText(flip_img, 
                    'R_Gaze = '+str(r_gaze), 
                    (left_edge, left_edge_H), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        cv2.putText(flip_img, 
                    'R_Cent = '+str(r_cent), 
                    (left_edge, left_edge_H+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        cv2.putText(flip_img, 
                    'L_Gaze = '+str(l_gaze), 
                    (left_edge, left_edge_H+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        cv2.putText(flip_img, 
                    'L_Cent = '+str(l_cent), 
                    (left_edge, left_edge_H+60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        if GAZE_AVG_FLAG:
            avg_str = 'ON' + ' frames = ' + str(num_frames)
        else:
            avg_str = 'OFF'
        cv2.putText(flip_img, 
                    'AVG = ' + str(avg_str), 
                    (left_edge, left_edge_H+85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 1)

        self.visualizer.image = cv2.flip(flip_img, 1)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        print('*'*50)
        right_eye_XY = (0,0)
        left_eye_XY = (0,0)
        r_gaze_ = (0,0,0)
        r_cent_ = (0,0,0)
        l_gaze_ = (0,0,0)
        l_cent_ = (0,0,0)

        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                
                if key.name.lower() == 'reye':
                    self.right_eye_cent.append(eye.center)
                    self.right_eye_gaze.append(eye.gaze_vector)
                    r_gaze_ = tuple(eye.gaze_vector)
                    r_cent_ = tuple(eye.center)
                    right_eye_XY = point_to_screen(eye.center, eye.gaze_vector)
                else:
                    self.left_eye_cent.append(eye.center)
                    self.left_eye_gaze.append(eye.gaze_vector)
                    left_eye_XY = point_to_screen(eye.center, eye.gaze_vector)
                    l_gaze_ = tuple(eye.gaze_vector)
                    l_cent_ = tuple(eye.center)

                print('{} gaze = '.format(key.name.lower()), eye.gaze_vector)
                print('{} center = '.format(key.name.lower()), eye.center)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            self.face_cent.append(face.center)
            self.face_gaze.append(face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

        global mid_point
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            # -----------------------------------------------
            if GAZE_AVG_FLAG:
                if len(self.right_eye_cent) >= num_frames:
                    mid_x, mid_y = self.avg_frames()
                else:
                    if PRINT_VALS:
                        self.draw_vals(r_gaze_, r_cent_, l_gaze_,l_cent_)
            else:
                mid_x = np.mean([right_eye_XY[0], left_eye_XY[0]])
                mid_y = np.mean([right_eye_XY[1], left_eye_XY[1]])
                if PRINT_VALS:
                    self.draw_vals(r_gaze_, r_cent_, l_gaze_,l_cent_)
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            XY = point_to_screen(face.center, face.gaze_vector)
            mid_x = XY[0]
            mid_y = XY[1]
            
        else:
            raise ValueError

        mid_point = (int(mid_x), int(mid_y))

def main():
    '''
    # EYE MODEL
    python demo.py --config configs/demo_mpiigaze_resnet.yaml
    # FACE MODEL
    python demo.py --config configs/demo_mpiifacegaze_resnet_simple_14.yaml
    '''
    global DEMO, CANV_MODE, IMG_SCALE, NORM_EYEZ, SAVE_VIDEO
    global RUNTIME, CUST_VIDEO, vid_file, PRINT_VALS
    start_time = time.time()
    config, custom = load_config()
    # pdb.set_trace()
    DEMO = custom['demo']
    # Save normalized eyes
    NORM_EYEZ = custom['eyes']
    # FLAG TO SAVE MOVE, DEFAULT = FALSE
    SAVE_VIDEO = custom['save_vid']
    # PRINT RUNTIME
    RUNTIME = custom['runtime'] #0
    # PRINTS VALS ON THE WEBCAM IMG
    PRINT_VALS = custom['printvals'] #0
    # CUSTOM VIDEO:
    CUST_VIDEO = custom['cust_vid']
    if CUST_VIDEO != None:
        vid_file = CUST_VIDEO
        CANV_MODE = custom['mode']
        if CANV_MODE == 'STABILITY' or CANV_MODE == 'UPDOWN' \
        or CANV_MODE == 'LEFTRIGHT' or CANV_MODE == 'SEQ':
            print('Current mode is {}'.format(CANV_MODE))
        else:
            print('Breaking since current mode is {}'.format(CANV_MODE))
            print('Set correct CANV_MODE --mode: ')
            print('*STABILITY* *UPDOWN* *LEFTRIGHT* *SEQ*')
            sys.exit(1)
    if DEMO:
        IMG_SCALE = custom['imgscale']
        CANV_MODE = custom['mode'] #'RNG'
    demo = Demo(config)
    demo.run()
    n_frames = len(demo.pts)
    tot_time = time.time()-start_time
    print('nr of frames: ', n_frames)
    print('All finished: ',tot_time , ' seconds.')
    print('FPS: ', round(n_frames/tot_time,2))
    # This part only gets executed in case there is input to the model

    if CUST_VIDEO:
        # COMPUTE ACCURACY METRICS HERE
        save_path = 'testResults/'
        try:
            os.mkdir(save_path)
        except:
            print('folder already existing {}'.format(save_path))
        str_name = vid_file.split('/')[1].split('.')[0] + '_LM_' +str(AVG_LANDMARKS) + '_GAZE_' + str(GAZE_AVG_FLAG)
        str_name = str(demo.gaze_estimator.camera.width) + 'x' + str(demo.gaze_estimator.camera.height) + '_' + str_name
        str_name = config.mode + str_name
        indices = [sum(item) for item in demo.cur_pos if sum(item) == 0]
        for item in reversed(indices):
            demo.true_pos.pop(item)
            demo.cur_pos.pop(item)
        # DUMP THE GAZE AND CENTER VALUES
        if config.mode == 'MPIIGaze':
            dump_dict(str_name,items=[demo.left_eye_cent,demo.left_eye_gaze, demo.right_eye_cent, demo.right_eye_gaze, demo.true_pos, demo.dist],
                  item_name = ['lcent', 'lgaze', 'rcent', 'rgaze', 'tpos', 'fdist'])
        elif config.mode == 'MPIIFaceGaze':
            dump_dict(str_name,items=[demo.face_cent,demo.face_gaze, demo.true_pos, demo.dist],
                  item_name = ['fcent', 'fgaze', 'tpos', 'fdist'])
        print('EXTI BEFORE METRICS & PLOTS')
        _, MAE, CEP, CE95 = calc_metrics((demo.true_pos,demo.cur_pos))
        print('MAE = ', MAE)
        print('CEP = ', CEP)
        print('CEP95 = ', CE95) 
        # draw results
        draw_utils.plot_pts((demo.true_pos,demo.cur_pos), str_name, MAE, save_path) 


if __name__ == '__main__':
    main()
