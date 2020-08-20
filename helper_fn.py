#helper fns
import pickle
import pdb
import numpy as np
from screen_conf import *

def round_tup(t, dec=5):
    #decimal = 5
    return tuple(map(lambda x: isinstance(x, float) and round(x, dec) or x, t))
    
def dump_dict(save_name, items, item_name, savepath=None):
    # THIS FN DUMPS ALL THE RECORDED DATA
    params = {}
    for i in range(len(item_name)):
        params[item_name[i]] = items[i]
    # DUMP PICKLE
    if savepath:
        with open(savepath+save_name+'.pkl', 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_name+'.pkl', 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calc_metrics(pts):
	# pts contains RNG POS AND CUR_POS
    # FORMAT: (rng_pos, cur_pos)
    # FORMAT: (TRUEPOS, CUR_POS)
    if 0:
	    rng_pos = [list(y) for y in [x[0] for x in pts]]
	    cur_pos = [list(y) for y in [x[1] for x in pts]]
    else:
        rng_pos = [list(y) for y in pts[0]]
        cur_pos = [list(y) for y in pts[1]]
    if 1:
        # REMOVE 0
        indices = [sum(item) for item in cur_pos if sum(item) == 0]
        for item in reversed(indices):
            rng_pos.pop(item)
            cur_pos.pop(item)
	
    rng_pos = np.array(rng_pos)
    cur_pos = np.array(cur_pos)
	
    ER = np.sqrt(np.sum(np.square(rng_pos-cur_pos), axis=1))
    MAE = np.mean(ER)
    norm_ER = ER/np.max(ER)
    sort_ER = np.sort(ER)
    CEP_ind = int(sort_ER.shape[0]*0.5)
    CEP95_ind = int(sort_ER.shape[0]*0.95)
    if 0: 
        print('MAE = ', MAE)
        print('CEP = ', sort_ER[CEP_ind])
        print('CEP95 = ', sort_ER[CEP95_ind])
    return ER, MAE, sort_ER[CEP_ind], sort_ER[CEP95_ind]

def point_to_screen(eye_center, gaze_vect, fix_d=None):
    '''
    Takes the center of the eye and the gaze vector as input!
    line equations:
    x = x0 + t*a
    y = y0 + t*b
    z = z0 + t*c
    t = (z - z0) / c
    abc = gaze_vect[]
    Returns point on the screen
    z == 0 as we assume its on the same plane with the sensor
    '''
    # -----------------------------------------------
    #pdb.set_trace()
    if fix_d:
    # FIX THE DIST
        t = fix_d / gaze_vect[-1] * (-1)
        # t = 0.5 / gaze_vect[-1] * (-1)
    else:
        t = eye_center[-1] / gaze_vect[-1] * (-1)
    # -----------------------------------------------
    x = eye_center[0] + t * gaze_vect[0]
    y = eye_center[1] + t * gaze_vect[1]
    # -----------------------------------------------
    if 0:
        # top in pixels instead of m
        new_top = top_dist*H_px / H_m
        print('x, y')
        print(x, y)
        p_Y = H_px / H_m * (y - new_top) 
    # -----------------------------------------------
    # The values are in meters
    # we are not taking the camera distortion values into account
    # Camera is in middle of the screen (hence adding half of the screen)
    p_X = W_px - W_px / W_m * (x + W_m/2)
    # The top_dist 
    p_Y = H_px / H_m * (y - top_dist) 
    #p_Y = (y - top_dist) * H_px / H_m
    #p_X = W_px / W_m * (x + W_m/2)
    # -----------------------------------------------
    if 0:
        ppp = point_2d_screen(eye_center, gaze_vect)
        dist = eye_center[-1]
        # Eye center point
        x1 = ppp[0,0][0]
        y1 = ppp[0,0][1]
        # Gaze point
        x2 = ppp[1,0][0]
        y2 = ppp[1,0][1]
        xx_len = x2-x1
        yy_len = y2-y1

        pp_Y = dist * yy_len/xx_len
        www = W_px - (W_px/2 * (W_m/2 - pp_Y))/ (W_m/2)
        hhh = H_px / H_m * (pp_Y - top_dist) * (-1)
        pdb.set_trace()
    # -----------------------------------------------
    # round the numbers
    if 1:
        p_Y = round(p_Y)
        p_X = round(p_X)
    return (p_X, p_Y)
