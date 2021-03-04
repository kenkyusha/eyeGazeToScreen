# drawing utils
import cv2
import time
import numpy as np
import pdb
import matplotlib.pyplot as plt
from screen_conf import *

focus = 0
rng_pos = (0,0)
# cur_pos = (W_px//2, adj_H//2) #pyautogui.position()
#CANV_MODE = 'STABILITY' #'LEFTRIGHT' # 'UPDOWN'
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
	'''(ndarray, 3-tuple, int, int) -> void
	draw gridlines on img
	line_color:
		BGR representation of colour
	thickness:
		line thickness
	type:
		8, 4 or cv2.LINE_AA
	pxstep:
		grid line frequency in pixels
	'''
	x = pxstep
	y = pxstep
	while x < img.shape[1]:
		cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
		x += pxstep

	while y < img.shape[0]:
		cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
		y += pxstep
	return img

def color_grid(img, pos, paint = BLUE, pxstep=50):
	#pdb.set_trace()
	x = pos[0]//pxstep*pxstep
	#y = pos[1]//pxstep*pxstep
	y = (pos[1]-bottom_line//2)//pxstep*pxstep 
	w, h = pxstep, pxstep
	img = cv2.rectangle(img, (x, y), (x + w, y + h), paint, -1)
	return img

def demo_sequence(img):
	'''
	# Test-01
	Demo sequence moving the points from mid of the screen, 
										then to the upper side,
										then to the right most corner,
										and from there to the bottom,
	side of the screen
	'''
	global focus, rng_pos
	focus += 1
	# print(focus)
	if focus < 90:
		rng_pos = (W_px//2, adj_H//2+GRID_STEP)
		#img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	if focus >= 120:
		# CENTER TOP
		rng_pos = (W_px//2, 0+GRID_STEP*2)
	if focus >= 150:
		# RIGHT TOP CORNER
		rng_pos = (W_px-GRID_STEP, 0+GRID_STEP*2)
	if focus >= 180:
		# RIGHT CENTER EDGE
		rng_pos = (W_px-GRID_STEP, adj_H//2+GRID_STEP)
	if focus >= 210:
		# RIGHT BOTTOM CORNER
		rng_pos = (W_px-GRID_STEP, adj_H)
	if focus > 260:
		# RESTART THE SEQUENCE
		focus = 0
	img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	return rng_pos

def demo_updown(img):
	'''
	# Test-02
	Demo sequence moving the points from upper side of the screen 
	to the lower side of the screen
	'''
	global focus, rng_pos
	focus += 1
	# print(focus)
	if focus < 90:
		rng_pos = (W_px//2, adj_H//2+GRID_STEP)
		#img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	if focus >= 90 and focus < 120:
		# CENTER TOP
		rng_pos = (W_px//2, adj_H//2-GRID_STEP)
	if focus >= 120 and focus < 150:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2-GRID_STEP*3)
	if focus >= 150 and focus < 180:
		# RIGHT CENTER EDGE
		rng_pos = (W_px//2, 0+GRID_STEP*2)
	if focus >= 180 and focus < 210:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2-GRID_STEP*3)
	if focus >= 210 and focus < 240:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2-GRID_STEP)
	if focus >= 240 and focus < 270:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2+GRID_STEP*2)
	if focus >= 270 and focus < 300:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2+GRID_STEP*4)
	if focus >= 300 and focus < 330:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2, adj_H//2+GRID_STEP*6)
	if focus > 330:
		# RIGHT BOTTOM CORNER
		rng_pos = (W_px//2, adj_H)
	if focus > 380:
		# RESTART THE SEQUENCE
		focus = 0
	img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	return rng_pos

def demo_leftright(img):
	'''
	Test-03
	Demo sequence moving the points from left side of the screen 
	to the right side of the screen
	'''
	global focus, rng_pos
	focus += 1
	# print(focus)
	if focus < 90:
		rng_pos = (W_px//2, adj_H//2+GRID_STEP)
		#img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	if focus >= 90 and focus < 120:
		# CENTER TOP
		rng_pos = (W_px//2+GRID_STEP*3, adj_H//2+GRID_STEP)
	if focus >= 120 and focus < 150:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2+GRID_STEP*6, adj_H//2+GRID_STEP)
	if focus >= 150 and focus < 180:
		# RIGHT CENTER EDGE
		rng_pos = (W_px//2+GRID_STEP*9, adj_H//2+GRID_STEP)
	if focus >= 180 and focus < 210:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2+GRID_STEP*12, adj_H//2+GRID_STEP)
	if focus >= 210 and focus < 240:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2+GRID_STEP*9, adj_H//2+GRID_STEP)
	if focus >= 240 and focus < 270:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2+GRID_STEP*6, adj_H//2+GRID_STEP)
	if focus >= 270 and focus < 300:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2+GRID_STEP*3, adj_H//2+GRID_STEP)
	if focus >= 300 and focus < 330:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2-GRID_STEP, adj_H//2+GRID_STEP)
	if focus >= 330 and focus < 370:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2-GRID_STEP*3, adj_H//2+GRID_STEP)
	if focus >= 370 and focus < 400:
		# RIGHT TOP CORNER
		rng_pos = (W_px//2-GRID_STEP*6, adj_H//2+GRID_STEP)
	if focus >= 400 and focus < 430:
		# RIGHT BOTTOM CORNER
		rng_pos = (W_px//2-GRID_STEP*9, adj_H//2+GRID_STEP)
	if focus > 430:
		rng_pos = (W_px//2-GRID_STEP*12, adj_H//2+GRID_STEP)
	if focus > 500:
		# RESTART THE SEQUENCE
		focus = 0
	img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	return rng_pos

def demo_stability(img):
	'''
	# Test-04, Test-05, Test-06
	# first case is to move towards the camera while focusing at the same point (20cm)
	# second case is moving away from the camera (20cm)
	# third case indian nodding
	'''
	global focus, rng_pos
	#focus += 1
	#print(focus)
	# FOCUS POINT - currently center:
	rng_pos = (W_px//2, adj_H//2+GRID_STEP)
	img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	return rng_pos

def random_sequence(img):
# def random_sequence(img, cur_pos):
	'''
	# Random Sequence - display random spot on the screen which the person can try to aim at!
	'''
	global focus, rng_pos
	if focus <= 80: # 100 - 4 seonds
		# KEEP OLD POSITION
		focus += 1
	else:
		# generate new ranom spot
		rng_pos = (np.random.randint(0, W_px),np.random.randint(0, H_px))
		#img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
		focus = 0
	return rng_pos



def plot_pts(pts, name, MAE = None, save_path = None):
	rng_pos = np.array([list(y) for y in [x[0] for x in pts]])
	cur_pos = np.array([list(y) for y in [x[1] for x in pts]])
	
	str_name = name #.split('/')[1].split('.')[0]
	if MAE:
		plt.title('Results of {}, MAE: '.format(str_name, MAE))
	else:
		plt.title('Results of {}'.format(str_name))
	plt.plot(rng_pos[:,0], rng_pos[:,1], 'r.', markersize = 20, label = 'target point')
	plt.plot(cur_pos[:,0], cur_pos[:,1], 'bo', label = 'gaze point', mfc='none')
	plt.gca().invert_yaxis()
	plt.grid()
	plt.legend()
	if save_path:
		plt.savefig(save_path+str_name+'.png')
	else:
		plt.savefig(str_name+'.png')
	plt.close()

def accuracy_measure(pts):
	'''
	if sum(cur_pos) == 0:
		return
	else:
		pdb.set_trace()
	'''
	# input is all the points
	# separate the rng_pos and cur_pos
	# remove values at indexes where cur_pos == 0
	rng_pos = [x[0] for x in pts] # TUPLES
	cur_pos = [x[1] for x in pts] # TUPS
	# TUPS TO LIST:
	rng_pos = [list(elem) for elem in rng_pos]
	cur_pos = [list(elem) for elem in cur_pos]
	# INDEX OF 0 cur pos:
	if 0:
		indices = [sum(item) for item in cur_pos if sum(item) == 0]

	pdb.set_trace()

def display_canv(CANV_MODE, cur_pos=None):
	# THIS FN returns RNG_POS and CUR_POS as TUPLES
	# RETURN FORMAT: (TRUE_POS, CUR_POS) ..  RNG == TRUE
	global focus, rng_pos
	img = np.zeros((adj_H, W_px,3))

	img = draw_grid(img, pxstep= GRID_STEP)
	
	if CANV_MODE == 'RNG':
		#cv2.putText(img, str_pos, (cur_pos[0]+5, cur_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
		#rng_pos = random_sequence(img, cur_pos)
		rng_pos = random_sequence(img)
	if CANV_MODE == 'SEQ':
		rng_pos = demo_sequence(img)
	if CANV_MODE == 'UPDOWN':
		rng_pos = demo_updown(img)
	if CANV_MODE == 'LEFTRIGHT':
		rng_pos = demo_leftright(img)
	if CANV_MODE == 'STABILITY':
		rng_pos = demo_stability(img)

	img = color_grid(img, rng_pos, paint = RED, pxstep=GRID_STEP)
	if cur_pos:
		#cur_pos = mid_point
		# accuracy_measure(cur_pos, rng_pos)
		str_pos = str(cur_pos)
		img = color_grid(img, cur_pos, paint=BLUE, pxstep=GRID_STEP)
		xy_cur = (cur_pos[0]//GRID_STEP*GRID_STEP,(cur_pos[1]-bottom_line//2)//GRID_STEP*GRID_STEP )
		xy_rng = (rng_pos[0]//GRID_STEP*GRID_STEP,(rng_pos[1]-bottom_line//2)//GRID_STEP*GRID_STEP )
		# IF RANDOM SPOT EQUALS THE ESTIMATED FOCUS SPOT COLOR IT GREEN!
		if xy_cur == xy_rng:
			img = color_grid(img, rng_pos, paint = GREEN, pxstep=GRID_STEP)
	else:
		cur_pos = (0,0)

	# SHOW IMG
	cv2.imshow('black_canv', img)
	cv2.moveWindow("black_canv", 0,0)
	return (rng_pos, cur_pos)

def plot_eye_XYZ(pts, name, savepath):

	X_arr = np.array([x[0] for x in pts])
	Y_arr = np.array([x[1] for x in pts])
	Z_arr = np.array([x[2] for x in pts])

	fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10,8))
	fig.suptitle('Scenario: ' + name)

	ax1.plot(X_arr, label = 'X coords')
	ax2.plot(Y_arr, label = 'Y coords')
	ax3.plot(Z_arr, label = 'Z coords')
	plt.legend()
	ax1.grid()
	ax2.grid()
	ax3.grid()
	# plt.show()
	if savepath:
		plt.savefig(savepath+name+'.pdf')
	else:
		plt.savefig(name+'.pdf')
	plt.close()
	
