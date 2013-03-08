from sys import *
from math import *
from random import *
from copy import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#==========================================
# SOM for Viviani's curve
#==========================================
#git test2

STATE_SIZE = 3
ACTION_SIZE = 3
INPUT_SIZE = STATE_SIZE+ACTION_SIZE
SOM_DIM = 4
LEARNING_RATE = 0.3
NEIGHBOURHOOD_SIZE = 2
LEARNING_DECAY_RATE = 0.95
NEIGHBOUR_DECAY_RATE = 0.6
PASSES_NUMOF = 100
STEPS_NUMOF = 16
GAP_STEPS_NUMOF = 2
CURVE_STEPS_NUMOF = STEPS_NUMOF + GAP_STEPS_NUMOF
GAP_SHIFT = CURVE_STEPS_NUMOF/4 + GAP_STEPS_NUMOF/2

def get_state_action(step):
	state = []
	a = 4.0
	t = (((step+GAP_SHIFT)%CURVE_STEPS_NUMOF)*4*pi/CURVE_STEPS_NUMOF)-2*pi
	x = a*(1+cos(t))
	state.append(x)
	y = a*sin(t)
	state.append(y)
	z = 2*a*sin(t/2.0)
	state.append(z)

	action = []
	t1 = (((step+1+GAP_SHIFT)%CURVE_STEPS_NUMOF)*4*pi/CURVE_STEPS_NUMOF)-2*pi
	x1 = a*(1+cos(t1))
	action.append(x1-x)
	y1 = a*sin(t1)
	action.append(y1-y)
	z1 = 2*a*sin(t1/2.0)
	action.append(z1-z)

	reward = [0.0]
	if step == STEPS_NUMOF-1:
		reward[0] = 1.0

	return state, action, reward

def find_winner_som(state):
	mindiff = maxint
	minx = -1
	miny = -1
	for x in range(SOM_DIM):
		for y in range(SOM_DIM):
			diff = 0.0
			for i in range(STATE_SIZE):
				diff += abs(som[x][y][i]-state[i])
			if diff < mindiff:
				mindiff = diff
				minx = x
				miny = y
	return minx, miny

def learn_som(winx,winy):
	for n in range(NEIGHBOURHOOD_SIZE+1):
		for m in range(NEIGHBOURHOOD_SIZE+1):
			for i in range(INPUT_SIZE):
				som[(winx+n)%SOM_DIM][(winy+m)%SOM_DIM][i]+=(inpt[i]-som[(winx+n)%SOM_DIM][(winy+m)%SOM_DIM][i])*lr*pow(NEIGHBOUR_DECAY_RATE,n+m)
				som[(winx+n)%SOM_DIM][(winy-m)%SOM_DIM][i]+=(inpt[i]-som[(winx+n)%SOM_DIM][(winy-m)%SOM_DIM][i])*lr*pow(NEIGHBOUR_DECAY_RATE,n+m)
				som[(winx-n)%SOM_DIM][(winy+m)%SOM_DIM][i]+=(inpt[i]-som[(winx-n)%SOM_DIM][(winy+m)%SOM_DIM][i])*lr*pow(NEIGHBOUR_DECAY_RATE,n+m)
				som[(winx-n)%SOM_DIM][(winy-m)%SOM_DIM][i]+=(inpt[i]-som[(winx-n)%SOM_DIM][(winy-m)%SOM_DIM][i])*lr*pow(NEIGHBOUR_DECAY_RATE,n+m)
	wincounts[winx][winy] += 1

# SOM initialisation
print "input size",INPUT_SIZE,"map dimensions",SOM_DIM,"x",SOM_DIM
print "learning rate",LEARNING_RATE,"number of passes",PASSES_NUMOF
som = [[[uniform(0.0,1.0) for i in range(INPUT_SIZE)] for j in range(SOM_DIM)] for k in range(SOM_DIM)]
wincounts = [[0 for j in range(SOM_DIM)] for k in range(SOM_DIM)]

# Plot Viviani's curve and initial weights
#ion()
fig = plt.figure()
ax = Axes3D(fig)
statexs = []
stateys = []
statezs = []
mapxs = []
mapys = []
mapzs = []
for step in range(STEPS_NUMOF):
	state, action, reward = get_state_action(step)
	statexs.append(state[0])
	stateys.append(state[1])
	statezs.append(state[2])
	inpt = state + action
	winx, winy = find_winner_som(inpt)
	mapxs.append(som[winx][winy][0])
	mapys.append(som[winx][winy][1])
	mapzs.append(som[winx][winy][2])
ax.plot(statexs, stateys, statezs, label="Viviani")
#line, = ax.plot(mapxs, mapys, mapzs, label='Discretized')
#draw()
#plt.show()

# SOM Learning loop
lastupdatessum = 0.0
for passnum in range(PASSES_NUMOF):
	if not passnum%10:
		print "Pass",passnum
	somupdatecounts = [[0 for i in range(SOM_DIM)] for j in range(SOM_DIM)]
	somupdatessum = 0.0
	lr = LEARNING_RATE * pow(LEARNING_DECAY_RATE,passnum)
	for s in range(STEPS_NUMOF):
		#step = s
		step = uniform(0,STEPS_NUMOF) 
		state, action, reward = get_state_action(step)

		# Update winner weights
		inpt = state + action
		winx, winy = find_winner_som(inpt)
		learn_som(winx,winy)

	# Plot map
	somerr = 0.0
	statexs = []
	stateys = []
	statezs = []
	mapxs = []
	mapys = []
	mapzs = []
	for step in range(STEPS_NUMOF):
		state, action, reward = get_state_action(step)
		inpt = state + action
		winx, winy = find_winner_som(inpt)
		mapxs.append(som[winx][winy][0])
		mapys.append(som[winx][winy][1])
		mapzs.append(som[winx][winy][2])
		for c in range(INPUT_SIZE):
			somerr += abs(som[winx][winy][c]-inpt[c])
	#line.set_xdata(mapxs)
	#line.set_ydata(mapys)
    #   	line.set_3d_properties(mapzs)
	#draw()
	if not passnum%10:
		print "SOM error",somerr
		engaged = 0
		for n in range(SOM_DIM):
			for m in range(SOM_DIM):
				if wincounts[n][m] > 0:
					engaged += 1
		#print engaged,SOM_DIM*SOM_DIM
#ax.plot(mapxs, mapys, mapzs, label='SOM')
#draw()
#print "SOM"
#for node in som:
#	print node
print "Win counts",wincounts
print "SOM error",somerr

#============================================
# RLSOM
#============================================

#RLSOM_DIMS = [8,4,1]
RLSOM_DIMS = [4,1]
STM_SIZE = 4

# RLSOM initialisation
global activations_som, rlsom, nxtxs, nxtys, activations_rlsoms, wincounts_rlsoms

activations_som = [[0 for i in range(SOM_DIM)] for j in range(SOM_DIM)]
rlsoms = []
nxtxs = []
nxtys = []
activations_rlsoms = []
wincounts_rlsoms = []
for lvl in range(len(RLSOM_DIMS)):
	if lvl == 0:
		rlsoms.append([[[[0 for i in range(SOM_DIM)] for j in range(SOM_DIM)] for k in range(RLSOM_DIMS[0])] for l in range(RLSOM_DIMS[0])])
	else:
		rlsoms.append([[[[0 for i in range(RLSOM_DIMS[lvl-1])] for j in range(RLSOM_DIMS[lvl-1])] for k in range(RLSOM_DIMS[lvl])] for l in range(RLSOM_DIMS[lvl])])
	nxtxs.append(0)
	nxtys.append(0)
	activations_rlsoms.append([[0 for i in range(RLSOM_DIMS[lvl])] for j in range(RLSOM_DIMS[lvl])])
	wincounts_rlsoms.append([[0 for i in range(RLSOM_DIMS[lvl])] for j in range(RLSOM_DIMS[lvl])])

def decay(acts):
	for n in range(len(acts)):
		for m in range(len(acts)):
			if acts[n][m] >= 1: 
				acts[n][m] = acts[n][m]/2

def activate_som(state):
	somx, somy = find_winner_som(state)
	win = som[somx][somy]
	wstate = [win[0],win[1],win[2]]
	train_seq.append(wstate)
	decay(activations_som)
	activations_som[somx][somy] += int(pow(2,STM_SIZE-1))
	actsum = 0
	for n in range(SOM_DIM):
		for m in range(SOM_DIM):
			actsum += activations_som[n][m]
	return actsum

def clear_activation_som():
	for n in range(SOM_DIM):
		for m in range(SOM_DIM):
			activations_som[n][m] = 0

def clear_activation_rlsom(lvl):
	for n in range(RLSOM_DIMS[lvl]):
		for m in range(RLSOM_DIMS[lvl]):
			activations_rlsoms[lvl][n][m] = 0

def find_winner_inpt_rlsom(lvl):
	mindiff = maxint
	minx = -1
	miny = -1
	for x in range(RLSOM_DIMS[lvl]):
		for y in range(RLSOM_DIMS[lvl]):
			diff = 0
			if lvl == 0:
				for n in range(SOM_DIM):
					for m in range(SOM_DIM):
						diff += abs(rlsoms[lvl][x][y][n][m]-activations_som[n][m])
			else:
				for n in range(RLSOM_DIMS[lvl-1]):
					for m in range(RLSOM_DIMS[lvl-1]):
						diff += abs(rlsoms[lvl][x][y][n][m]-activations_rlsoms[lvl-1][n][m])
			#print "new diff",x,y,diff
			if diff < mindiff:
				mindiff = diff
				minx = x
				miny = y
	return minx, miny, mindiff

def learn_rlsom(lvl):
	winx, winy, err = find_winner_inpt_rlsom(lvl)
	wincounts_rlsoms[lvl][winx][winy] += 1
	if err > 0:
		#print "lvl nxtx nxty",lvl,nxtxs[lvl],nxtys[lvl]
		if lvl == 0:
			for n in range(SOM_DIM):
				for m in range(SOM_DIM):
					rlsoms[lvl][nxtxs[lvl]][nxtys[lvl]][n][m]=activations_som[n][m]	
		else:
			for n in range(RLSOM_DIMS[lvl-1]):
				for m in range(RLSOM_DIMS[lvl-1]):
					rlsoms[lvl][nxtxs[lvl]][nxtys[lvl]][n][m]=activations_rlsoms[lvl-1][n][m]	
		#print rlsoms[lvl][nxtxs[lvl]][nxtys[lvl]]	
		nxtys[lvl] = (nxtys[lvl]+1)%RLSOM_DIMS[lvl]
		if nxtys[lvl] == 0:
			nxtxs[lvl] += 1

def activate_rlsom(lvl):
	winx, winy, err = find_winner_inpt_rlsom(lvl)
	#print "act win",winx,winy,err
	decay(activations_rlsoms[lvl])
	activations_rlsoms[lvl][winx][winy] += int(pow(2,STM_SIZE-1))
	actsum = 0
	for n in range(RLSOM_DIMS[lvl]):
		for m in range(RLSOM_DIMS[lvl]):
			actsum += activations_rlsoms[lvl][n][m]
	#print "a",activations_rlsoms[lvl]
	return actsum

#Train RLSOM
vivi_seq = []
train_seq = []
for step in range(STEPS_NUMOF):
	#print "step",step
	state, action, reward = get_state_action(step)
	vivi_seq.append(state)
	a = activate_som(state)
	#print "som activations",activations_som
	for lvl in range(len(RLSOM_DIMS)):
		if a >= pow(2,STM_SIZE)-1:
			learn_rlsom(lvl)
			a = activate_rlsom(lvl)
			if lvl == 0:
				clear_activation_som()
			else:
				clear_activation_rlsom(lvl-1)

def weight_sequence(lvl,nx,ny):
	states = []
	tmpws = deepcopy(rlsoms[lvl][nx][ny])
	for pp in range(STM_SIZE):
		#print lvl,"looking for pp 2^pp",pp,int(pow(2,pp))
		for wx in range(len(rlsoms[lvl][nx][ny])):
			for wy in range(len(rlsoms[lvl][nx][ny])):
				if tmpws[wx][wy] != 0 and tmpws[wx][wy]%int(pow(2,pp+1))!=0:
					#print "lvl",lvl,"found",int(pow(2,pp)),"val",tmpws[wx][wy]
					tmpws[wx][wy] = tmpws[wx][wy]-int(pow(2,pp))
					#print "reduced",tmpws[wx][wy]
					if lvl == 0:
						state = []
						for j in range(INPUT_SIZE):
							state.append(som[wx][wy][j])
						#print state
						states.append(state)
					else:
						ss = weight_sequence(lvl-1,wx,wy)
						for s in range(len(ss)):
							states.append(ss[s])
					break
	return states

def print_states(step,s1,s2,s3,a):
	str1 = ""
	str2 = ""
	str3 = ""
	str4 = ""
	for s in range(len(s1)):
		str1 += "%.2f"%s1[s]+" "
		str2 += "%.2f"%s2[s]+" "
		str3 += "%.2f"%s3[s]+" "
		str4 += "%.2f"%a[s]+" "
	print step,str1,":",str2,":",str3,"-",str4

# Reconstruct weight defined sequence
weight_seq = weight_sequence(1,0,0)
xs = []
ys = []
zs = []
for s in range(len(weight_seq)):
	xs.append(weight_seq[s][0])
	ys.append(weight_seq[s][1])
	zs.append(weight_seq[s][2])	
ax.plot(xs,ys,zs,label="Weights")

# Action-based sequence from initial state at step 0
state, action, reward = get_state_action(0)
act_seq = []
for s in range(len(weight_seq)):
	ns = []
	ns.append(state[0]+weight_seq[s][STATE_SIZE])
	ns.append(state[1]+weight_seq[s][STATE_SIZE+1])
	ns.append(state[2]+weight_seq[s][STATE_SIZE+2])
	act_seq.append(ns)
	state = ns
xs = []
ys = []
zs = []
for s in range(len(act_seq)):
	xs.append(act_seq[s][0])
	ys.append(act_seq[s][1])
	zs.append(act_seq[s][2])	
ax.plot(xs,ys,zs,label="Actions")

# Action-based sequence from winning nodes based on input state match only
state, action, reward = get_state_action(0)
actwin_seq = []
actwin_actseq = []
for s in range(len(weight_seq)):
	actwin_seq.append(state)
	actwin_actseq.append(action)
	ns = []
	ns.append(state[0]+action[0])
	ns.append(state[1]+action[1])
	ns.append(state[2]+action[2])
	somx, somy = find_winner_som(ns)
	action = []
	action.append(som[somx][somy][STATE_SIZE])
	action.append(som[somx][somy][STATE_SIZE+1])
	action.append(som[somx][somy][STATE_SIZE+2])
	state = ns
xs = []
ys = []
zs = []
for s in range(len(actwin_seq)):
	xs.append(actwin_seq[s][0])
	ys.append(actwin_seq[s][1])
	zs.append(actwin_seq[s][2])	
ax.plot(xs,ys,zs,label="Actions-Winners")

#============================================
# Historical match
#============================================

histmatchs = []
histmatchs_som = [[0 for i in range(SOM_DIM)] for j in range(SOM_DIM)]
for lvl in range(len(RLSOM_DIMS)):
	histmatchs.append([[0 for i in range(RLSOM_DIMS[lvl])] for j in range(RLSOM_DIMS[lvl])])

def history_match(lvl):
	if lvl < len(RLSOM_DIMS):
		for x in range(RLSOM_DIMS[lvl]):
			for y in range(RLSOM_DIMS[lvl]):
				weights = deepcopy(rlsoms[lvl][x][y])
				if lvl == 0:
					acts = deepcopy(activations_som)
				else:
					acts = deepcopy(activations_rlsoms[lvl-1])
				for d in range(STM_SIZE-1):
					pwx = -1
					pwy = -1
					match = 0
					for i in range(len(rlsoms[lvl][x][y])):
						for j in range(len(rlsoms[lvl][x][y])):
							weights[i][j] *= 2 # Timeshift!
							if weights[i][j] >= int(pow(2,STM_SIZE)): # Target
								pwx = i
								pwy = j 
								weights[i][j] = weights[i][j] - int(pow(2,STM_SIZE))
							# Check for matches for all powers of 2
							for p in range(STM_SIZE-1):
								sw = int(pow(2,p+1))
								if weights[i][j]%sw!=0 and acts[i][j]%sw!=0:
									match += 1
								else:
									if weights[i][j]%sw!=0 or acts[i][j]%sw!=0:
										match -= 1
								weights[i][j] = weights[i][j]%int(pow(2,p))
								acts[i][j] = acts[i][j]%int(pow(2,p))
					if pwx != -1:
						if lvl == 0:
							histmatchs[lvl][pwx][pwy] = match
						else:
							histmatchs_som[pwx][pwy] = match
					#else:
					#	print "No target found for",x,y,i,j,d
	
# Action-based sequence from winning nodes based on input and historical match 
state, action, reward = get_state_action(0)
hist_seq = []
hist_actseq = []
#for s in range(len(actwin_seq)):
for lvl in range(len(RLSOM_DIMS)):
	print "lvl",lvl
	for row in range(len(rlsoms[lvl])):
		for col in range(len(rlsoms[lvl])):
			print rlsoms[lvl][row][col]
for s in range(2):
	print "step",s
	ns = []
	ns.append(state[0]+action[0])
	ns.append(state[1]+action[1])
	ns.append(state[2]+action[2])
	hist_seq.append(ns)
	hist_actseq.append(action)
	history_match(0)
	print "hist",histmatchs_som
	a = activate_som(state)
	print "acts",activations_som
	for lvl in range(len(RLSOM_DIMS)):
		if a >= pow(2,STM_SIZE)-1:
			a = activate_rlsom(lvl)
			history_match(lvl+1)
			if lvl == 0:
				clear_activation_som()
			else:
				clear_activation_rlsom(lvl-1)
	somx, somy = find_winner_som(ns)
	action = []
	action.append(som[somx][somy][STATE_SIZE])
	action.append(som[somx][somy][STATE_SIZE+1])
	action.append(som[somx][somy][STATE_SIZE+2])
	state = ns
#xs = []
#ys = []
#zs = []
#for s in range(len(hist_seq)):
#	xs.append(hist_seq[s][0])
#	ys.append(hist_seq[s][1])
#	zs.append(hist_seq[s][2])	
#ax.plot(xs,ys,zs,label="History and Input Winners")
	
#print "Viviani/weights/winners - actions"
#for s in range(len(actwin_seq)):
#	print_states(s,vivi_seq[s],weight_seq[s],actwin_seq[s],actwin_actseq[s])
ax.legend()
plt.show()

