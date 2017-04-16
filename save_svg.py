

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import xml.etree.ElementTree

def remove_bounding_box(fid, out_fid='out.svg'):
	''' janky way '''
	f = open(fid)
	lines = f.readlines()
	new_lines = lines[:11]+lines[19:]
	f.close()
	f = open(out_fid, 'w')
	f.writelines(new_lines)
	f.close()

an = np.arange(0,100,0.5)

# draw vertical line from (70,100) to (70, 250)
plt.plot([70, 70], [100, 250], 'k-', lw=2)

# draw diagonal line from (70, 90) to (90, 200)
plt.plot([70, 90], [90, 200], 'k-')

#plt.figure(figsize=[6,6])
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
plt.savefig("test.svg")
