import signal
import cv2
import cv2.cv as cv
import numpy
from SimpleCV import Camera, Image, Segmentation
from SimpleCV.Segmentation.RunningSegmentation import RunningSegmentation
from scipy.cluster.vq import kmeans, vq
from scipy import ndimage
from numpy import reshape, uint8, flipud


class Object:
	featureVector = None
	
class Feature:
	def __init__(self, keypoints, descriptors):
		self.keypoints = keypoints
		self.descriptors = descriptors
	
class Database:
	def __init(self):
		pass

def reduceVal(val):
	#used for colour space quantization to 27 colours
	if val < 64:
		return 0
	if val < 128:
		return 64
	return 255
def colourQuantization(image):
	red = image[:,:,2]
	green = image[:,:,1]
	blue = image[:,:,0]

	for j in range(480):
		for i in range(640):
			image[:,:,2][i,j] = reduceVal(red[i,j])
			image[:,:,1][i,j] = reduceVal(green[i,j])
			image[:,:,0][i,j] = reduceVal(blue[i,j])
	return image
	
def segmentation(image):
	"""Executes image segmentation based on various features of the video stream"""
	edgeThreshold = 1
	lowThreshold = 0
	max_lowThreshold = 100 #use a function later
	ratio = 3
	kernel_size = 3
	gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
	blurred = cv2.GaussianBlur(gray,(5,5),0)

	ret, bw = cv2.threshold(blurred, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#bw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,0)
	erosion = cv2.erode(bw,disk(3),iterations = 1)
	dilation = cv2.dilate(erosion,disk(3),iterations=1)
	
	
	#quantized = equalize(image, diamond(5))
	edges = cv2.Canny(blurred, lowThreshold, max_lowThreshold)

	dilateEdges = cv2.dilate(edges,disk(4),iterations=1)
	erodeEdges = cv2.erode(dilateEdges,disk(4), iterations=1)
	area = erosion | dilateEdges
	#area = cv2.erode(area,disk(3), iterations=1)
	#area = cv2.dilate(area, disk(3), iterations=1)
	area, labels = ndimage.label(area)
	area = area*50
	return area

def featureExtractor(segmented):
	"""Extracts features from segmented image(s)"""
	return 0;

def matchFinder(features):
	"""Matches object features against database"""
	#

def addToDatabase(object):
	#
	pass
	
def getImage(cam):
	img = Image("lenna")
	try:
		img = cam.getImage()
	except:
		pass
	return img

def main():
	"""Main execution of the program"""
	#initializing camera
	#cam = Camera()
	camera = cv2.VideoCapture(0)
	i = 0
	while 1:
		ret, frame = camera.read()

		#io.imsave("%i%s" % (i,'.jpg'), seg)
		#img = RunningSegmentation.addImage(img)
		#seg = img.RunningSegmentation.getSegmentedImage()
		#seg.getRawImage()
		#seg.save(str(i))
		segmented = segmentation(frame)
		
		cv2.imwrite("%i%s" % (i,'.jpg'), segmented)
		print 'saving image', i
		#features = featureExtractor(segmented)
		#print 'features: ', features
		i+=1

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print 'Interrupted, proceeding'
