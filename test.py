import signal
from SimpleCV import Camera, Image, Segmentation
from SimpleCV.Segmentation import RunningSegmentation
from skimage import data, draw,color, morphology, transform, feature, io, filter
from skimage import data
from skimage.filter import threshold_adaptive, threshold_yen, threshold_otsu
from skimage.feature import match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF

#cam = Camera()
#img = cam.getImage()
#img = img.getPIL()

class Object:
	featureVector = None
	
class Feature:
	def __init__(self, keypoints, descriptors):
		self.keypoints = keypoints
		self.descriptors = descriptors
	
class Database:
	def __init(self):
		pass

def segmentation(image):
	"""Executes image segmentation based on various features of the video stream"""
	
	bw = color.rgb2grey(image)
	#binary = threshold_adaptive(bw,500, method='mean', mode='nearest')
	threshold = threshold_yen(bw)
	binary = bw < threshold
	return binary

def featureExtractor(segmented):
	"""Extracts features from segmented image(s)"""
	keypoints = corner_peaks(corner_harris(segmented), min_distance=5)
	extractor = BRIEF()
	extractor.extract(segmented, keypoints)
	return Feature(keypoints[extractor.mask], extractor.descriptors)

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
	cam = Camera()
	i = 0
	while 1:
		
		img = getImage(cam)
		PIL = img.getNumpy() #Numpy required for some functions, flips the axes. PIL orientation would be correct
		seg = segmentation(PIL)
		feature = featureExtractor(seg)

		io.imsave("%i%s" % (i,'.jpg'), seg)
		#img = RunningSegmentation.addImage(img)
		#seg = img.RunningSegmentation.getSegmentedImage()
		#seg.getRawImage()
		#seg.save(str(i))
		i+=1

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print 'Interrupted, proceeding'
