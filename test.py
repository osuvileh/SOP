import cv2
import cv2.cv as cv
import numpy
from numpy import reshape, uint8, flipud
from scipy import ndimage
from scipy.cluster.vq import kmeans, vq
from skimage.morphology import disk, diamond
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.segmentation import clear_border


class Object:
	featureVector = None
	
class Feature:
	def __init__(self, keypoints, descriptors, name):
		self.keypoints = keypoints
		self.descriptors = descriptors
		self.name = name
		
class Match:
	def __init__(self, dbFeature, feature, keypointPairs):
		self.dbFeature = dbFeature
		self.feature = feature
		self.keypointPairs = keypointPairs
		self._calculateBoundingBox()
		
	def _calculateBoundingBox(self):
		minX = 9999
		minY = 9999
		maxX = -9999
		maxY = -9999
		for pair in self.keypointPairs:
			if minX > pair[1].pt[0]:
				minX = pair[1].pt[0]
			if minY > pair[1].pt[1]:
				minY = pair[1].pt[1]
			
			if maxX < pair[1].pt[0]:
				maxX = pair[1].pt[0]
			if maxY < pair[1].pt[1]:
				maxY = pair[1].pt[1]
				
		self.min = (int(minX), int(minY))
		self.max = (int(maxX), int(maxY))
	
class Database:
	def __init(self):
		pass
	
def segmentation(image):
	"""Executes image segmentation based on various features of the video stream"""
	edgeThreshold = 1
	lowThreshold = 0
	max_lowThreshold = 100 #use a function later
	ratio = 3
	kernel_size = 3
	gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	ret, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#bw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,0)
	
	result = cv2.dilate(bw, square(10), iterations = 1)
	result = cv2.erode(result, square(10), iterations = 1)
	
	result = cv2.erode(result, square(10), iterations = 1)
	result = cv2.dilate(result, square(10), iterations = 1)
	
	
	#quantized = equalize(image, diamond(5))
	#edges = cv2.Canny(image, lowThreshold, max_lowThreshold)

	#dilateEdges = cv2.dilate(edges,disk(4),iterations=1)
	#erodeEdges = cv2.erode(dilateEdges,disk(3), iterations=1)
	#area = erosion | erodeEdges
	#area = cv2.erode(area,disk(3), iterations=1)
	#area = cv2.dilate(area, disk(3), iterations=1)
	#area, labels = ndimage.label(area)
	#area = area*50
	return label(result) * 50
	
def extractSegments(image, segmented):
	segments = []
	values = numpy.unique(segmented)
	gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
	for value in values:
		segment = gray.copy()
		segment[segmented != value] = 0
		segments.append(segment)
	return segments;

def featureExtractor(detector, extractor, segments, frameNumber):
	"""Extracts features from segmented image(s)"""
	features = []
	for i, segment in enumerate(segments):
		ret, mask = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY)
		keypoints = detector.detect(segment, mask)
		keypoints, descriptors = extractor.compute(segment, keypoints, mask)
		features.append(Feature(keypoints, descriptors, str(frameNumber) + str(i)))
	return features;

def matchFinder(features):
	"""Matches object features against database"""
	#

def addToDatabase(object):
	#
	pass
	
def filterMatches(kp1, kp2, matches, ratio = 0.6):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

def main():
	"""Main execution of the program"""
	database = []
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
	detector = cv2.FeatureDetector_create("ORB")
	extractor = cv2.DescriptorExtractor_create("ORB")
	camera = cv2.VideoCapture("test2.mp4")
	frameNumber = 0
	
	colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
	while 1:
		ret, frame = camera.read()
		
		segmented = segmentation(frame)
		print "saving labeled segmentation"
		cv2.imwrite("%i%s" % (frameNumber, 'labels.jpg'), segmented)
		segments = extractSegments(frame, segmented)
		features = featureExtractor(detector, extractor, segments, frameNumber)
		
		featureMatches = []
		for a, feature in enumerate(features):
			isMatch = False
			for b, data in enumerate(database):
				if (data.descriptors != None and feature.descriptors != None):
					matches = matcher.knnMatch(data.descriptors, feature.descriptors, k = 2)
					pairs = filterMatches(data.keypoints, feature.keypoints, matches)
					if len(pairs) >= 7:
						featureMatches.append(Match(data, feature, pairs))
						isMatch = True
			if isMatch == False:
				database.append(feature)
		
		colorIndex = 0
		for match in featureMatches:
			cv2.rectangle(frame, match.min, match.max, colors[colorIndex % len(colors)], 2)
			cv2.putText(frame, match.dbFeature.name, match.min, cv2.FONT_HERSHEY_PLAIN, 2, colors[colorIndex % len(colors)], 2)
			for pair in match.keypointPairs:
				cv2.line(frame, (int(pair[0].pt[0]), int(pair[0].pt[1])),(int(pair[1].pt[0]), int(pair[1].pt[1])), colors[colorIndex % len(colors)], 1)
			colorIndex += 1
		
		cv2.imwrite("%i%s" % (frameNumber, '.jpg'), frame)
		print 'saving image', frameNumber
		
		for i, segment in enumerate(segments):
			cv2.imwrite("%i%s%i%s" % (frameNumber, '_seg', i, '.jpg'), segment)
		
		frameNumber += 1

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print 'Interrupted, proceeding'
