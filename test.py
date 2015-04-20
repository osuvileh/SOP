import cv2
import cv2.cv as cv
import numpy
from skimage.measure import label
from skimage.morphology import square

class Object:
	def __init__(self, name):
		self.features = []
		self.name = name
	
class Feature:
	def __init__(self, keypoints, descriptors, bounds):
		self.keypoints = keypoints
		self.descriptors = descriptors
		self.bounds = bounds
		
class Match:
	def __init__(self, object, feature, keypointPairs):
		self.object = object
		self.feature = feature
		self.keypointPairs = keypointPairs
	
class Database:
	def __init(self):
		pass
	
def segmentation(image):
	"""Executes image segmentation based on various features of the video stream"""
	gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	ret, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	
	# Close binary image
	result = cv2.dilate(bw, square(10), iterations = 1)
	result = cv2.erode(result, square(10), iterations = 1)
	
	# Open binary image
	result = cv2.erode(result, square(10), iterations = 1)
	result = cv2.dilate(result, square(10), iterations = 1)
	
	return label(result) * 50
	
def extractSegments(image, segmented):
	"""Extracts segments from labeled image"""
	segments = []
	bounds = []
	values = numpy.unique(segmented)
	gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
	for value in values:
		segment = gray.copy()
		segment[segmented != value] = 0
		_, thresh = cv2.threshold(segment, 1, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		x, y, w, h = cv2.boundingRect(contours[0])
		imgHeight, imgWidth = segment.shape[:2]
		if w < imgWidth * 0.98 and h < imgHeight * 0.98:
			segments.append(segment[y:y+h,x:x+w])
			bounds.append([x, y, x+w, y+h])
	return segments, bounds;

def featureExtractor(detector, extractor, segments, bounds):
	"""Extracts features from segmented image(s)"""
	features = []
	for i, segment in enumerate(segments):
		ret, mask = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY)
		keypoints = detector.detect(segment, mask)
		keypoints, descriptors = extractor.compute(segment, keypoints, mask)
		features.append(Feature(keypoints, descriptors, bounds[i]))
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
    kpPairs = zip(mkp1, mkp2)
    return kpPairs

def main():
	"""Main execution of the program"""
	objects = []
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
	detector = cv2.FeatureDetector_create("ORB")
	extractor = cv2.DescriptorExtractor_create("ORB")
	camera = cv2.VideoCapture("test2.mp4")
	frameNumber = 0
	
	while 1:
		ret, frame = camera.read()
		
		segmented = segmentation(frame)
		cv2.imwrite("%i%s" % (frameNumber, 'labels.jpg'), segmented)
		segments, bounds = extractSegments(frame, segmented)
		features = featureExtractor(detector, extractor, segments, bounds)
		
		# Iterate through each feature found in the frame
		featureMatches = []
		for a, feature in enumerate(features):
			isKnownObject = False
			b = 0
			
			# Iterate through every known object
			while b < len(objects):
				object = objects[b]
				
				# To limit processing power needed only n newest occurences of an object are kept
				if len(object.features) > 10:
					object.features = object.features[1:]
				isSameObject = False
				
				# Iterate through each occurence of the object
				for c, data in enumerate(object.features):
					if (data.descriptors != None and feature.descriptors != None):
						matches = matcher.knnMatch(data.descriptors, feature.descriptors, k = 2)
						pairs = filterMatches(data.keypoints, feature.keypoints, matches)
						# Keypoints are matched and filtered
						# If n matched pairs remain feature is declared matching
						if len(pairs) >= 10:
							featureMatches.append(Match(object, feature, pairs))
							isSameObject = True
							
				# The feature is the same object if the keypoints match with the currently iterating object
				if isSameObject and isKnownObject:
					 # Object is deleted from the pool of known objects if feature found has already been found previous objects
					 # This is a crude way of removing duplicate objects
					objects.pop(b)
				else:
					if isSameObject:
						isKnownObject = True
						object.features.append(feature)
					b += 1
				
			# This feature is a known object if its keypoints match with one existing object
			if not isKnownObject:
				# If the feature is not a known object, add it as a the first occurence of a new object
				object = Object(str(frameNumber) + str(a))
				object.features.append(feature)
				objects.append(object)
				featureMatches.append(Match(object, feature, None))
		
		# Render object bounding box, keypoints and name if found in current frame
		lastName = ""
		for match in featureMatches:
			cv2.rectangle(frame, tuple(match.feature.bounds[:2]), tuple(match.feature.bounds[2:4]), (255, 255, 0), 2)
			if lastName != match.object.name:
				cv2.putText(frame, match.object.name, tuple(match.feature.bounds[:2]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
			lastName = match.object.name
		
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
