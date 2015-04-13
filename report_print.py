import cv2
import cv2.cv as cv
import numpy
from skimage.measure import label
from skimage.morphology import square
import os
import time

import random
class BinarySearchTree:
        """A binary search tree implementation"""
        root = None
        def __init(self, root=None):
                self.root = root

        def insertObject(self, object, randomize=True):
                """Inserts the object in weighted position in the tree. Implementation from 
                521144A Algorithms and Data Structures 2014 assignment implementation code."""
                x = self.root
                y = None
                i = 0
                while x != None:
                        y = x
                        i = random.randint(0,1)
                        if i == 0:
                                x = x.left
                        else:
                                x = x.right
                if y == None:
                        #Tree was empty
                        self.root = object
                        print "added root", self.root
                else:
                        if i == 0:
                                #print "adding child left", object
                                y.left = object
                        else:
                                y.right = object
                                #print "adding child right", object


                #--------For weighted objects. Weighting to be implemented----------
                #       if object.key < x.key:
                #               x = x.left
                #       elif object.key == x.key and randomize==True:
                #               #random assignment on equal can significantly reduce the height of the tree
                #               rand=random.choice([x.right, x.left])
                #               x = rand
                #       else:
                #               x=x.right

                # object.parent = y
                
                # if y == None:
                #       #tree was empty
                #       self.root = object

                # else:

                #       if object.key < y.key:
                #               y.left = object

                #       elif randomize==True:
                #               #random assignment on equal can significantly reduce the height of the tree
                #               if object.key == y.key:
                #                       rand=random.choice([y.right, y.left])
                #                       rand=object
                #       else:
                #               y.right = object

        def startSearch(self, featureList, matcher, colorIndex):
                #check if tree is empty
                if self.root == None:
                        #add all new objects to tree if the tree is empty
                        for a, feature in enumerate(featureList):
                                object = Object(str(frameNumber) + str(a), colors[colorIndex % len(colors)], featureList=[feature])
                                #Insert the node into bst
                                self.insertObject(object)
                                #print "object", object
                                colorIndex += 1
                
                node = self.root
                print "node in startSearch", node
                #Initialize featureMatches for new search
                featureMatches = []
                featureMatches = self.objectSearch(node, featureList, matcher, featureMatches, colorIndex)

                return featureMatches

        def objectSearch(self, node, featureList, matcher, featureMatches, colorIndex):
                """Recursively iterates through the tree searching for a match"""
                #skip rotation if leaf node reached

                if not node:
                        return featureMatches
                print "node.left", node.left
                left = self.objectSearch(node.left, featureList, matcher, featureMatches, colorIndex)
                print "node.right", node.right
                right = self.objectSearch(node.right, featureList, matcher, featureMatches, colorIndex)
                featureMatches = self.searchMatch(node.left, node.right, featureList, matcher, featureMatches, colorIndex)

                return  


        def searchMatch(self, leftNode, rightNode, featureList, matcher, featureMatches, colorIndex):
                """Look for feature matches in current node's children."""
                #matcher = cv2.BFMatcher object
                print "asdfasdf", leftNode, rightNode
                if leftNode == None and rightNode == None:
                        return featureMatches
                nodes = [leftNode, rightNode]
                for a, feature in enumerate(featureList):

                        isKnownObject = False
                        #Check if left or right object matches
                        for node in nodes:
                                print "nodes", nodes
                                #Check that node is Object instances
                                if isinstance(node, Object):
                                        print "isinstance node", node
                                        #To limit processing power needed only n newest occurrences of an object are kept in the feature list
                                        if len(node.features) > 5:
                                                node.features = node.features[1:]
                                        for featureObject in node.features:

                                                if featureObject.descriptors != None and node.features != None:
                                                        for feature in node.features:
                                                                #Filter keypoints and matches
                                                                matches = matcher.knnMatch(featureObject.descriptors, feature.descriptors, k=2)
                                                                pairs = filterMatches(featureObject.keypoints, feature.keypoints, matches)
                                                                #Feature is declared matching if n matched pairs remain
                                                                if len(pairs) >= 10:
                                                                        #add new features to existing object
                                                                        node.features.append(feature)
                                                                        #Add match to found matches
                                                                        featureMatches.append(Match(node, feature, pairs))

                                                                        isSameObject = True
                                                                        isKnownObject = True

                        if not isKnownObject:
                                #If the feature is not a known object, add it as the first occurrence of a new object
                                object = Object(str(frameNumber) + str(a), colors[colorIndex % len(colors)], feature)
                                #Insert the node into bst
                                self.insertObject(object)

                                colorIndex += 1

                return featureMatches

        def searchNeighborhood(self, k):
                """Returns the first node with key k in the subtree. Used as starting point node for searchMatch()"""
                #Follows the binary search tree's iterative search algorithm
                node = self.root
                while node != None and k != node.key:
                        if k < node.key:
                                node = node.left
                        else:
                                node = node.right
                return node


class Object:
        def __init__(self, name, color, featureList=[], key=None, shape=None):
                self.features = featureList
                self.name = name
                self.color = color
                self.key = key # The key weight value of the node
                self.left = None # A pointer to the left child node
                self.right = None # A pointer to the right child node
                self.parent = None # A pointer to the parent node
                self.shape = shape
class Feature:
        def __init__(self, keypoints, descriptors):
                self.keypoints = keypoints
                self.descriptors = descriptors
                
class Match:
        def __init__(self, object, feature, keypointPairs):
                self.object = object
                self.feature = feature
                self.keypointPairs = keypointPairs
                self._calculateBoundingBox()
                
        def _calculateBoundingBox(self):
                """Private function for calculating bounding box of keypoints of the new feature"""
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
        values = numpy.unique(segmented)
        gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)
        for value in values:
                segment = gray.copy()
                segment[segmented != value] = 0
                segments.append(segment)
        #delete the segment containing all segments
        del segments[0]
        return segments;

def featureExtractor(detector, extractor, segments):
        """Extracts features from segmented image(s)"""
        features = []
        shapes = []
        for i, segment in enumerate(segments):
                ret, mask = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY)
                keypoints = detector.detect(segment, mask)
                keypoints, descriptors = extractor.compute(segment, keypoints, mask)

                shape = shapeDetection(segment)
                shapes.append(shape)
                features.append(Feature(keypoints, descriptors))
        return features, shapes;


def shapeDetection(image):
        """Detects object shape from image"""
        shape = "None"
        #Check if object shape is circle
        circle = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT, 2, 100)
        if circle is not None:
                shape = "circle"
                
        return shape

def matchFinder(features, objects, frameNumber, colorIndex, matcher, shapes):
        """Matches object features against database"""
        # Iterate through each feature found in the frame
        featureMatches = []
        for a, feature in enumerate(features):
                isKnownObject = False
                b = 0
                
                # Iterate through every known object
                while b < len(objects):
                        object = objects[b]
                        
                        # To limit processing power needed only n newest occurences of an object are kept
                        if len(object.features) > 5:
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
                        object = Object(str(a), colors[colorIndex % len(colors)], shape=str(shapes[a]))
                        object.features.append(feature)
                        #Insert object to BinarySearchTree
                        #bst.insertObject(object)
                        objects.append(object) #placeholder
                        colorIndex += 1
        return featureMatches

def addToDatabase(object):
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

        # Videos need to be named in a videos.txt as video per line without the file extension.
        videos = []
        videos_txt = open("videos.txt","r")
        for line in videos_txt:
                line = line.strip()
                videos.append(line)
        global i
        i =0
        while 1:
                objects = []
                print videos[i]
                testvideo = videos[i]
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                detector = cv2.FeatureDetector_create("ORB")
                extractor = cv2.DescriptorExtractor_create("ORB")
                camera = cv2.VideoCapture(str(testvideo)+".mp4")
                global frameNumber
                frameNumber = 0

                # Colors for debugging, each object is given a color to differentiate in the debug image
                global colors
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
                colorIndex = 0
                #bst = BinarySearchTree()
                all_time_s = 0
                all_time_f = 0
                flag = True
        
                while flag == True:
                        ret, frame = camera.read()
                        txtfile = open(testvideo +".txt","a")
                        if frame == None:
                                txtfile.write("\nOverall segmentation time: "+str(all_time_s))
                                txtfile.write("\nOverall Feature and shape extraction time: "+str(all_time_f))
                                txtfile.close()
                                i += 1
                                print "Done"
                                flag = False
                                break
                        start_seg = time.clock()
                        segmented = segmentation(frame)
                        segments = extractSegments(frame, segmented)
                        end_seg = time.clock()
                        time_s = end_seg - start_seg
                        all_time_s += time_s
                        txtfile.write("\nFrameNumber: "+ str(frameNumber)+"\nSegmentation time: " +str(time_s)+"\n")

                        feature_start = time.clock()
                        
                        features, shapes = featureExtractor(detector, extractor, segments)
                        featureMatches = matchFinder(features, objects, frameNumber, colorIndex, matcher, shapes)
                        #featureMatches = bst.startSearch(features, matcher, colorIndex)
                        feature_end = time.clock()
                        time_f = feature_end - feature_start
                        all_time_f += time_f
                        txtfile.write("Feature and shape extraction time: " +str(time_f)+"\n")
                        
                        for a in objects:
                                txtfile.write("Object: "+str(a.name)+", ")
                                txtfile.write("Color: "+str(a.color)+", ")
                                txtfile.write("Shape: "+str(a.shape)+"\n")

                        frameNumber += 1

if __name__ == '__main__':
        try:
                main()
        except KeyboardInterrupt:
                print 'Interrupted, proceeding'
