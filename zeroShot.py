# -*- coding: utf-8 -*-
"""
This file containes my methods and examples of performing zero shot learning.
The are 2 different feature extractor methods, this is because I utilised 2 different approaches in my report.
"""
import  numpy as np
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import os

"""
Below loads all the paths then files needed.
assuming the folder follows the same structure given in the assignment, then
base_dir is all that needs to be changed to run on a different system.
"""
#base directory for the 'Animals_with_attributes2' folder
base_dir = r'\\smbhome.uscs.susx.ac.uk\sb888\Documents\Vision\Subset_of_Animals_with_Attributes2\Animals_with_Attributes2'
#Location of the list of classes
class_path = os.path.join(base_dir,'classes.txt')
#location of the file containing the classes attributes
predicate_matrix_path = os.path.join(base_dir,'predicate-matrix-binary.txt')
#the attributes
predicate_path = os.path.join(base_dir,'predicates.txt')
#Location of the folder containing the animal image folders
image_dir = os.path.join(base_dir,'JPEGImages')
#the path for the training classes
train_path = os.path.join(base_dir,'trainclasses.txt')
#the path for the test classes
test_path = os.path.join(base_dir,'testclasses.txt')

#All of the classes loaded form file
classes = np.loadtxt(class_path, delimiter='\t', usecols=[1], dtype=np.str)
#Training classes loaded from file
train_classes = np.loadtxt(train_path, delimiter='\t', usecols=[0], dtype=np.str)
train_classes = np.array([cl.strip() for cl in train_classes])#proccess the loaded classes to strip away whitespace
#Testing classes loaded from file
test_classes = np.loadtxt(test_path, delimiter='\t', usecols=[0], dtype=np.str)
test_classes = np.array([cl.strip() for cl in test_classes])#whitespace stripped

"""
Extracts image descriptors using SURF then clusters accross all images to create
a histogram which is normalised to 1 to represent features per image.
I used 2 different zero shot approaches, this is the feature extractor for the standard zero shot approach

params: dir = base directory where image folders are stored
        classes = list of all classes
        CLUSTERS = the amount of output features per histogram and the amount
             of clusters when clustered over all image descriptors.
        HESSIAN_THRESHOLD = the hessian value used to determine the detail of
             descriptors extracted.
        UPRIGHT_ORIENTATION = 0 if orientation is calculated during SURF,
             1 if it is not.

returns: histogram per image where each bin of each histogram is refering to
         the normalised proportion of the same feature.
         for each image there is a Dx1 array where D is the amount of features/clusters
         for ease of use, the histogram sets are saved in a touples with the 0th elem being the name of that class and the 1th being the list of image histograms being that class. e.g.
         returns a list of tuples [(sheep,[featuresetimage1,featuresetimage2]),(goat,[featuresetimage1,featuresetimage2])...]
         this allows me to keep track of the featuresets.
"""

def extract_features(dir, classes, CLUSTERS = 120, HESSIAN_THRESHOLD = 500,UPRIGHT_ORIENTATION = 0):
    #image names in the directory
    files = [path for path in os.listdir(dir) if path != '.DS_Store']
    #pics per animal is used to take the first x images from each animal class
    pics_per_animal = 50

    """
        start extracting descriptors, saves them all in one variable along with
        a second variable of image_count in order to keep track of which descriptors where for which image
    """
    #All_descriptors collected from each image in the folders
    All_descriptors = np.empty([0,128])
    #descriptors saved per image, used to  keep track of amount of descriptors per image
    image_count = []
    #Iterates through each animal class
    for x in range(len(classes)):
        #the path to the animal image folder
        class_path = (dir+'/'+classes[x])
        #initialise that entry in dictionary to store  that animals features per image
        image_count.append((classes[x],[]))
        #list of all images in that animal file
        files = [path for path in os.listdir(class_path) if path != '.DS_Store']

        #ensures that there is enough images to reduce to pics per animal else doesnt change the image amount
        if(pics_per_animal < len(files)):
            #reduce the amount of pictures to the first x pictures
            files = [files[i] for i in range(pics_per_animal)]

        #Cycles through each image in the folder
        for image_path in files:
            #read image
            img = cv2.imread(os.path.join(class_path,image_path))
            #Creates the surf object from openCV. extended = true sets the descriptor size to 128
            surf = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD, extended = True, upright = UPRIGHT_ORIENTATION)
            #Compute the All_descriptors
            kp, des = surf.detectAndCompute(img,None)

            #concatenates the descriptors from the current image to the matrix of all descriptors
            All_descriptors = np.concatenate([All_descriptors,des])
             #records the number of descriptors per image to be later used for the histogram,also saved by class for further keeping track.
            image_count[x][1].append(len(des))

    """
        clusters descriptors, clusters are created from every images descriptors at the same time
    """
    #ensure descriptors are correct data type
    All_descriptors = np.float32(All_descriptors)
    #needed for the clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #performs K-means over all the descriptors from every image
    ret,labels,center=cv2.kmeans(All_descriptors,CLUSTERS,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    """
    Creates the histograms from the clusters and normalises the hisograms
    """
    #extracts the labels from the clustering
    labels = [x[0] for x in np.int32(labels)]
    #initialises where the histograms are stored
    hists= []
    #previous is used in conjunction with  image count to keep track of image number while iterating through every label
    previous = 0
    #iterates through image count which is the same as iterating through each animal class
    for count in image_count:
        #initialises the class histograms, this method returns a list of these
        class_hist = (count[0],[])
        #iterates through each image in that count, so im will be the amount of descriptors for that image.
        for im in count[1]:
            #below takes a slice of the labels for the particular image
            label_count = [labels[previous:previous+im].count(l) for l in range(CLUSTERS)]
            #ensures that the next image label count starts at the end of current images count
            previous += im
            #total is used for normalising the histogram
            total = sum(label_count)
            #normalises the histogram proportionaly so the sum of every bin equals 1
            hist = [bin/total for bin in label_count]
            #adds the histogram for this image to the set of class histograms
            class_hist[1].append(hist)
        #appends the set of image histograms for that class to the return product
        hists.append(class_hist)


    #simply use to measure the size of all descriptors used to compute memory cost before ending the method
    print('descriptors length: {}'.format(len(All_descriptors)))
    #histograms returns, grouped by class
    return hists


"""
Extracts image descriptors using SURF then clusters accross all images to create
a histogram which is normalised to 1 to represent features per image.
I used 2 different zero shot approaches, this is the feature extractor for the double cluster approach
the difference between this and the standard approach is that this performs an aditional inner cluster per image to reduce the memory requirements of saving the descriptors.

params: dir = base directory where image folders are stored
        classes = list of all classes
        CLUSTERS = the amount of output features per histogram and the amount
             of clusters when clustered over all image descriptors.
        HESSIAN_THRESHOLD = the hessian value used to determine the detail of
             descriptors extracted.
        UPRIGHT_ORIENTATION = 0 if orientation is calculated during SURF,
             1 if it is not.

returns: histogram per image where each bin of each histogram is refering to
         the normalised proportion of the same feature.
         for each image there is a Dx1 array where D is the amount of features/clusters
         for ease of use, the histogram sets are saved in a touples with the 0th elem being the name of that class and the 1th being the list of image histograms being that class. e.g.
         returns a list of tuples [(sheep,[featuresetimage1,featuresetimage2]),(goat,[featuresetimage1,featuresetimage2])...]
         this allows me to keep track of the featuresets.
"""
def double_cluster_extract_features(dir, classes, CLUSTERS = 120, HESSIAN_THRESHOLD = 500,UPRIGHT_ORIENTATION = 0):
    #image names in the directory
    files = [path for path in os.listdir(dir) if path != '.DS_Store']#sub directory names
    #for double clustering purposes, the amount of clusters extracted from each picture individually
    INNER_CLUSTERS = 500
    #the first 'x' pics from each animal folder
    pics_per_animal = 170
    #All_descriptors collected from each image in the folders, matrix is initialised here
    All_descriptors = np.empty([0,128])

    #descriptors saved per image
    image_count = []
    #iterates through each animal
    for x in range(len(classes)):
        #directory path for that animal
        class_path = (dir+'/'+classes[x])
        #initialise that animal as a tuple to keep track of the numbers
        image_count.append((classes[x],[]))
        #initialise list of all images in that animal file
        files = [path for path in os.listdir(class_path) if path != '.DS_Store']
        #only use the specified amount of images from each animal
        if(len(files) >= pics_per_animal):
            files = [files[i] for i in range(pics_per_animal)]



        """
            In the for loop below, I extract all the descriptors from each image while saving a the clusters of each image, this part I do to reducde ram requirements by a incredible amount
            while sacrificing accuracy.
        """
        #Cycles through each image in the folder
        for image_path in files:
            #read image
            img = cv2.imread(os.path.join(class_path,image_path))
            #Creates the surf object from openCV. extended = true sets the descriptor size to 128
            surf = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD, extended = True, upright = UPRIGHT_ORIENTATION)
            #Compute the All_descriptors using SURF
            kp, des = surf.detectAndCompute(img,None)
            if(type(des) != None):
                #if there are enough desriptors to inner cluster
                if(len(des) > INNER_CLUSTERS-1):
                    #convert to appropriate type
                    descriptors = np.float32(des)


                    """
                    This inner cluster step below is not neccessary and will reduce accurcy
                    However, it vastly reduces ram needs, allowing me to use alot more pictures than I would be able to without it
                    """
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    #saves the clusters per image as center
                    ret,labels,center=cv2.kmeans(descriptors,INNER_CLUSTERS,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                    #add the clusters of descriptors to all the discriptors
                    All_descriptors = np.concatenate([All_descriptors,center])
                    #records the number of descriptors per image to be later used for the histogram
                    image_count[x][1].append(len(center))
            #else add the descriptors without innner clustering
            else:
              All_descriptors = np.concatenate([All_descriptors,des])
              image_count[x][1].append(len(des))
        print('finished {}'.format(classes[x]))


    """
        clusters descriptors, clusters are created from every images descriptors at the same time
    """
    #ensure descriptors are correct data type
    All_descriptors = np.float32(All_descriptors)
    #needed for the clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #performs K-means over all the descriptors from every image
    ret,labels,center=cv2.kmeans(All_descriptors,CLUSTERS,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    """
    Creates the histograms from the clusters and normalises the hisograms
    """
    #extracts the labels from the clustering
    labels = [x[0] for x in np.int32(labels)]
    #initialises where the histograms are stored
    hists= []
    #previous is used in conjunction with  image count to keep track of image number while iterating through every label
    previous = 0
    #iterates through image count which is the same as iterating through each animal class
    for count in image_count:
        #initialises the class histograms, this method returns a list of these
        class_hist = (count[0],[])
        #iterates through each image in that count, so im will be the amount of descriptors for that image.
        for im in count[1]:
            #below takes a slice of the labels for the particular image
            label_count = [labels[previous:previous+im].count(l) for l in range(CLUSTERS)]
            #ensures that the next image label count starts at the end of current images count
            previous += im
            #total is used for normalising the histogram
            total = sum(label_count)
            #normalises the histogram proportionaly so the sum of every bin equals 1
            hist = [bin/total for bin in label_count]
            #adds the histogram for this image to the set of class histograms
            class_hist[1].append(hist)
        #appends the set of image histograms for that class to the return product
        hists.append(class_hist)


    #simply use to measure the size of all descriptors used to compute memory cost before ending the method
    print('descriptors length: {}'.format(len(All_descriptors)))
    #histograms returns, grouped by class
    return hists

"""
    This method iterates through each attribute and trains a classifier for each before returning those classifiers
    The classifiers are trained by compiling all of the negative data and positive data into those 2 groups

    params:classes = list of training classes used to train the classifiers
           features = feature histograms for all of the training training
           predicate_binary = the matrix of predicate binary of wether animals have an attribute
           attributes = list of all attributes
    returns: a list of 85 classifiers, one for each attribute.
"""
#outputs 85 attribute classifier models
def train_attribute_models(classes,features,predicate_binary,attributes):
    #initialises classifier list
    classifiers = []


    #iterates through each attribute index
    for i in range(len(attributes)):
        """
        This section compiles all the negative and positive data for the current attribute
        """
      positive_data = []
      negative_data = []
      #iterates through each line in the predicate table, so iterates through each animal classes predicate
      for x in range(len(predicate_binary)):
         #if that animal has the attribute being looked at then all its image features are added to positive
        if(predicate_binary[x][i] == 1):
          positive_data.append(features[x][1])
         #if it does not have the attribute then all its image features are added to negative data
        elif (predicate_binary[x][i] == 0):
          negative_data.append(features[x][1])

     """
     Below I proccess the data to be in the correct shape to train a classifier
     """
     #initialises data to store negative and positive data together
      data = []
      #initialises expected output, will be the same length as the the variable data
      output = []
      #iterates through the positive data, adds it to data with a coresponding expeted output in output
      for animal in positive_data:
        for image in animal:
          data.append(image)
          output.append(1)
     #iterates through the negative data, adds it to data with a coresponding expeted output in output
      for animal in negative_data:
        for image in animal:
          data.append(image)
          output.append(0)

      #Below I create the classifier, train it on data and output then add it to the classifier list.
      svm = LinearSVC()
      clf = CalibratedClassifierCV(svm, cv = 3)
      clf.fit(data, output)#data = training data, output = expected output per training data
      classifiers.append(clf)

     #returns list of classifiers
    return classifiers

"""
This method turns each set of features for each image into attribute probabilities, where each probability is the prob that picture contains that attribute.
params:classifiers = list of 85 classifiers, one per attributes
       test_data = the extracted feature histograms for all of the test images

returns: a list of tuples, for each tuple 0th element is the animal class name and the 1th element is a list of element per image where each element is the 85 probabilities that element contains that attribute
         once again I kep the classified images split up  by class to make it easier to process in the next step.

"""
#outputs a 85(attributes) by Ntest(testimages) matrix of probabilities with the (j,i) entry the probability that the j attribute is present in the i test image
def compute_attribute_probs(classifiers,test_data):
  output = []
  #iterates through each animal class in test_data
  for animal in test_data:
    #initialises a tuple, of (animal class string,list of attribute probabilities per image)
    animal_cls = (animal[0],[])
    #iterates through each images features
    for image in animal[1]:
      #initialises the attribute probabilities for this image
      attrib_probs = []
      #iterates through each classifier and store the returned probability in attrib probs
      for clf in classifiers:
        attrib_probs.append(clf.predict_proba([image])[0])
      #store the list of 85 attribute probabilities for the image in the tuple animal cls
      animal_cls[1].append(attrib_probs)
    #store the set of attribute probability sorted by class in the output list
    output.append(animal_cls)

  #returns for each animal a set of classified images with 1 classification per attribute that returns the probability that image contains that attribute
  return output

"""
This method turns the attribute probabilities into class probabilities for each image, so each image has the probability associated with it being each test class. this is achieved
by multiplying the probability that each attribute matches the attribute contained in the test predicate for each class in the test predicate.

params:test_predicate_matrix = the predicate matric partioned for to only include the predicates from the test classes
       test_probabilities = the test probabilities output from compute_attribute_probs()
       atts = the attributes used to produce the class probabilities, to use all attributes then a list of ints from 0 - 84 is used as atts

returns: a list of tuples where for each animal class where each tuples 0th element is the name of the animal class and the 1th element is the list of class probabilities per images
        I store the class probabilities in the way in order to make the computing of the accuracy easier and to keep track of the data easier.
"""
#outputs a 10(animal classes) by Ntest matrix of probabilities with (i,l) entry that the ith class is present in the lth image
def compute_class_probs(test_predicate_matrix,test_probabilities,atts):

  class_probs = []
  #the attributes used to calculate class probs(for standard test use all)
  attributes_used = atts
  #iterates the attribute probabilities by animal class
  for animal in test_probabilities:
    #stores the probabilities for each image in this class in animal probs
    animal_probs = []
    #iterates through each image set of attribute probabilities
    for image in animal[1]:
      #stores the class probabilities for one image, the length is the amount of test classes
      im_class_probs = []#per image
      #iterates through each animal class in the test predicate matrix
      for test_class in test_predicate_matrix:
         #the probability of this image being this class starts at 1, if it started at 0 then i would be multiplying by 0 constantly
        class_prob = 1
        #iterates through each attribute probability for this image and multiplies the probability that this image has that attribute and multiplies class probs by it
        for i in attributes_used:
          class_prob = class_prob * (1+image[i][int(test_class[i])])
        #saves the class probability for this image to im_class_prob
        im_class_probs.append(class_prob)
      #save the probabilities of this image being each class to animal_probs
      animal_probs.append(im_class_probs)
     #save this tuple as the animal name and the probabilites for each of its images
    class_probs.append((animal[0],animal_probs))

  return class_probs

"""
computes accuracy, this method works out the amount of images where the highest class probability is correct and divides it by the total amount of images to get the accuracy score between
1 and 0 where 0.2 euqates to 20% accuracy.

params: probs = the output of the class probabilites method

returns: float in range 0 <= x <= 1
"""
#outputs real number based on accuracy of system averaged over Ntest images
def compute_accuracy(probs):
  correct = 0
  total = 0
  #iterates through 10 test classes
  for i in range(len(probs)):
      #iterates through images
    for im in probs[i][1]:
      biggest = 0 #rolling biggest number
      index = 0 #index of biggest number
      #iterates through each probability of the image being that class, saving the biggest and its index
      for x in range(len(im)):
        if(im[x]> biggest):
          biggest = im[x]
          index = x
      #if the highest probability was correct then the amount of correct classifications is increased by one
      if(index == i):
        correct = correct+1
      #total is always increased by one
      total = total + 1
  #returns the amount of correct classifications/the total amount of test images
  return correct/total


"""
below i perform a bit of proccessing and use the above methods to perform the tests
"""
#50 by 85 matrix, 50 animal classes by 85 attributes
predicate_matrix = np.loadtxt(predicate_matrix_path)



#The predicate matrix for the training data
train_predicate_matrix = [predicate_matrix[i,:] for i in range(len(classes)) if classes[i] in train_classes]
#the predicate matrix for the test data
test_predicate_matrix = [predicate_matrix[i,:] for i in range(len(classes)) if classes[i] in test_classes]
#All the Attributes
attributes = np.loadtxt(predicate_path, delimiter='\t', usecols=[1], dtype=np.str)

#Features extracted from all classes with all images(SURF bags of words per image)
all_features = extract_features(image_dir,classes)
#Training features(SURF bags of words per image)
training_data =[feat for feat in all_features if (np.isin(feat[0],train_classes))]
#Testing features(SURF bags of words per image)
test_data = [feat for feat in all_features if (np.isin(feat[0],test_classes))]



"""
This is where the method are chained together to produce the accuracy
"""
classifiers = train_attribute_models(train_classes,training_data,train_predicate_matrix,attributes)
probabilities = compute_attribute_probs(classifiers,test_data)
class_probs = compute_class_probs(test_predicate_matrix,probabilities,[x for x in range(84)])
print('origional accuracy: {}'.format(compute_accuracy(class_probs)))

"""
Below is just a prtition of the attributes for a brief analysis using a subset of attributes to produce the class probabilities
"""
physical = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
colour = [0,1,2,3,4,5,6,7,8]
skin = [9,10,11,12,13]
temperment = [46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84]

class_probs = compute_class_probs(test_predicate_matrix,probabilities,colour)
print('colour atts accuracy: {}'.format(compute_accuracy(class_probs)))

class_probs = compute_class_probs(test_predicate_matrix,probabilities,skin)
print('skin accuracy: {}'.format(compute_accuracy(class_probs)))

class_probs = compute_class_probs(test_predicate_matrix,probabilities,physical)
print('physical accuracy: {}'.format(compute_accuracy(class_probs)))

class_probs = compute_class_probs(test_predicate_matrix,probabilities,temperment)
print('temperment accuracy: {}'.format(compute_accuracy(class_probs)))
