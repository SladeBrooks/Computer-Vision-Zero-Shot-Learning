# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Below I plot the accuracy ratings on a graph of accuracy to pictures per animals used in training.
This gives an indicator of expected results with differing amounts of source images used.
While my hardware limitations restricted me from using the full set of images for training, these graphs prodive a rough idea of what results could be expected with full image set.
####################################talk about how achieving the same accuracy is possible with double clust method but more pics required
"""

#Plots the accuracy ratings for the standard zero shot approach I employed
standard_zero_shot = [[10,20,30,40,100],[0.17,0.22,0.23,0.22,0.219]]# the 0th element is the pictures per class and the 1th is the accuracy ratings
single_clust_plot = plt.plot( standard_zero_shot[0],standard_zero_shot[1], label = 'Standard Zero Shot Solution')

#Plots the accuracy rates for the zero shot approach where each image descriptors are pre clustered to reduce ram
double_clust_results = [[20,50,120,130],[0.135,0.1729,0.24,0.23]]# the 0th element is the pictures per class and the 1th is the accuracy ratings#######------double check 140 ims
two = plt.plot(double_clust_results[0],double_clust_results[1], label = 'Zero Shot With Per Image Clustering')

#displaying the plotted performances of the 2 approaches employed
plt.xlabel('Max images per class')#this axis is on amount of images used per animal class
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
Below I plot the cost requirements against the accuracy for each algorithm
The cost is amount of descriptors, each descriptor being a vector of 126 floating point numbers
###################################talk about how this vastl decreased ram needs but increased picture amount needs.
"""
#Plots the size cost against accuracy for the standard zero shot approach
standard_zero_shot = [[0.17,0.22,0.23],[1299520,2579395,5119889]]# the 0th element is the accuracies and the 1th is the size cost
single_clust_plot = plt.plot( standard_zero_shot[0],standard_zero_shot[1], label = 'Standard Zero Shot Solution')

#Plots the size cost against accuracy for the double clustering zero shot approach
double_clust_results = [[0.17,0.22,0.23],[20*50*500,50*50*500,120*50*500]]# the 0th element is the accuracies and the 1th is the size cost
double_clust_plot = plt.plot( double_clust_results[0],double_clust_results[1], label = 'Zero Shot With Per Image Clustering')

#displaying the plotted performances of the 2 approaches employed
plt.xlabel('Accuracy')#this axis is on amount of images used per animal class
plt.ylabel('Size Cost')
plt.legend()
plt.show()




"""
Below I plot the attribute type accuracies
#################################use this to talk about the importance performing a type of feature selection on attributes
"""
#Plots the accuracies for colour related attributes
colour_acc = [[10,20,30,40,100],[0.1,0.16,0.16,0.135,0.123]]#the 0th element is the pics per class and the 1th element is the accuracies for this group of attributes
colour_plot = plt.plot( colour_acc[0],colour_acc[1], label = 'Colour Related Attributes')

#Plots the accuracies for skin related attributes
colour_acc = [[10,20,30,40,100],[0.12,0.115,0.12666666666666668,0.1275,0.123]]#the 0th element is the pics per class and the 1th element is the accuracies for this group of attributes
colour_plot = plt.plot( colour_acc[0],colour_acc[1], label = 'Skin Related Attributes')

#Plots the accuracies for physical related attributes
colour_acc = [[10,20,30,40,100],[0.21,0.19,0.17666666666666667,0.1875,0.208]]#the 0th element is the pics per class and the 1th element is the accuracies for this group of attributes
colour_plot = plt.plot( colour_acc[0],colour_acc[1], label = 'Physical Related Attributes')

#Plots the accuracies for temperment related attributes
colour_acc = [[10,20,30,40,100],[0.16,0.21,0.20666666666666667,0.2025,0.209]]#the 0th element is the pics per class and the 1th element is the accuracies for this group of attributes
colour_plot = plt.plot( colour_acc[0],colour_acc[1], label = 'Temperment Related Attributes')

#Plots total accuracies
colour_acc = [[10,20,30,40,100],[0.17,0.22,0.23,0.22,0.219]]#the 0th element is the pics per class and the 1th element is the accuracies for this group of attributes
colour_plot = plt.plot( colour_acc[0],colour_acc[1], label = 'All Atributes Accuracy')

#displaying the plotted performances of the 2 approaches employed
plt.xlabel('Pics Per Animal Class')#this axis is on amount of images used per animal class
plt.ylabel('Attribute Class Accuracy')
plt.legend()
plt.show()
