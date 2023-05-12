#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:41:10 2023
Modify on Wed Feb 23 15:22:20 2023

@author: michaelka，Xin
"""
import math

# Enter student number here
# student number : 33662797
#----------------------------------------------------------------------

#TASK 1

def change_calculator(value):
    """
    Function description
    @ Function: The least amount of coins required to make the given value in pounds.
    @ Keyword arguements input:
    value : float
            The value of pounds converted from the coins.
    @ Return:
    two_pound, one_pound, p50, p20, p10, p5, p2, p1: tuple
            If the value is outside the range of £0 to £5, the function will raise ValueError: "Oops! The Value have to between £0 and £5."
            else return the number of coins required for each denomination, in the following order:
            (two_pound, one_pound, p50, p20, p10, p5, p2, p1)
    """
    # confirm that the value is in the range £0 to £5
    if value < 0 or value > 5:
        raise ValueError("Oops! The Value have to between £0 and £5.")
    # convert the value to pence
    pence = int(value * 100)
    # calculate the number of coins required
    two_pound = pence//200
    one_pound = (pence - two_pound*2) // 100
    p50 = (pence-two_pound*200-one_pound*100) // 50
    p20 = (pence-two_pound*200-one_pound*100-p50*50) // 20
    p10 = (pence-two_pound*200-one_pound*100-p50*50-p20*20) // 10
    p5 = (pence-two_pound*200-one_pound*100-p50*50-p20*20-p10*10) // 5
    p2 = (pence-two_pound*200-one_pound*100-p50*50-p20*20-p10*10-p5*5) // 2
    p1 = (pence-two_pound*200-one_pound*100-p50*50-p20*20-p10*10-p5*5-p2*2) // 1

    return two_pound, one_pound, p50, p20, p10, p5, p2, p1


#----------------------------------------------------------------------

#TASK 2

def radians_to_degrees(angle_radians):
    """
    Function description
    @ function: Converts an angle in radians to degrees.
    @ Keyword arguements input:
    angle_radians : float
                    The angle in radians.
    @ Return:
    angle_degrees : float
                    The angle in degrees.
    """
    # Tip: math.pi = 3.141592653589793, so the math.pi can replace by 3.141592653589793
    angle_degrees = angle_radians * 180 / math.pi
    return angle_degrees

#----------------------------------------------------------------------

#TASK 3

def myMean(data):
    """
    Function description
    @ Function:  Calculates the mean of a given dataset.
    @ Keyword arguements input:
    data : list or tuple
           The numerical values representing the dataset.
    @ Return:
    mean : float
           The mean of the dataset.
    """
    sum = 0 # initialize the sum to zero.
    # use the loops to calculate the sum.
    for i in range(len(data)):
        sum += data[i]
    mean = sum / len(data)
    return mean



def myStd(data):
    """
    Function description
    @ function:  Calculates the standard deviation of a given dataset.
    @ Keyword arguements input:
    data : list or tuple
           The numerical values representing the dataset, which must be more than two elements.
    @ Returns:
    std : float
          If the data is less than two elements, the function will raise ValueError: "Oops! Dataset must contain at least two elements".
          Else return the standard deviation of the dataset.
    """
    # check the length of dataset
    length = len(data)
    if length < 2:
        raise ValueError("Oops! Dataset must contain at least two elements.")
    # calculate the mean
    mean = myMean(data)
    # use loops to calculate the sum squared diff.
    sum_squared_diff = sum((x - mean) ** 2 for x in data)
    # Tip: also can use : (sum_squared_diff / length) ** 0.5 instead of math.sqrt
    std = math.sqrt(sum_squared_diff / length)
    return std

#----------------------------------------------------------------------

#TASK 4

def mark_total(data):
    """
    Function description
    @ Function:  given the mean and standard deviations of exam marks (each mark is denoted EM),
                 assigns a grade to each student mark and outputs the total number of students who obtained each grade.
    @ Keyword arguements input:
    data : list or tuple
           The numerical values representing the dataset of exam mark.
    @ Returns:
    A, B ,C, D, F : tuple
                    The number of each grade, in the following order: (A, B, C, D, F)
    """
    A = B = C = D = F = 0
    # caculate the mean and standard deviations using function
    mean = myMean(data)
    std = myStd(data)
    # loops exam marks
    for mark in data:
        # A
        if mark >= mean + 1.5 * std:
            A += 1
        # B
        elif mean + 0.5 * std <= mark < mean + 1.5 * std:
            B += 1
        # C
        elif mean - 0.5 * std <= mark < mean + 0.5 * std:
            C += 1
        # D
        elif mean - 1.5 * std <= mark < mean - 0.5 * std:
            D += 1
        # F
        else:
            F += 1
    return A, B, C, D, F

if __name__ == '__main__':
    print(change_calculator(4.44))