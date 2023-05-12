#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:15:24 2023

@author: michaelka
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pandas as pd

# Enter student number here
# Student number: 33662797
#----------------------------------------------------------------------

#TASK 1

def Gaussian_Distribution(mu, sigma, x):
    """
    Function description
    @ Function:  Calculates the Gaussian distribution using μ, σ and x.
    @ Keyword arguements input:
    mu : float, the μ
    sigma : float, the σ
    x : float, the x
    @ Returns:
    Fx : float
          If the sigma is equal to 0, the function will raise ValueError: "Oops! Standard deviation (sigma) cannot be zero".
          Else return the value of Gaussian distribution.
    """
    # check if the sigma is equal to 0
    if sigma == 0:
        raise ValueError("Oops! Standard deviation (sigma) cannot be zero.")

    # calculate the gaussian by the fomula of f(x) = (1 / (σ * sqrt(2 * pi))) * exp(-((x - μ)2) / (2 * σ2))
    Fx = (1 / (sigma * np.sqrt(2 * pi))) * np.exp(-(np.power(x - mu, 2) / (2 * np.power(sigma, 2))))
    return Fx


def Gaussian_plot():
    """
    Function description
    @ Function:  Plot f(x) for the following values of μ and σ2 for −4 ≤ x ≤ 4:
    μ = 0 σ2 = 0.2
    μ = 0 σ2 = 0.4
    μ = 0 σ2 = 4
    μ = −2 σ2 = 0.5.
    """
    # define the μ, σ_square, and x range
    mu_values = [0, 0, 0, -2]
    sigma_square_values = [0.2, 0.4, 4, 0.5]
    x = np.arange(-4, 4, 0.1)
    # loop the values of mu and sigma_square to plot the Gaussian distribution curves
    for mu, sigma_square in zip(mu_values, sigma_square_values):
        y = Gaussian_Distribution(mu=mu, sigma=np.sqrt(sigma_square), x=x)
        plt.plot(x, y, label=r'$\mu={}$, $\sigma^2={}$'.format(mu, sigma_square))

    # plot the labels and legend
    # avoid the figure overlapping with sunshine_hours_city()
    fig = plt.figure(1, figsize=(8, 6))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gaussian Distribution Curves')
    plt.legend(loc='upper left')
    plt.show()
    

#----------------------------------------------------------------------

#TASK 2
def collatz(n, m):
    """
    Function description
    @ Function:  Which one reached 1 in the fewest iterations from given two positive integers.
    @ Keyword arguements input:
    n : int, positive integer
    m : int, positive integer
    @ Returns:
    result : String
            If the n or m are not positive, the function will raise ValueError: "Oops! The n or m has to be positive".
            Else return the standard deviation of the dataset.
    """
    # check if n or m is positive number
    if n <= 0 | m <= 0:
        raise ValueError("Oops! The n or m has to be positive.")

    # save the original n and m
    original_n = n
    original_m = m

    # initialize the iterations of n and m
    n_iteration_count = 0
    m_iteration_count = 0

    # loop that will continue as long as either n or m is greater than 1
    while n > 1 or m > 1:
        # if n is greater than 1, calculate the next number in its Collatz sequence
        if n > 1:
            if n % 2 == 0:
                n //= 2
            else:
                n = n * 3 + 1
            n_iteration_count += 1
        # if m is greater than 1, calculate the next number in its Collatz sequence
        if m > 1:
            if m % 2 == 0:
                m //= 2
            else:
                m = m * 3 + 1
            m_iteration_count += 1

    # compare the number of iterations it took for n and m to reach 1
    if n_iteration_count < m_iteration_count:
        result = f"n={original_n} took {m_iteration_count - n_iteration_count} fewer iterations than m={original_m} to reach 1"
    else:
        result = f"m={original_m} took {n_iteration_count - m_iteration_count} fewer iterations than n={original_n} to reach 1"

    return result


#----------------------------------------------------------------------

#TASK 3
def sunshine_hours_city(file_name):
    """
    Function description
    @ Function:  Find the continent with the yearly largest average sunshine hours, the city with the smallest number of sunshine hours in the month of July,
    and find and plot the 12 cities with the lowest sunshine hours in the month of January.
    @ Keyword arguements input:
    file_name : String, the file name of dataset of CSV file.
    @ Returns:
    continent_with_max_avg_sunshine : String, the continent with the highest average yearly sunshine hours
    city_with_min_sunshine_in_july: String, the city with the smallest number of sunshine hours in July
    plt.gcf(): matplotlib.figure.Figure, the histogram from the function.
    """
    df = pd.read_csv(file_name)

    # 1. Find the continent with the yearly largest average sunshine hours
    continent_avg_sunshine = df.groupby('Continent')['Year'].mean()
    continent_with_max_avg_sunshine = continent_avg_sunshine.idxmax()
    print('Continent with the highest average yearly sunshine hours: ', continent_with_max_avg_sunshine)

    # 2. Find the city with the smallest number of sunshine hours in July
    city_with_min_sunshine_in_july = df.loc[df['Jul'].idxmin()]['City']
    print('City with the smallest number of sunshine hours in July: ', city_with_min_sunshine_in_july)

    # 3.1 Find the 12 cities with the lowest sunshine hours in January and produce a bar chart of their yearly sunshine hours
    cities_with_lowest_jan_sunshine = df.nsmallest(12, 'Jan')

    # 3.2 produce a bar chart of these cities of their yearly sunshine hours
    # avoid the figure overlapping with Gaussian_plot()
    fig = plt.figure(2, figsize=(10, 6))
    plt.bar(cities_with_lowest_jan_sunshine['City'], cities_with_lowest_jan_sunshine['Year'])
    # plot the numbers of hour on the bar
    for a, b in zip(cities_with_lowest_jan_sunshine['City'], cities_with_lowest_jan_sunshine['Year']):
        plt.text(a, b,
                 b,
                 ha='center',
                 va='bottom',
                 )
    plt.xticks(rotation=30)
    plt.xlabel('City')
    plt.ylabel('Sunshine Hours')
    plt.title('Cities with the lowest sunshine hours in January')
    plt.show()

    return continent_with_max_avg_sunshine, city_with_min_sunshine_in_july, plt.gcf()

    
    

#----------------------------------------------------------------------

#TASK 4  
    
def area(file_name):
    '''
    Function description
    @ Function:  Using the prototype provided, write a function that calculates and returns the area A.
    A =1/2(xny1 − x1yn) + 1/2(nX−1∑i=1)(xiyi+1 − xi+1yi) .
    @ Keyword arguements input:
    file_name : String, the file name of dataset of CSV file.
    @ Returns:
    polygon_area : float, the number of polygon area.
    '''
    # load the CSV file as a pandas dataframe
    df = pd.read_csv(file_name)

    # extract the 'X' and 'Y' columns as a list of points
    points = df[['X', 'Y']].values.tolist()
    n = len(points)
    polygon_area = 0

    # Get the coordinates of the last and first points
    x_n, y_n = points[n-1]
    x_1, y_1 = points[0]

    # calculate the area of the polygon using the formula
    for i in range(n-1):
        xi, yi = points[i]
        xi_p_1, yi_p_1 = points[i + 1]
        polygon_area += xi * yi_p_1 - xi_p_1 * yi

    # calculate the absolute value of the area using formula
    polygon_area = abs(1/2 * polygon_area + 1/2 * (x_n * y_1 - x_1 * y_n))

    return polygon_area
    

def area_minimal(file_name):
    '''
    Function description
    @ Function:  minimises the polygon area by removing one point.
    A =1/2(xny1 − x1yn) + 1/2(nX−1∑i=1)(xiyi+1 − xi+1yi).
    @ Keyword arguements input:
    file_name : String, the file name of dataset of CSV file.
    @ Returns:
    points[min_index] : tuple, the optimal removing point minimising the area.
    min_area : float, the minimised polygon area by removing one point.
    '''
    # load the CSV file as a pandas dataframe
    df = pd.read_csv(file_name)

    # extract the 'X' and 'Y' columns
    points = df[['X', 'Y']].values.tolist()

    # calculate the area of the original polygon
    area_original = area(file_name)

    # initialize variables to keep track of the minimum area and index of the point that minimizes the area
    min_area = area_original
    min_index = -1
    n = len(points)

    # loop to remove one point  at a time and calculate the area of the new polygon
    for i in range(n-1):
        # copy the original points to a new list
        new_points = points.copy()

        # Initialize the new polygon area equal to 0
        new_area = 0

        # delete i-th point
        new_points.pop(i)

        # if remove one point, the index of the last point in the new list is n-2
        x_n, y_n = new_points[n - 2]
        x_1, y_1 = new_points[0]

        # calculate the area of the new polygon using the formula
        for j in range(n - 2):
            print(j)
            xi, yi = new_points[j]
            xi_p_1, yi_p_1 = new_points[j + 1]  # index wraps around to 0 for last point
            new_area += xi * yi_p_1 - xi_p_1 * yi

        # calculate the absolute value of the new polygon
        new_area = abs(1 / 2 * new_area + 1 / 2 * (x_n * y_1 - x_1 * y_n))

        # update the minimum area and index
        if new_area < min_area:
            min_area = new_area
            min_index = i

    return points[min_index], min_area




