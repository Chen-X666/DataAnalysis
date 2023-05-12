#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:49:59 2023

@author: michaelka
"""
import random
from collections import Counter


# Enter student number here
# 33662797
#----------------------------------------------------------------------

#TASK 1

def English_to_Morse(text):
    """
    Take a string of characters (words) in English and coverts the string to Morse code.
    Parameters
    ----------
    text : string
        The input string of English characters to convert to Morse code.
    Returns
    -------
    result: string
        The input string converted to Morse code.
        Words are separated by a single space character.
    """
    # Define a dictionary that maps English characters to their Morse code equivalents
    eng_to_morse_dict = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
                       'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
                       'M': '--', 'N': '-.', 'O': '---', 'P':    '.--.', 'Q': '--.-', 'R': '.-.',
                       'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
                       'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
                       '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
                       '8': '---..', '9': '----.', '.': '.-.-.-', ',': '--..--', '?': '..--..',
                       "'": '.----.', '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
                       '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.',
                       '-': '-....-', '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.',
                       ' ': ' '}
    # Initialize an empty list to store the Morse code symbols
    morse_text = []
    # Iterate over each character in the input text
    for char in text.upper():
        # If the character is in the dictionary, append its Morse code symbol to the list
        if char in eng_to_morse_dict:
            morse_text.append(eng_to_morse_dict[char])
    # Join the Morse code symbols together with spaces between them
    result = ' '.join(morse_text)
    return result

#----------------------------------------------------------------------

#TASK 2
def run_regional_election(prob_A_wins):
    """
    Simulates a regional election given the probability of candidate A winning.
    Parameters
    ----------
    prob_A_wins : float
        The probability of candidate A winning the election. Must be between 0 and 1.
    Returns
    -------
    result: boolean
        Returns True if candidate A wins the simulated election, and False otherwise.
    """
    # Initialize the result variable to False
    result = False
    # Generate a random number between 0 and 1, and compare it to the given probability of A winning
    if random.random() < prob_A_wins:
        # If the random number is less than the probability of A winning, set result to True
        result = True
        return result
    else:
        return result


def run_election(regional_chances):
    """
    Simulates the election given the chances of candidate A winning in each region.
    Candidate A wins the election if they win at least 3 regions.
    Parameters
    -----------
    regional_chances : list of floats
        A list containing the probability of candidate A winning in each region. Each value must be between 0 and 1.
    Returns
    -------
    result: boolean
        Returns True if candidate A wins in at least 3 regions, and False otherwise.
    """
    # Initialize the number of wins for candidate A to 0 and the result to False
    num_wins_A = 0
    result = False
    # Iterate through the regional chances
    for prob_A in regional_chances:
        # Simulate a regional election with the given probability for candidate A
        if run_regional_election(prob_A):
            num_wins_A += 1
        # If candidate A has won in at least 3 regions, set the result to True and return it
        if num_wins_A == 3:
            result = True
            return result
    # If candidate A has not won in at least 3 regions, return False
    return result
        
def percentage_A(N):
    """
    By using the idea of Monte Carlo simulation，simulates N elections using run_election() function,
    and calculates the percentage of elections that candidate A wins.
    The regional probabilities of candidate A winning are fixed to:
    Region 1: 84% chance
    Region 2: 60% chance
    Region 3: 10% chance
    Region 4: 35% chance
    Region 5: 40% chance
    Parameters
    ----------
    N : int
        The number of elections to simulate.
    Returns
    -------
    percentage : float
        The percentage of simulated elections that candidate A wins.
    """
    # Initialize a counter for the number of times candidate A wins
    num_wins_A = 0
    # Simulate N elections with fixed regional probabilities and count the number of times candidate A wins
    for i in range(N):
        if run_election([0.84, 0.6, 0.1, 0.35, 0.4]):
            num_wins_A += 1
    # Calculate the percentage of simulated elections that candidate A wins
    percentage = (num_wins_A / N)
    return percentage

#----------------------------------------------------------------------

#TASK 3

def lettersFrequency(file_name):
    """
    Finds the letter with the highest and lowest number of occurrences in the given file.
    Parameters
    -----------
    file_name : string
        The name of the text file.
    Returns
    -------
    letter_highest : string
    freq_highest : int
    letter_lowest : string
    freq_lowest : int
        Returns four variables: letter highest, freq highest, letter lowest, freq lowest.
    """
    # Read the file
    with open(file_name, 'r') as file:
        contents = file.read()

    # Convert all letters to lowercase to ensure case-insensitivity
    contents = contents.lower()
    # Count the occurrences of each letter
    letter_counts = Counter(char for char in contents if char.isalpha())

    # Find the letter with the highest and lowest frequency
    letter_highest, freq_highest = letter_counts.most_common(1)[0]
    letter_lowest, freq_lowest = letter_counts.most_common()[-1]

    return letter_highest, freq_highest, letter_lowest, freq_lowest


#----------------------------------------------------------------------

#TASK 4

def rock_paper_scissors():
    """
    Simulates a game of rock-paper-scissors between the user and the computer. Prompts the user to input their choice and validates it.
    1. If the user enters 'exit', the function returns a message thanking the user for the game
    2. If the user's choice is valid, the function randomly selects the computer's choice, determines the outcome of the game and returns a sentence describing the outcome of the game
    3. If the user's selection is invalid then loop 1.2
    Returns:
    --------
    str_resul : string
        A sentence describing the outcome of the game.
    """
    # define the possible choices
    game_choices = ['rock', 'paper', 'scissors']

    # ask the user to input their choice or exit, if choices that validate it
    while True:
        user_choice = input("Please choose rock, paper or scissors(type 'exit' to exit the game): ").lower()
        if user_choice in game_choices:
            break
        elif user_choice == 'exit':
            return "Thanks for playing!"
        print("Oops!! Invalid choice.")

    # randomly select the computer's choice
    comp_choice = random.choice(game_choices)

    # determine the user outcome of the game using a dictionary
    outcomes = {'rock': {'rock': 'tie', 'paper': 'lose', 'scissors': 'win'},
                'paper': {'rock': 'win', 'paper': 'tie', 'scissors': 'lose'},
                'scissors': {'rock': 'lose', 'paper': 'win', 'scissors': 'tie'}}
    result = outcomes[user_choice][comp_choice]

    # return the outcome of the game as a sentence
    str_result = f"You chose {user_choice}. The computer chose {comp_choice}. You {result}!"
    return str_result

