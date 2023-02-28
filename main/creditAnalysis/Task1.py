import numpy as np
import pandas as pd
import random


def new_dataset(df,
                student_id,
                first_name_start_letter,
                last_name_start_letter,
                ordered_preferences):
    # df = pd.read_csv('understat_per_game')
    # student_id = '32655932'
    # first_name_start_letter = 'w'
    # last_name_start_letter = 'z'
    # ordered_preferences = ['La Liga', 'EPL', 'BundesLiga', 'Seire_A', 'Ligue 1', 'RFPL']
    l1 = first_name_start_letter
    l2 = last_name_start_letter
    seed1 = int(student_id[0:4])
    seed2 = int(student_id[5:])
    seed3 = int(student_id)

    while l1 == 'A' or 'B' or 'C' or 'D':
        league1 = 'Bundesliga'
        break
    while l1 == 'E' or 'F' or 'G' or 'H':
        league1 = 'EPL'
        break
    while l1 == 'I' or 'J' or 'K' or 'L':
        league1 = 'La_liga'
        break
    while l1 == 'M' or 'N' or 'O' or 'P':
        league1 = 'Ligue_1'
        break
    while l1 == 'Q' or 'R' or 'S' or 'T':
        league1 = 'RFPL'
        break
    while l1 == 'U' or 'V' or 'W' or 'X' or 'Y' or 'Z':
        league1 = 'Serie_A'
        break

    while l2 == 'A' or 'B' or 'C' or 'D':
        league2 = 'Serie_A'
        break
    while l2 == 'E' or 'F' or 'G' or 'H':
        league2 = 'Bundesliga'
        break
    while l2 == 'I' or 'J' or 'K' or 'L':
        league2 = 'EPL'
        break
    while l2 == 'M' or 'N' or 'O' or 'P':
        league2 = 'La_liga'
        break
    while l2 == 'Q' or 'R' or 'S' or 'T':
        league2 = 'Ligue_1'
        break
    while l2 == 'U' or 'V' or 'W' or 'X' or 'Y' or 'Z':
        league2 = 'RFPL'
        break

    if league1 == league2:
        np.random.seed(seed1)
        league2 = np.random.choice(ordered_preferences.remove(league1))

    if ordered_preferences[0] != league1 and ordered_preferences[0] != league2:
        league3 = ordered_preferences[0]
    elif ordered_preferences[1] != league1 and ordered_preferences[1] != league2:
        league3 = ordered_preferences[1]
    else:
        league3 = ordered_preferences[2]

    if ordered_preferences[5] == league1:
        np.random.seed(seed2)
        a = np.random.uniform()
        if a < 0.5:
            a = 0.0
        else:
            a = 1.0
            league1 = np.random.choice(ordered_preferences.remove(league1, league2, league3))
    elif ordered_preferences[5] == league2:
        np.random.seed(seed2)
        b = np.random.uniform()
        if b < 0.5:
            b = 0.0
        else:
            b = 1.0
            league2 = np.random.choice(ordered_preferences.remove(league2, league1, league3))

    subset = df[(df["league"] == league1) | (df["league"] == league2) | (df["league"] == league3)]
    subset.reset_index(drop=True)
    np.random.seed(seed3)
    c = np.random.randint(0, 500)
    list1 = list(df.index)
    list2 = random.sample(list1, c)
    my_dataset = subset.drop(list2)
    return my_dataset


if __name__ == "__main__":
    df = pd.read_csv('understat_per_game.csv')
    student_id = '32655932'
    first_name_start_letter = 'W'
    last_name_start_letter = 'Z'
    ordered_preferences = ['La_liga', 'EPL', 'BundesLiga', 'Seire_A', 'Ligue_1', 'RFPL']
    new_dataset(df, student_id, first_name_start_letter, last_name_start_letter, ordered_preferences)


