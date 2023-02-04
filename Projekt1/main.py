import argparse
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

BASE_MATRIX_CORRECTION = .1
MAP_FOR_THE_WIN = {
    'R': 'P'
    , 'P': 'S'
    , 'S': 'R'}
MAP_FOR_POSITION = {
    'R': 0
    , 'P': 1
    , 'S': 2
}


def main(args):
    _numer_of_games = args.mandatory_argument
    _matrix_correction = args.optional_argument

    _game_results = []

    start = ['R', 'P', 'S']
    # start probability
    p_start = np.array([0.4, 0.4, 0.2])

    # player next moves
    p_rest_sel = ['R', 'P', 'S']
    # player next moves matrix
    p_rest_prob = np.array([[.3, .3, .4],
                            [.3, .4, .3],
                            [.4, .3, .3]])
    # matrix
    wyst = np.array([[.33, .33, .33],
                     [.33, .33, .33],
                     [.33, .33, .33]])

    initial = np.random.choice(start, replace=True, p=p_start)

    last_chosen_move = ''
    for i in range(_numer_of_games):
        print(f'Game number: {i + 1}')

        if i == 0:
            player_selection = initial
            # in first iteration last chosen move == player selection
            last_chosen_move = player_selection

            # predicate player selection
            pred = np.random.choice(p_rest_sel, p=wyst[MAP_FOR_POSITION.get(last_chosen_move)] / sum(
                wyst[MAP_FOR_POSITION.get(last_chosen_move)]))

            # prepare response:
            response = MAP_FOR_THE_WIN.get(pred)

            # result of the game
            result_for_ai = _get_game_result(player_selection, response)

            # save game result
            _game_results.append(result_for_ai)

            # modify matrix
            wyst = _get_modified_matrix(wyst, _matrix_correction, last_chosen_move, player_selection, result_for_ai)

            # print game results
            print(f'AI Score: {_game_results[-1]}')
            print_results(player_selection, _get_game_result(player_selection, initial), wyst)

        # second and further iterations
        else:
            player_selection = np.random.choice(p_rest_sel, p=p_rest_prob[MAP_FOR_POSITION.get(last_chosen_move)])

            # predicate player selection
            pred = np.random.choice(p_rest_sel, p=wyst[MAP_FOR_POSITION.get(last_chosen_move)] / sum(
                wyst[MAP_FOR_POSITION.get(last_chosen_move)]))

            # prepare response:
            response = MAP_FOR_THE_WIN.get(pred)

            # result of the game
            result_for_ai = _get_game_result(player_selection, response)

            # save game result
            _game_results.append((_game_results[i - 1] + result_for_ai))

            # modify matrix
            wyst = _get_modified_matrix(wyst, _matrix_correction, last_chosen_move, player_selection, result_for_ai)

            # remember player selection
            last_chosen_move = player_selection

            # print game results
            print(f'AI Score: {_game_results[-1]}')
            print_results(player_selection, _get_game_result(player_selection, response), wyst)

    # Drew graph
    plt.plot(_game_results, linestyle='--', color='black', marker='o', mfc='red', mec='k')
    plt.ylabel("Game state")
    plt.xlabel("Number of games")
    plt.yticks(np.arange(min(_game_results), max(_game_results) + 1, 1.0))
    plt.grid(True)
    plt.title("HMM")
    plt.show()


def _get_game_result(_player_move, _ai_move):
    """
    Function provides game result from AI perspective.
    :param _player_move: char, ['R','P','S']
    :param _ai_move: char, ['R','P','S']
    :return: Game result [-1 lose | 0 draw | 1 win]
    """
    if _player_move == _ai_move:
        return 0

    if MAP_FOR_THE_WIN.get(_ai_move) == _player_move:
        return -1

    return 1


def _get_modified_matrix(matrix_to_modify, _matrix_correction, last_chosen_move, player_selection, game_result_for_ai):
    """
    Function modifies probability value in matrix based on last chosen move and player selection.
    Values position in matrix is found by const map (MAP_FOR_POSITION).

    If player won / draw:
        Value selected by player += _matrix_correction.
        Both other values -= (_matrix_correction / 2).
    If ai won:
        Value selected by player -= _matrix_correction.
        Both other values += (_matrix_correction / 2).

    If modified value == 1 or ==0 - no action for this value is performed.

    :param matrix_to_modify: 3x3 float matrix, range 0-1
    :param _matrix_correction: script const
    :param last_chosen_move: char, ['R','P','S']
    :param player_selection: char, ['R','P','S']
    :param game_result_for_ai:
    :return: modified 3x3 float matrix, range 0-1
    """
    modified_matrix = matrix_to_modify
    # find tables of possibilities for last played move
    last_sel_possibilities = modified_matrix[MAP_FOR_POSITION.get(last_chosen_move)]
    # find possibility of currently played move
    possibility_of_played_move = last_sel_possibilities[MAP_FOR_POSITION.get(player_selection)]

    # AI did not win
    if game_result_for_ai != 1:
        # add correction to played value
        last_sel_possibilities[MAP_FOR_POSITION.get(player_selection)] = min(1,
                                                                             possibility_of_played_move + _matrix_correction)
        # substract half of correction from both other values
        for j in MAP_FOR_POSITION.values():
            if MAP_FOR_POSITION.get(player_selection) != j:
                last_sel_possibilities[j] = max(0, last_sel_possibilities[j] - (_matrix_correction / 2))
    # AI won
    else:
        # substract correction to played value
        last_sel_possibilities[MAP_FOR_POSITION.get(player_selection)] = max(0,
                                                                             possibility_of_played_move - _matrix_correction)
        # add half of correction to both other values
        for j in MAP_FOR_POSITION.values():
            if MAP_FOR_POSITION.get(player_selection) != j:
                last_sel_possibilities[j] = min(1, last_sel_possibilities[j] + (_matrix_correction / 2))

    return modified_matrix


def print_results(player_selection, game_result, matrix):
    """
    Function prints information of game result and matrix details after modification.
    Results and colors:
        AI won - green,
        AI lost - red,
        AI draw - yellow.
    :param player_selection: char, ['R','P','S']
    :param game_result: int, [-1 lose | 0 draw | 1 win] from AI perspective
    :param matrix: 3x3 float matrix, range 0-1
    :return: None
    """
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if game_result == -1:
        print(colored(f'User chose: {player_selection} - ai lost', 'red'))
    elif game_result == 0:
        print(colored(f'User chose: {player_selection} - ai draw', 'yellow'))
    else:
        print(colored(f'User chose: {player_selection} - ai won', 'green'))

    print(
        f'R transition: R:{round(matrix[0][0], 2): .2f} P:{round(matrix[0][1], 2): .2f} S:{round(matrix[0][2], 2): .2f}')
    print(
        f'P transition: R:{round(matrix[1][0], 2): .2f} P:{round(matrix[1][1], 2): .2f} S:{round(matrix[1][2], 2): .2f}')
    print(
        f'S transition: R:{round(matrix[2][0], 2): .2f} P:{round(matrix[2][1], 2): .2f} S:{round(matrix[2][2], 2): .2f}')
    print('--------------------------------')


def parse_arguments():
    """
    This is a function that parses arguments from command line.

    :param: None
    :returns: Namespace storing all arguments from command line
    """
    parser = argparse.ArgumentParser(
        description='This is just a boilerplate Hello World')
    parser.add_argument('-m',
                        '--mandatory_argument',
                        type=int,
                        required=True,
                        help='Mandatory - numbers of games to process')
    parser.add_argument('-o',
                        '--optional_argument',
                        type=float,
                        default=BASE_MATRIX_CORRECTION,
                        help='Just an optional arg, dont mind me')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
