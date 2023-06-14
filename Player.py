import numpy as np


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.depth_counter = 0
        self.depth = 6
        self.alpha = -np.inf
        self.beta = np.inf
        self.is_max_node = 1
        self.is_expectimax = 0


    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        def max_value(state, alpha, beta, depth):
            successors_array = self.get_successors(state)
            highest_value = -np.inf
            highest_value_column = -1
            depth = depth + 1

            if depth == self.depth or len(successors_array) == 0:
                return self.evaluation_function(state)

            # Loop through all the successors and determine a value from them
            for i in range(len(successors_array)):
                # Change self_is_mode so it changes to min node when traversing through tree
                self.is_max_node = 0

                # If the successor is -1, which means column is full, just continue to next iteration
                if len(successors_array[i]) < 2:
                    continue

                # Grabs successor value
                successor_value = min_value(successors_array[i], alpha, beta, depth)

                # If the successor value is greater than our current highest value, we'll save the column
                # associated with the new high value
                if successor_value > highest_value:
                    highest_value_column = i

                highest_value = max(highest_value, successor_value)

                if highest_value >= beta:
                    return highest_value
                alpha = max(alpha, highest_value)

            # When depth hits 0, we're back at the root so return the column associated with the highest valued node
            if depth == 1:
                self.alpha = -np.inf
                self.beta = np.inf
                return highest_value_column

            # If we're not at the root node (which is at depth 0), return the value of the node
            return highest_value

        def min_value(state, alpha, beta, depth):
            successors_array = self.get_successors(state)
            depth = depth + 1

            if depth == self.depth or len(successors_array) == 0:
                return self.evaluation_function(state)

            lowest_value = np.inf

            # Loop through all the successors and determine a value from them
            for i in range(len(successors_array)):
                # Change self_is_mode so it changes to max node when traversing through tree
                self.is_max_node = 1

                # If the successor is -1, which means column is full, just continue to next iteration
                if len(successors_array[i]) < 2:
                    continue

                lowest_value = min(lowest_value, max_value(successors_array[i], alpha, beta, depth))
                if lowest_value <= alpha:
                    return lowest_value
                beta = min(beta, lowest_value)

            return lowest_value

        return max_value(board, self.alpha, self.beta, 0)

        # raise NotImplementedError('Whoops I don\'t know what to do')

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        # # Checks to see if it is a leaf node or it has reached the depth limit
        # if self.depth_counter == self.depth or len(successors_array) == 0:
        #     # Decrease the depth counter by as we will be traversing the tree back up again
        #     self.depth_counter = self.depth_counter - 1
        #
        #     return self.evaluation_function(board)

        def max_value(state, depth):
            depth = depth + 1

            # Get successors and one level to the depth level
            successors_array = self.get_successors(state)

            v = -np.inf
            highest_value_column = -1

            # Checks to see if it is a leaf node or it has reached the depth limit
            if depth == self.depth or len(successors_array) == 0:
                return self.evaluation_function(state)

            # Loop through all the successors and determine a value from them
            for i in range(len(successors_array)):
                # If the successor is -1, which means column is full, just continue to next iteration
                if len(successors_array[i]) < 2:
                    continue

                # Grabs successor value
                successor_value = get_exp_value(state, depth)

                # If the successor value is greater than our current highest value, we'll save the column
                # associated with the new high value
                if successor_value > v:
                    highest_value_column = i

                v = max(v, successor_value)

            if depth == 1:
                return highest_value_column

            # If we're not at the root node (which is at depth 0), return the value of the node
            return v

        def get_exp_value(state, depth):
            depth = depth + 1

            # Get successors and one level to the depth level
            successors_array = self.get_successors(state)

            v = 0

            # Checks to see if it is a leaf node or it has reached the depth limit
            if depth == self.depth or len(successors_array) == 0:
                return self.evaluation_function(state)

            # Loop through all the successors and determine a value from them
            for i in range(len(successors_array)):
                # Change self_is_mode so it changes to min node when traversing through tree
                self.is_expectimax = 0

                # If the successor is -1, which means column is full, just continue to next iteration
                if len(successors_array[i]) < 2:
                    continue

                p = 1/7

                v += p * max_value(successors_array[i], depth)

            # If we're not at the root node (which is at depth 0), return the value of the node
            return v

        return max_value(board, 0)

        #raise NotImplementedError('Whoops I don\'t know what to do')

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        # Grabs opponent number
        if self.player_number == 1:
            opponent = 2
        else:
            opponent = 1

        player_win_str = '{0}{0}{0}{0}'.format(self.player_number)  # checks for 4 in a row
        player_three_str = '{0}{0}{0}{1}'.format(self.player_number, 0)  # checks for 3 in a row
        player_three_2_str = '{1}{0}{0}{0}'.format(self.player_number, 0)  # checks for 3 in a row
        player_three_separated_1_str = '{0}{0}{1}{0}'.format(self.player_number, 0)  # checks for 3 with 1 gap
        player_three_separated_2_str = '{0}{1}{0}{0}'.format(self.player_number, 0)  # checks for 3 with 1 gap
        player_two_str = '{0}{0}{1}{1}'.format(self.player_number, 0)  # checks for 2 in a row
        player_two_2_str = '{1}{1}{0}{0}'.format(self.player_number, 0)  # checks for 2 in a row
        player_two_separated_str = '{0}{1}{0}{1}'.format(self.player_number, 0)  # checks for 2 with 1 gap
        player_two_separated_2_str = '{1}{0}{1}{0}'.format(self.player_number, 0)  # checks for 2 with 1 gap

        opponent_3_1_str = '{0}{0}{0}{1}'.format(opponent, self.player_number)  # checks if opponent has 3 in a row
        opponent_3_2_str = '{1}{0}{0}{0}'.format(opponent, self.player_number)  # checks if opponent has 3 in a row
        opponent_3_3_str = '{0}{1}{0}{0}'.format(opponent, self.player_number)  # checks if opponent has 3 with a gap
        opponent_3_4_str = '{0}{0}{1}{0}'.format(opponent, self.player_number)  # checks if opponent has 3 with a gap
        opponent_2_1_str = '{0}{0}{1}{2}'.format(opponent, self.player_number, 0)  # checks if opponent has 2 in a row
        opponent_2_2_str = '{2}{1}{0}{0}'.format(opponent, self.player_number, 0)  # checks if opponent has 2 in a row
        # opponent_2_1_str = '{0}{0}{1}{1}'.format(opponent, 0)  # checks for 4 in a row
        # opponent_2_2_str = '{1}{1}{0}{0}'.format(opponent, 0)  # checks for 4 in a row
        # opponent_2_3_str = '{1}{0}{1}{0}'.format(opponent, 0)  # checks for 4 in a row
        # opponent_2_4_str = '{0}{1}{0}{1}'.format(opponent, 0)  # checks for 4 in a row

        board = board
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            score = 0
            for row in b:
                if player_win_str in to_str(row):
                    score = score + 8100000

                if player_three_str in to_str(row):
                    score = score + 20

                if player_three_2_str in to_str(row):
                    score = score + 20

                if player_three_separated_1_str in to_str(row):
                    score = score + 20

                if player_three_separated_2_str in to_str(row):
                    score = score + 20

                if player_two_str in to_str(row):
                    score = score + 5

                if player_two_2_str in to_str(row):
                    score = score + 5

                if player_two_separated_str in to_str(row):
                    score = score + 3

                if player_two_separated_2_str in to_str(row):
                    score = score + 3

                # Gives high points to player for blocking opponent if they have a potential winning move
                if opponent_3_1_str in to_str(row):
                    score = score + 200

                if opponent_3_2_str in to_str(row):
                    score = score + 200

                if opponent_3_3_str in to_str(row):
                    score = score + 200

                if opponent_3_4_str in to_str(row):
                    score = score + 200

                if opponent_2_1_str in to_str(row):
                    score = score + 7

                if opponent_2_2_str in to_str(row):
                    score = score + 7

            return score

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            score = 0
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    score = score + 8100000

                if player_three_str in to_str(root_diag):
                    score = score + 20

                if player_three_2_str in to_str(root_diag):
                    score = score + 20

                if player_three_separated_1_str in to_str(root_diag):
                    score = score + 20

                if player_three_separated_2_str in to_str(root_diag):
                    score = score + 20

                if player_two_str in to_str(root_diag):
                    score = score + 5

                if player_two_2_str in to_str(root_diag):
                    score = score + 5

                if player_two_separated_str in to_str(root_diag):
                    score = score + 3

                if player_two_separated_2_str in to_str(root_diag):
                    score = score + 3

                if opponent_3_1_str in to_str(root_diag):
                    score = score + 200

                if opponent_3_2_str in to_str(root_diag):
                    score = score + 200

                if opponent_3_3_str in to_str(root_diag):
                    score = score + 200

                if opponent_3_4_str in to_str(root_diag):
                    score = score + 200

                if opponent_2_1_str in to_str(root_diag):
                    score = score + 7

                if opponent_2_2_str in to_str(root_diag):
                    score = score + 7

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            score = score + 8100000

                        if player_three_str in diag:
                            score = score + 20

                        if player_three_2_str in diag:
                            score = score + 20

                        if player_three_separated_1_str in diag:
                            score = score + 20

                        if player_three_separated_2_str in diag:
                            score = score + 20

                        if player_two_str in diag:
                            score = score + 5

                        if player_two_2_str in diag:
                            score = score + 5

                        if player_two_separated_str in diag:
                            score = score + 3

                        if player_two_separated_2_str in diag:
                            score = score + 3

                        if opponent_3_1_str in diag:
                            score = score + 200

                        if opponent_3_2_str in diag:
                            score = score + 200

                        if opponent_3_3_str in diag:
                            score = score + 200

                        if opponent_3_4_str in diag:
                            score = score + 200

                        if opponent_2_1_str in diag:
                            score = score + 7

                        if opponent_2_2_str in diag:
                            score = score + 7

            return score

        # return (check_horizontal(board) or
        #        check_verticle(board) or
        #        check_diagonal(board))

        return check_horizontal(board) + check_verticle(board) + check_diagonal(board)

    def get_successors(self, board):
        # Make an array of the successors
        successors = []

        # Only adds successors if the current node doesn't have a four in a row
        if self.evaluation_function(board) <= 8100000:
            # Make a copy of our current board
            copy_board = board.copy()

            # Looping through the whole board
            # j is row
            # i is column
            for i in range(0, 7):
                for j in range(6):
                    # Checks to see if column is filled
                    if board[j][i] != 0 and j == 0:
                        # Add a -1 to indicate the column is full
                        successors.append([-1])
                        break

                    # Inserts player_num on top of existing piece
                    if board[j][i] != 0:
                        copy_board[j - 1][i] = self.player_number

                        # Add successor to array of successors
                        successors.append(copy_board)

                        # Reset copy_board to the original board
                        copy_board = board.copy()

                        # copy_board[j - 1][i] = 0
                        break

                    # Inserts player_num at the bottom if column is empty
                    if j == 5 and board[j][i] == 0:
                        copy_board[j][i] = self.player_number

                        # Add successor to array of successors
                        successors.append(copy_board)

                        # Reset copy_board to the original board
                        copy_board = board.copy()

                        break

        return successors


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
