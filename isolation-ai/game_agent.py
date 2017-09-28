"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # return if you won or lost
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(game.get_opponent(player))
    distance = float(math.sqrt((player_location[0] - opponent_location[0]) ** 2 +
                               (player_location[1] - opponent_location[1]) ** 2))


    return 100*float(distance) + float(len(game.get_blank_spaces()))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # return if you won or lost
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_legal_moves = game.get_legal_moves(player)
    opponent_legal_moves = game.get_legal_moves(game.get_opponent(player))
    player_moves = len(player_legal_moves)
    opponent_moves = len(opponent_legal_moves)

    # detect game board edges
    edges = []
    rays = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    location = (0, 0)

    for dx, dy in rays:
        _x, _y = location
        while 0 <= _x + dx < game.width and 0 <= _y + dy < game.height:
            _x, _y = _x + dx, _y + dy
            location = (_x, _y)
            edges.append(location)

    w1 = 1
    for m in player_legal_moves:
        if m not in edges:
            w1 += 1

    w2 = 1
    for m in opponent_legal_moves:
        if m not in edges:
            w2 += 1

    return float(w1*player_moves - 2*w2*opponent_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # return if you won or lost
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves - opponent_moves)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if there are no legal moves return (-1,-1)
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)

        # assume the worst outcome and take a random first legal move
        best_score = float("-inf")
        # initialize the best move to randomly selected legal_move
        best_move = legal_moves[random.randint(0, len(legal_moves)) - 1]
        # find the best moves between all legal moves
        for m in legal_moves:
            v = self.min_value(game.forecast_move(m), depth-1)
            if v > best_score:
                best_score = v
                best_move = m

        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        return not bool(game.get_legal_moves())

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # we reached max depth or terminal state
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        # initialize value to worse score for MIN node
        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m), depth-1))
        return v


    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # we reached max depth or terminal state
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        # initialize value to worse score for MAX node
        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), depth-1))
        return v

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # initialize best move
        best_move = (-1, -1)
        # start with depth==1
        depth = 1
        # iterate until reaching timeout
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth+1
        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # if there are no legal moves return (-1,-1)
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        # assume the worst outcome and take a random first legal move
        best_score = float("-inf")
        # initialize best_move to randomly selected legal_move
        best_move = legal_moves[random.randint(0, len(legal_moves)) - 1]
        # find the best score for the next move and update alpha
        for m in legal_moves:
            v = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            if v > best_score:
                best_score = v
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    def terminal_test(self, game):
        return not bool(game.get_legal_moves())

    def min_value(self, game, depth, alpha, beta):
        # timeout expeirec
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # we reached max depth or terminal state
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        # initialize value to worse score for MIN node
        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m), depth-1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, game, depth, alpha, beta):
        # timeout expeirec
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # we reached max depth or terminal state
        if depth == 0 or self.terminal_test(game):
            return self.score(game, self)
        # initialize value to worse score for MAX node
        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), depth-1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
