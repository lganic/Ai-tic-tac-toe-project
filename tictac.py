import tileGraphics # Personal graphics library, ignore it
from copyUtils import fastCopyMatrix as fcm # Self descriptive import
import math
import depth_calc # Used to calculate automatic tree depth
import time

from typing import Dict, Iterable, Iterator, List, Tuple, Union

CALCLIMIT = 1000000 # Maximum number of score calculations AI is allowed to perform per turn

# For info into how CALCLIMIT is used, see depth_calc.py

AI_Start = True # Should AI start the game, or player?

size = 5 # Size of board

class boardState:
    """
    Tracks positions on board, as well as current turn.
    Implements behavior to make algorithm implementation cleaner.
    """
    def __init__(self,oStart: bool = True, n: int = 3) -> None:
        "Inititalize board"
        self.state = oStart # Set starting state
        self.n = n # set board size
        row = [None] * n # create empty row
        self.board = [row.copy() for _ in range(n)] # Create empty matrix from row copys
    def render(self) -> None:
        "Render board onto graphics surface"
        g.fill(7) # Fill white background
        # Loop over all tiles
        for y in range(self.n):
            for x in range(self.n):
                # If tile is occupied, place sprite
                if self.board[y][x] == "X":
                    g.putSprite(x,y,xSprite)
                if self.board[y][x] == "O":
                    g.putSprite(x,y,oSprite)
        g.outline(0,thickness=5,width=self.n) # Create grid outlines
    def possibleMoves(self) -> Iterator[Tuple]:
        "Create an iterator that returns all possible move positions in (x,y) format"
        for y in range(self.n):
            for x in range(self.n):
                if self.board[y][x] is None:
                    yield (x,y)
    def copy(self) -> 'boardState':
        "Return copy of board object"
        newBoard = boardState(self.state, self.n) # Create new board object
        newBoard.board = fcm(self.board) # Copy contents between board
        return newBoard
    def set(self,x: int,y: int) -> None:
        "Set tile at X and Y coordinate to currently active player pawn"
        if self.board[y][x] is not None:
            raise IndexError("No room at the inn")
        self.board[y][x] = "XO"[self.state]
        self.state = not self.state
    def possibleBoards(self) -> Iterator[Tuple[Tuple, 'boardState']]:
        "Create iterator that yields all possible boards, with the applied move"
        for newX, newY in self.possibleMoves():
            newBoard = self.copy()
            newBoard.set(newX, newY)
            yield ((newX, newY), newBoard)
    def row(self, index: int) -> Tuple:
        "Return row of grid at index"
        return tuple(self.board[index])
    def column(self, index: int) -> Tuple:
        "Return column of grid at index"
        return tuple([self.board[y][index] for y in range(self.n)])
    def RCDsets(self) -> Tuple:
        "Return a tuple matrix of all rows, columns, and diagonals"
        rows = tuple([self.row(k) for k in range(self.n)])
        cols = tuple([self.column(k) for k in range(self.n)])
        diag1 = (tuple([self.board[k][k] for k in range(self.n)]),)
        diag2 = (tuple([self.board[k][self.n - k - 1] for k in range(self.n)]),)
        sets = rows + cols + diag1 + diag2
        return sets
    def score1(self, who : str) -> int:
        "Score the board for a single player"
        def singleScore(set, who):
            "Score single set"
            nWho = chr(167 - ord(who)) # Get opposing player str using ASCII magic
            if nWho in set: # If enemy in set, no win is possible so the score is 0
                return 0
            return set.count(who) ** 2
        sets = self.RCDsets() # Get all sets
#        sign = 1
#        if who == "X":
#            sign = -1
        return sum([singleScore(s, who) for s in sets]) # Return sum of score for all sets
    def score(self):
        "Return the total score for the entire board"
        # Get the score for the AI and subtract the score for the player
        return self.score1("O") - self.score1("X")
    def filled(self) -> bool:
        "Check if board is filled"
        for y in range(self.n):
            for x in range(self.n):
                if self.board[y][x] is None:
                    return False
        return True
    def gamestate(self) -> str:
        """Check game state

        None -> Game not finished
        T -> Tie
        O -> O wins
        X -> X wins
        """
        sets = self.RCDsets()
        # Loop over all sets
        for s in sets:
            if s[0] is not None and s.count(s[0]) == self.n:
                return s[0]
        if self.filled():
            return "T"
        return None
    def print(self):
        "Print current board in console"
        # This method uses a LOT of list comprehension
        # Read at your own risk
        map = {None: " ", "X":"X", "O":"O"}
        k = ("\n"+"*".join(['-']*self.n)+"\n")
        print(k.join(["|".join([map[c] for c in r]) for r in self.board]))
        print("")

def minimax_alpha_beta(board, limit, alpha=-math.inf, beta=math.inf, maximizingPlayer=True):
    "Find best move from board using minimax and alpha beta pruning"
    # Check the current game state to see if it is a terminal state (win, lose, draw).
    gs = board.gamestate()
    if gs is not None:
        # If the game state indicates a win for 'X', return a negative score.
        if gs == "X":
            return -(board.n ** 3), board
        # If the game state indicates a win for 'O', return a positive score.
        elif gs == "O":
            return board.n ** 3, board
        # If the game is a draw, return 0.
        return 0, board

    # If the depth limit is reached, return the heuristic score of the board.
    if limit == 0:
        return board.score(), board

    # If it is the maximizer's turn.
    if maximizingPlayer:
        # Initialize the maximum evaluation score to negative infinity.
        maxEval = -math.inf
        # Initialize the board associated with the maximum evaluation.
        maxBoard = None
        # Explore all possible moves.
        for _, newBoard in board.possibleBoards():
            # Recursively call the minimax function for the child node (opponent's turn).
            eval, _ = minimax_alpha_beta(newBoard, limit - 1, alpha, beta, False)
            # Update the maximum score and board if a better evaluation is found.
            if eval > maxEval:
                maxEval = eval
                maxBoard = newBoard
            # Update the alpha value.
            alpha = max(alpha, eval)
            # Perform alpha-beta pruning.
            if beta <= alpha:
                break
        # Return the maximum evaluation and associated board.
        return maxEval, maxBoard
    else:
        # Initialize the minimum evaluation score to infinity.
        minEval = math.inf
        # Initialize the board associated with the minimum evaluation.
        minBoard = None
        # Explore all possible moves.
        for _, newBoard in board.possibleBoards():
            # Recursively call the minimax function for the child node (opponent's turn).
            eval, _ = minimax_alpha_beta(newBoard, limit - 1, alpha, beta, True)
            # Update the minimum score and board if a better evaluation is found.
            if eval < minEval:
                minEval = eval
                minBoard = newBoard
            # Update the beta value.
            beta = min(beta, eval)
            # Perform alpha-beta pruning.
            if beta <= alpha:
                break
        # Return the minimum evaluation and associated board.
        return minEval, minBoard

# Create graphics surface at 50% of screen size
g = tileGraphics.graphics(2 * size,size, roughPercent = .5,fps = 60)

# Load sprites
xSprite = tileGraphics.sprite(g,"x.png")
oSprite = tileGraphics.sprite(g,"o.png")

# Load sounds
playerSound = tileGraphics.sound('m1.wav')
aiSound = tileGraphics.sound('m2.wav')

# Define text box
textbox = tileGraphics.textBox(g, size, 0, size, size, nRows = 3, center = True, textColor = (0,0,0), backgroundColor=(255,255,255))

# Set window params
g.setName("TIC-TAC-TOE")
g.setIcon('x.png')

# Create board
board = boardState(AI_Start, n = size)

while True: # This loop gets broken by tileGraphics, so no worries about inf loop
    # Loop till game finished
    while board.gamestate() is None:
        if board.state: # If Ai move
            # Get number of possible move (This is the size of the root of the tree)
            nPossible = len(list(board.possibleBoards()))
            # Calculate the maximum depth using only CALCLIMIT calculations
            depthLimit = depth_calc.calc_tree_depth(nPossible, CALCLIMIT)
            # Graphics stuff
            g.putTextArray(textbox, ['AI move','',f'Limit: {depthLimit}'])
            g.update()
            # Use minimax to find best move using calculated depth limit
            _, board = minimax_alpha_beta(board, depthLimit)
            aiSound.play()
        else:
            # Take player input
            x = 99
            y = 99
            # Loop till valid square selected
            while x >= size or y >= size:
                # Loop till clicked
                while not g.checkClick():
                    board.render()
                    g.putTextArray(textbox, ["Player",'Move'])
                    g.update()
                # Loop till click is released
                while g.checkClick():
                    g.update()
                # Get tile that mouse landed on
                x, y = g.mouseTile()
                # Check if tile is on board
                if x < size and y < size and board.board[y][x] is not None:
                    x = 99
                    y = 99
            # Set board at location
            board.set(x,y)
            board.render()
            g.update()
            playerSound.play()
            time.sleep(.1)
    # Play game over screen
    t = time.time()
    board.render()
    g.putTextArray(textbox,["GAME","OVER",{'T':'Tie!','O':'AI Wins!','X':'Player Wins!'}[board.gamestate()]])
    # Wait for 10 seconds, or till player click
    while time.time() - t < 10:
        g.update()
        if g.checkClick():
            break
    while g.checkClick():
        g.update()
    dialog = tileGraphics.textBox(g, 0, 0, size * 2, size, nRows = 3, rowBoxes = True, textColor = (0,0,0), backgroundColor=(255,255,255))
    g.putTextArray(dialog,["New game?","Yes, AI starts","Yes, player starts"])
    while not g.checkClick():
        g.update()
    while g.checkClick():
        g.update()
    y_mouse_tile = g.rawMouseTile()[1]
    AI_Start = int(3 * (y_mouse_tile / size)) == 1
    # Create new board
    board = boardState(AI_Start, n = size)
