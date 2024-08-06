from enum import Enum, auto
import random
from IPython.display import clear_output

class PieceType(Enum):
    PAWN = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK = auto()
    QUEEN = auto()
    KING = auto()

class Color(Enum):
    WHITE = auto()
    BLACK = auto()

class Piece:
    def __init__(self, piece_type, color):
        self.piece_type = piece_type
        self.color = color

class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Color.WHITE
        self.initialize_board()

    def initialize_board(self):
        # Set up pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.WHITE)
            self.board[6][col] = Piece(PieceType.PAWN, Color.BLACK)

        # Set up other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                       PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.WHITE)
            self.board[7][col] = Piece(piece_type, Color.BLACK)

    def display(self):
        piece_symbols = {
            (PieceType.PAWN, Color.WHITE): '♙', (PieceType.PAWN, Color.BLACK): '♟',
            (PieceType.KNIGHT, Color.WHITE): '♘', (PieceType.KNIGHT, Color.BLACK): '♞',
            (PieceType.BISHOP, Color.WHITE): '♗', (PieceType.BISHOP, Color.BLACK): '♝',
            (PieceType.ROOK, Color.WHITE): '♖', (PieceType.ROOK, Color.BLACK): '♜',
            (PieceType.QUEEN, Color.WHITE): '♕', (PieceType.QUEEN, Color.BLACK): '♛',
            (PieceType.KING, Color.WHITE): '♔', (PieceType.KING, Color.BLACK): '♚'
        }

        print("   a  b  c  d  e  f  g  h")  # Column labels
        print(" +--+--+--+--+--+--+--+--+")  # Top border

        for row in range(7, -1, -1):
            print(f"{row + 1}|", end="")  # Row label and left border
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    print(f"{piece_symbols[(piece.piece_type, piece.color)]}", end="|")
                else:
                    print("  ", end="|")  # Empty square
            print(f"{row + 1}")  # Right border and row label
            print(" +--+--+--+--+--+--+--+--+")  # Horizontal separator

        print("   a  b  c  d  e  f  g  h")  # Column labels

    def is_valid_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        piece = self.board[from_row][from_col]

        if not piece or piece.color != self.current_player:
            return False

        if piece.piece_type == PieceType.PAWN:
            return self.is_valid_pawn_move(from_square, to_square)
        elif piece.piece_type == PieceType.KNIGHT:
            return self.is_valid_knight_move(from_square, to_square)
        elif piece.piece_type == PieceType.BISHOP:
            return self.is_valid_bishop_move(from_square, to_square)
        elif piece.piece_type == PieceType.ROOK:
            return self.is_valid_rook_move(from_square, to_square)
        elif piece.piece_type == PieceType.QUEEN:
            return self.is_valid_queen_move(from_square, to_square)
        elif piece.piece_type == PieceType.KING:
            return self.is_valid_king_move(from_square, to_square)

        return False

    def is_valid_pawn_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        piece = self.board[from_row][from_col]
        direction = 1 if piece.color == Color.WHITE else -1

        # Moving forward
        if from_col == to_col and self.board[to_row][to_col] is None:
            if to_row == from_row + direction:
                return True
            if (from_row == 1 and piece.color == Color.WHITE) or (from_row == 6 and piece.color == Color.BLACK):
                if to_row == from_row + 2 * direction and self.board[from_row + direction][from_col] is None:
                    return True

        # Capturing
        if abs(from_col - to_col) == 1 and to_row == from_row + direction:
            if self.board[to_row][to_col] is not None and self.board[to_row][to_col].color != piece.color:
                return True

        return False

    def is_valid_knight_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        if abs(from_row - to_row) == 2 and abs(from_col - to_col) == 1:
            return True
        if abs(from_row - to_row) == 1 and abs(from_col - to_col) == 2:
            return True
        return False

    def is_valid_bishop_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        if abs(from_row - to_row) == abs(from_col - to_col):
            # Check if the path is clear
            step_row = 1 if to_row > from_row else -1
            step_col = 1 if to_col > from_col else -1
            for i in range(1, abs(to_row - from_row)):
                if self.board[from_row + i * step_row][from_col + i * step_col] is not None:
                    return False
            return True
        return False

    def is_valid_rook_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        if from_row == to_row or from_col == to_col:
            # Check if the path is clear
            if from_row == to_row:
                step = 1 if to_col > from_col else -1
                for i in range(from_col + step, to_col, step):
                    if self.board[from_row][i] is not None:
                        return False
            else:
                step = 1 if to_row > from_row else -1
                for i in range(from_row + step, to_row, step):
                    if self.board[i][from_col] is not None:
                        return False
            return True
        return False

    def is_valid_queen_move(self, from_square, to_square):
        # Queen moves like both a rook and a bishop
        return self.is_valid_rook_move(from_square, to_square) or self.is_valid_bishop_move(from_square, to_square)

    def is_valid_king_move(self, from_square, to_square):
        from_row, from_col = from_square
        to_row, to_col = to_square
        if abs(from_row - to_row) <= 1 and abs(from_col - to_col) <= 1:
            return True
        return False
    
    def make_move(self, from_square, to_square):
        if self.is_valid_move(from_square, to_square):
            piece = self.board[from_square[0]][from_square[1]]
            self.board[to_square[0]][to_square[1]] = piece
            self.board[from_square[0]][from_square[1]] = None
            self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
            return True
        return False

class RandomAI:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        valid_moves = self.get_all_valid_moves(board)
        return random.choice(valid_moves) if valid_moves else None

    def get_all_valid_moves(self, board):
        valid_moves = []
        for from_row in range(8):
            for from_col in range(8):
                piece = board.board[from_row][from_col]
                if piece and piece.color == self.color:
                    for to_row in range(8):
                        for to_col in range(8):
                            # Ensure the target square is either empty or occupied by an opponent's piece
                            target_piece = board.board[to_row][to_col]
                            if board.is_valid_move((from_row, from_col), (to_row, to_col)):
                                if not target_piece or target_piece.color != self.color:
                                    valid_moves.append(((from_row, from_col), (to_row, to_col)))
        return valid_moves


class MaterialCountAI:
    def __init__(self, color):
        self.color = color
        self.piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0  # King is invaluable for the game's sake
        }

    def evaluate_board(self, board):
        score = 0
        for row in board.board:
            for piece in row:
                if piece:
                    value = self.piece_values[piece.piece_type]
                    if piece.color == self.color:
                        score += value
                    else:
                        score -= value
        return score

    def get_move(self, board):
        valid_moves = self.get_all_valid_moves(board)
        best_move = None
        best_score = float('-inf') if self.color == Color.WHITE else float('inf')

        for move in valid_moves:
            board_copy = self.make_hypothetical_move(board, move)
            score = self.evaluate_board(board_copy)

            if self.color == Color.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def make_hypothetical_move(self, board, move):
        board_copy = ChessBoard()
        board_copy.board = [row[:] for row in board.board]
        from_square, to_square = move
        board_copy.make_move(from_square, to_square)
        return board_copy

    def get_all_valid_moves(self, board):
        valid_moves = []
        for from_row in range(8):
            for from_col in range(8):
                piece = board.board[from_row][from_col]
                if piece and piece.color == self.color:
                    for to_row in range(8):
                        for to_col in range(8):
                            if board.is_valid_move((from_row, from_col), (to_row, to_col)):
                                valid_moves.append(((from_row, from_col), (to_row, to_col)))
        return valid_moves


class MinimaxAI:
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth
        self.piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0  # The King is invaluable for the game's sake
        }

    def evaluate_board(self, board):
        score = 0
        for row in board.board:
            for piece in row:
                if piece:
                    value = self.piece_values[piece.piece_type]
                    if piece.color == self.color:
                        score += value
                    else:
                        score -= value
        return score

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.evaluate_board(board)

        valid_moves = self.get_all_valid_moves(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                board_copy = self.make_hypothetical_move(board, move)
                eval = self.alpha_beta(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                board_copy = self.make_hypothetical_move(board, move)
                eval = self.alpha_beta(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_move(self, board):
        valid_moves = self.get_all_valid_moves(board)
        best_move = None
        best_score = float('-inf') if self.color == Color.WHITE else float('inf')

        print("AI is considering the following moves:")
        for move in valid_moves:
            board_copy = self.make_hypothetical_move(board, move)
            score = self.alpha_beta(board_copy, self.depth - 1, float('-inf'), float('inf'), self.color == Color.WHITE)

            print(f"Move {move}: Evaluated score = {score}")

            if self.color == Color.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        if best_move:
            print(f"AI selected move: {best_move} with score {best_score}")
        else:
            print("AI could not find a valid move")

        return best_move

    def make_hypothetical_move(self, board, move):
        board_copy = ChessBoard()
        board_copy.board = [row[:] for row in board.board]
        from_square, to_square = move
        board_copy.make_move(from_square, to_square)
        return board_copy

    def get_all_valid_moves(self, board):
        valid_moves = []
        for from_row in range(8):
            for from_col in range(8):
                piece = board.board[from_row][from_col]
                if piece and piece.color == self.color:
                    for to_row in range(8):
                        for to_col in range(8):
                            if board.is_valid_move((from_row, from_col), (to_row, to_col)):
                                # Ensure the move doesn't result in a piece moving to a square occupied by its own side
                                target_piece = board.board[to_row][to_col]
                                if not target_piece or target_piece.color != self.color:
                                    valid_moves.append(((from_row, from_col), (to_row, to_col)))
                                    print(f"Valid move found: {from_row, from_col} to {to_row, to_col}")  # Debug line

        if not valid_moves:
            print("No valid moves found.")
        return valid_moves




def play_game_with_ai():
    board = ChessBoard()
    # ai = MaterialCountAI(Color.BLACK)
    ai = MinimaxAI(Color.BLACK, depth=2)

    while True:
        board.display()
        if board.current_player == Color.WHITE:
            print("Your turn (White)")
            from_square = input("Enter the square to move from (e.g., e2): ")
            to_square = input("Enter the square to move to (e.g., e4): ")
            
            from_col, from_row = ord(from_square[0]) - ord('a'), int(from_square[1]) - 1
            to_col, to_row = ord(to_square[0]) - ord('a'), int(to_square[1]) - 1

            if board.make_move((from_row, from_col), (to_row, to_col)):
                print("Move successful")
            else:
                print("Invalid move")
                continue
        else:
            print("AI's turn (Black)")
            ai_move = ai.get_move(board)
            if ai_move:
                from_square, to_square = ai_move
                board.make_move(from_square, to_square)
                print(f"AI moved from {chr(from_square[1] + ord('a'))}{from_square[0] + 1} to {chr(to_square[1] + ord('a'))}{to_square[0] + 1}")
            else:
                print("AI has no valid moves")
                break

if __name__ == "__main__":
    play_game_with_ai()