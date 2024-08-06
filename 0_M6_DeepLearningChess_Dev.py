from enum import Enum, auto

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

def play_game():
    board = ChessBoard()
    while True:
        board.display()
        print(f"Current player: {board.current_player.name}")
        from_square = input("Enter the square to move from (e.g., e2): ")
        to_square = input("Enter the square to move to (e.g., e4): ")
        
        from_col, from_row = ord(from_square[0]) - ord('a'), int(from_square[1]) - 1
        to_col, to_row = ord(to_square[0]) - ord('a'), int(to_square[1]) - 1

        if board.make_move((from_row, from_col), (to_row, to_col)):
            print("Move successful")
        else:
            print("Invalid move")


def play_game_jupyter():
    board = ChessBoard()
    game_active = True
    
    while game_active:
        board.display()
        print(f"Current player: {board.current_player.name}")
        
        from_square = input("Enter the square to move from (e.g., e2): ")
        to_square = input("Enter the square to move to (e.g., e4): ")
        
        from_col, from_row = ord(from_square[0]) - ord('a'), int(from_square[1]) - 1
        to_col, to_row = ord(to_square[0]) - ord('a'), int(to_square[1]) - 1
        
        if board.make_move((from_row, from_col), (to_row, to_col)):
            print("Move successful")
        else:
            print("Invalid move")
        
        # You might want to add some condition to break the loop, for example, checking for a checkmate or stalemate
        # or by limiting the number of moves:
        continue_game = input("Continue playing? (yes/no): ")
        if continue_game.lower() != "yes":
            game_active = False


if __name__ == "__main__":
    board = ChessBoard()
    board.display()