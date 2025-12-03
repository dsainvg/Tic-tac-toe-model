import pygame
import pickle
import os
import sys
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 700
BOARD_SIZE = 3
CELL_SIZE = 180
BOARD_OFFSET_X = 30
BOARD_OFFSET_Y = 100
LINE_WIDTH = 5
FPS = 60

# Colors (Modern, vibrant palette)
BG_COLOR = (15, 23, 42)  # Dark slate
GRID_COLOR = (71, 85, 105)  # Slate
X_COLOR = (248, 113, 113)  # Vibrant red
O_COLOR = (96, 165, 250)  # Vibrant blue
TEXT_COLOR = (241, 245, 249)  # Light slate
ACCENT_COLOR = (168, 85, 247)  # Purple
WIN_LINE_COLOR = (34, 197, 94)  # Green
DRAW_COLOR = (234, 179, 8)  # Yellow

# Fonts
TITLE_FONT = pygame.font.Font(None, 48)
INFO_FONT = pygame.font.Font(None, 32)
SMALL_FONT = pygame.font.Font(None, 24)

# Create display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe History Replay")

# Load game histories
def load_game_histories():
    file_path = os.path.join('histories', 'game_histories.pkl')
    try:
        with open(file_path, 'rb') as f:
            game_histories = pickle.load(f)
        print(f"✓ Loaded {len(game_histories)} game histories!")
        return game_histories
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"Error loading game histories: {e}")
        return []

# Board class
class Board:
    def __init__(self):
        self.board = [0] * 9
        self.winner = None
        self.game_over = False
        
    def reset(self):
        self.board = [0] * 9
        self.winner = None
        self.game_over = False
    
    def make_move(self, position, player):
        if 0 <= position < 9 and self.board[position] == 0:
            self.board[position] = player
            return True
        return False
    
    def check_winner(self):
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != 0:
                return self.board[i], [(i, i+1, i+2)]
        
        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != 0:
                return self.board[i], [(i, i+3, i+6)]
        
        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return self.board[0], [(0, 4, 8)]
        if self.board[2] == self.board[4] == self.board[6] != 0:
            return self.board[2], [(2, 4, 6)]
        
        # Check draw
        if 0 not in self.board:
            return 0, []
        
        return None, []

# Drawing functions
def draw_gradient_bg():
    for y in range(WINDOW_HEIGHT):
        ratio = y / WINDOW_HEIGHT
        r = int(15 + (30 - 15) * ratio)
        g = int(23 + (41 - 23) * ratio)
        b = int(42 + (59 - 42) * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))

def draw_board():
    # Draw grid lines with glow effect
    for i in range(1, BOARD_SIZE):
        # Vertical lines
        x = BOARD_OFFSET_X + i * CELL_SIZE
        # Glow
        for offset in range(3, 0, -1):
            alpha_surface = pygame.Surface((LINE_WIDTH + offset * 2, CELL_SIZE * BOARD_SIZE), pygame.SRCALPHA)
            pygame.draw.line(alpha_surface, (*GRID_COLOR, 50), 
                           (LINE_WIDTH // 2 + offset, 0), 
                           (LINE_WIDTH // 2 + offset, CELL_SIZE * BOARD_SIZE), 
                           LINE_WIDTH + offset * 2)
            screen.blit(alpha_surface, (x - offset, BOARD_OFFSET_Y))
        # Main line
        pygame.draw.line(screen, GRID_COLOR, 
                        (x, BOARD_OFFSET_Y), 
                        (x, BOARD_OFFSET_Y + CELL_SIZE * BOARD_SIZE), 
                        LINE_WIDTH)
        
        # Horizontal lines
        y = BOARD_OFFSET_Y + i * CELL_SIZE
        # Glow
        for offset in range(3, 0, -1):
            alpha_surface = pygame.Surface((CELL_SIZE * BOARD_SIZE, LINE_WIDTH + offset * 2), pygame.SRCALPHA)
            pygame.draw.line(alpha_surface, (*GRID_COLOR, 50), 
                           (0, LINE_WIDTH // 2 + offset), 
                           (CELL_SIZE * BOARD_SIZE, LINE_WIDTH // 2 + offset), 
                           LINE_WIDTH + offset * 2)
            screen.blit(alpha_surface, (BOARD_OFFSET_X, y - offset))
        # Main line
        pygame.draw.line(screen, GRID_COLOR, 
                        (BOARD_OFFSET_X, y), 
                        (BOARD_OFFSET_X + CELL_SIZE * BOARD_SIZE, y), 
                        LINE_WIDTH)

def draw_x(row, col, animation_progress=1.0):
    center_x = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
    center_y = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
    offset = int(CELL_SIZE * 0.3 * animation_progress)
    
    # Glow effect
    for glow_offset in range(5, 0, -1):
        alpha_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.line(alpha_surface, (*X_COLOR, 30), 
                        (CELL_SIZE // 2 - offset, CELL_SIZE // 2 - offset),
                        (CELL_SIZE // 2 + offset, CELL_SIZE // 2 + offset),
                        LINE_WIDTH + glow_offset * 2)
        pygame.draw.line(alpha_surface, (*X_COLOR, 30),
                        (CELL_SIZE // 2 + offset, CELL_SIZE // 2 - offset),
                        (CELL_SIZE // 2 - offset, CELL_SIZE // 2 + offset),
                        LINE_WIDTH + glow_offset * 2)
        screen.blit(alpha_surface, (center_x - CELL_SIZE // 2, center_y - CELL_SIZE // 2))
    
    # Main X
    pygame.draw.line(screen, X_COLOR, 
                    (center_x - offset, center_y - offset),
                    (center_x + offset, center_y + offset),
                    LINE_WIDTH)
    pygame.draw.line(screen, X_COLOR,
                    (center_x + offset, center_y - offset),
                    (center_x - offset, center_y + offset),
                    LINE_WIDTH)

def draw_o(row, col, animation_progress=1.0):
    center_x = BOARD_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
    center_y = BOARD_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
    radius = int(CELL_SIZE * 0.3 * animation_progress)
    
    # Glow effect
    for glow_offset in range(5, 0, -1):
        alpha_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(alpha_surface, (*O_COLOR, 30), 
                         (CELL_SIZE // 2, CELL_SIZE // 2), 
                         radius + glow_offset, 
                         LINE_WIDTH + glow_offset)
        screen.blit(alpha_surface, (center_x - CELL_SIZE // 2, center_y - CELL_SIZE // 2))
    
    # Main O
    pygame.draw.circle(screen, O_COLOR, (center_x, center_y), radius, LINE_WIDTH)

def draw_pieces(board, animation_pos=None, animation_progress=1.0):
    for i, cell in enumerate(board.board):
        row = i // 3
        col = i % 3
        
        if cell == 1:  # X
            if animation_pos == i:
                draw_x(row, col, animation_progress)
            else:
                draw_x(row, col)
        elif cell == -1:  # O
            if animation_pos == i:
                draw_o(row, col, animation_progress)
            else:
                draw_o(row, col)

def draw_win_line(winning_combo):
    if not winning_combo:
        return
    
    positions = winning_combo[0]
    start_pos = positions[0]
    end_pos = positions[-1]
    
    start_row, start_col = start_pos // 3, start_pos % 3
    end_row, end_col = end_pos // 3, end_pos % 3
    
    start_x = BOARD_OFFSET_X + start_col * CELL_SIZE + CELL_SIZE // 2
    start_y = BOARD_OFFSET_Y + start_row * CELL_SIZE + CELL_SIZE // 2
    end_x = BOARD_OFFSET_X + end_col * CELL_SIZE + CELL_SIZE // 2
    end_y = BOARD_OFFSET_Y + end_row * CELL_SIZE + CELL_SIZE // 2
    
    # Glow effect
    for glow_offset in range(8, 0, -1):
        alpha_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(alpha_surface, (*WIN_LINE_COLOR, 40),
                        (start_x, start_y),
                        (end_x, end_y),
                        LINE_WIDTH * 2 + glow_offset * 2)
        screen.blit(alpha_surface, (0, 0))
    
    # Main line
    pygame.draw.line(screen, WIN_LINE_COLOR,
                    (start_x, start_y),
                    (end_x, end_y),
                    LINE_WIDTH * 2)

def draw_ui(game_info, move_num, total_moves, total_histories):
    # Title
    title_text = TITLE_FONT.render("History Replay", True, TEXT_COLOR)
    title_rect = title_text.get_rect(center=(WINDOW_WIDTH // 2, 40))
    screen.blit(title_text, title_rect)
    
    # Stats
    stats_y = BOARD_OFFSET_Y + CELL_SIZE * BOARD_SIZE + 30
    
    # Game info
    epoch_text = INFO_FONT.render(f"Epoch {game_info['epoch']} - Game #{game_info['board_idx']}", True, ACCENT_COLOR)
    epoch_rect = epoch_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y))
    screen.blit(epoch_text, epoch_rect)
    
    # Move counter
    move_text = SMALL_FONT.render(f"Move: {move_num}/{total_moves}", True, TEXT_COLOR)
    move_rect = move_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 35))
    screen.blit(move_text, move_rect)
    
    # Result
    result = game_info.get('result', 'ongoing')
    if result == 'win':
        winner = "O" if game_info['active_player'] == 1 else "X"
        result_color = O_COLOR if game_info['active_player'] == 1 else X_COLOR
        result_text = SMALL_FONT.render(f"Winner: {winner}", True, result_color)
    elif result == 'draw':
        result_text = SMALL_FONT.render("Result: Draw", True, DRAW_COLOR)
    else:
        result_text = SMALL_FONT.render("In Progress...", True, TEXT_COLOR)
    
    result_rect = result_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 65))
    screen.blit(result_text, result_rect)
    
    # Total games available
    total_text = SMALL_FONT.render(f"Histories: {total_histories}", True, GRID_COLOR)
    screen.blit(total_text, (10, WINDOW_HEIGHT - 30))
    
    # Controls hint
    controls_text = SMALL_FONT.render("SPACE: Pause | N: Next | R: Random | Q: Quit", True, GRID_COLOR)
    controls_rect = controls_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
    screen.blit(controls_text, controls_rect)

def main():
    clock = pygame.time.Clock()
    
    # Load game histories
    print("Loading game histories...")
    game_histories = load_game_histories()
    
    if not game_histories:
        print("No game histories found! Exiting...")
        pygame.quit()
        sys.exit()
    
    # Game state
    board = Board()
    current_game = None
    current_move_idx = 0
    
    # Animation state
    animating = False
    animation_pos = None
    animation_frame = 0
    animation_duration = 15  # frames
    
    # Timing
    move_delay = 800  # milliseconds between moves
    last_move_time = pygame.time.get_ticks()
    
    # Win animation
    show_win_line = False
    winning_combo = []
    
    # Control
    paused = False
    auto_next = True
    
    # Select random game
    def select_random_game():
        game = random.choice(game_histories)
        print(f"\n{'='*50}")
        print(f"Selected Game: Epoch {game['epoch']}, Board #{game['board_idx']}")
        print(f"Result: {game['result'].upper()}")
        print(f"Moves: {len(game['game_state'])}")
        print(f"{'='*50}")
        return game
    
    current_game = select_random_game()
    
    print("\n=== Controls ===")
    print("SPACE: Pause/Resume")
    print("N: Next game")
    print("R: Random game")
    print("Q/ESC: Quit\n")
    
    running = True
    game_finished = False
    
    while running:
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
                elif event.key == pygame.K_n:
                    # Next game
                    board.reset()
                    current_game = select_random_game()
                    current_move_idx = 0
                    show_win_line = False
                    game_finished = False
                    animating = False
                elif event.key == pygame.K_r:
                    # Random game
                    board.reset()
                    current_game = select_random_game()
                    current_move_idx = 0
                    show_win_line = False
                    game_finished = False
                    animating = False
        
        # Drawing
        draw_gradient_bg()
        draw_ui(current_game, current_move_idx, len(current_game['game_state']), len(game_histories))
        draw_board()
        
        # Animation progress
        if animating:
            animation_frame += 1
            progress = min(animation_frame / animation_duration, 1.0)
            draw_pieces(board, animation_pos, progress)
            
            if animation_frame >= animation_duration:
                animating = False
                animation_frame = 0
        else:
            draw_pieces(board)
        
        if show_win_line:
            draw_win_line(winning_combo)
        
        pygame.display.flip()
        
        if paused:
            continue
        
        # Game replay logic
        if not game_finished and not animating and current_time - last_move_time >= move_delay:
            if current_move_idx < len(current_game['game_state']):
                # Get next move
                player, position = current_game['game_state'][current_move_idx]
                
                # Make move
                board.make_move(position, player)
                
                # Start animation
                animating = True
                animation_pos = position
                
                current_move_idx += 1
                last_move_time = current_time
                
                # Check if game is finished
                if current_move_idx >= len(current_game['game_state']):
                    game_finished = True
                    winner, combo = board.check_winner()
                    if winner is not None:
                        winning_combo = combo
                        show_win_line = len(combo) > 0
        
        elif game_finished and auto_next:
            # Auto-load next game after delay
            if current_time - last_move_time >= move_delay * 2:
                board.reset()
                current_game = select_random_game()
                current_move_idx = 0
                show_win_line = False
                game_finished = False
    
    print(f"\n=== Session Complete ===")
    print(f"Total histories available: {len(game_histories)}")
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
