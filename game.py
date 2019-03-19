import games.connect4.game as Connect4Game
import games.tictac.game as TittacGame
from config import GAME

possible_games = {
    'CONNECT4': Connect4Game,
    'TICTAC': TittacGame
}

current_game = possible_games.get(GAME, Connect4Game)

Game, GameState = current_game.Game, current_game.GameState
