# self-play
- iniciar com o melhor jogador
- para cada episódio:
	- reiniciar o estado atual
	- aleatoriamente escolher a vez do jogador
	- enquanto não for o fim do jogo:
		- na primeira rodada:
			- criar a mcts
		- para cada uma das simulações mcts:
			- mover para a folha
			- avaliar a folha:
				- predict da rede neural
				- para cada uma das ações possíveis:
					- adicionar na mcts
					- criar aresta entre o novo nó e a folha
			- realizar backpropagation
		- obter value e policy
		- escolher a ação com a maior policy
	- verificar vencedor
- treinar a rede atual com a memória das jogadas passadas do melhor jogador
- realizar torneio entre melhor e atual jogador
- se o atual for melhor, atual vira o melhor

# jogar contra
from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg
env = Game()
playMatchesBetweenVersions(
    env
    , 9  # the run version number where the computer player is located
    , -1 # the version number of the first player (-1 for human)
    , 1 # the version number of the second player (-1 for human)
    , 10 # how many games to play
    , lg.logger_tourney # where to log the game to
    , 1  # which player to go first - 0 for random
)