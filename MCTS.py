import math

import numpy as np

import config
import loggers as lg


class Node():

    def __init__(self, state):
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn
        self.action = action
        self.q_plus_u = 0

        self.stats = {
            'N': 0,  # visit count
            'W': 0,  # total action-value
            'Q': 0,  # mean action-value
            'P': prior,  # prior probability of selecting that edge
        }
        # AlphaGo Zero - Mastering the game of Go without human knowledge, page 25, Search Algorithm
        # https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero


class MCTS():

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self):

        lg.logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0

        while not currentNode.isLeaf():

            lg.logger_mcts.info('PLAYER TURN...%d',
                                currentNode.state.playerTurn)

            # AlphaGo Zero - Mastering the game of Go without human knowledge, page 25, Search Algorithm
            # https://github.com/LeelaChessZero/lc0/wiki/Technical-Explanation-of-Leela-Chess-Zero
            # This is the same search specified by the AGZ paper, PUCT (Predictor + Upper Confidence Bound tree search).
            # Multi-armed Bandits with Episode Context - Christopher D. Rosin

            maxQU = -math.inf
            currentMaxQU = -math.inf

            # apply noise dirichlet on P
            # AlphaGo Zero - Mastering the game of Go without human knowledge, page 24, Self-play
            # leela - search.cc - ApplyDirichletNoise - line 83
            if currentNode == self.root:
                epsilon = config.EPSILON
                noise = np.random.dirichlet(
                    [config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                noise = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb += edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):
                U = self.cpuct * \
                    edge.stats['P'] * (1 - epsilon) + epsilon * noise[idx] * \
                    (np.sqrt(Nb) / 1 + edge.stats['N'])

                Q = edge.stats['Q']

                lg.logger_mcts.info(
                    'action: %d (%d)... N = %d, P = %f, noise = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f',
                    action, action % 7, edge.stats['N'], np.round(
                        edge.stats['P'], 6), np.round(noise[idx], 6),
                    ((1 - epsilon) * edge.stats['P'] + epsilon * noise[idx]), np.round(edge.stats['W'], 6),
                    np.round(Q, 6), np.round(U, 6), np.round(Q + U, 6))

                edge.q_plus_u = Q + U

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge
                    currentMaxQU = edge.q_plus_u

            lg.logger_mcts.info(
                'action with highest Q + U...%d %d', simulationAction, currentMaxQU)

            # the value of the newState from the POV of the new playerTurn
            _, value, done = currentNode.state.takeAction(simulationAction)
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        lg.logger_mcts.info('DONE...%d', done)

        return currentNode, value, done, breadcrumbs

    def backFill(self, leaf, value, breadcrumbs):
        lg.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f',
                                value * direction, playerTurn, edge.stats['N'], edge.stats['W'], edge.stats['Q']
                                )

            edge.outNode.state.render(lg.logger_mcts)

    def addNode(self, node):
        self.tree[node.id] = node
