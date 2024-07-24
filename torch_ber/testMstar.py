from od_mstar3 import cpp_mstar
# import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import mapf_gym
# 为了numpy不省略矩阵展示，方便debug。
import  numpy as np
np.set_printoptions(threshold=np.inf)
num_agents = 5
DIAG_MVMT = False
# ENVIRONMENT_SIZE = (10,70)
ENVIRONMENT_SIZE = (10,70)
GRID_SIZE = 10
OBSTACLE_DENSITY = (0,.5)
FULL_HELP = False


def show_world(world,start_positions,goals):
    print(np.array(world).shape)
    print('-' * 10, ' obstacle ', '-' * 10)
    print(world)
    print('-' * 10, ' start_positions ', '-' * 10)
    print(start_positions)
    print('-' * 10, ' goals ', '-' * 10)
    print(goals)


gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
world = gameEnv.getObstacleMap()
start_positions = tuple(gameEnv.getPositions())
goals = tuple(gameEnv.getGoals())

show_world(world,start_positions,goals)

try:
    mstar_path = cpp_mstar.find_path(world, start_positions, goals, 2, 5)
    # rollouts[self.metaAgentID] = self.parse_path(mstar_path)
    print(mstar_path)
    print(np.array(mstar_path).shape)
except OutOfTimeError:
    # M* timed out
    print("timeout", episode_count)
except NoSolutionError:
    print("nosol????", episode_count, start_positions)