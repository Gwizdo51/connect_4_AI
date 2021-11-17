import numpy as np
from pathlib import Path
import sys

ROOT_DIR_PATH = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR_PATH not in sys.path:
    sys.path.insert(1, ROOT_DIR_PATH)


def winning_grids():
    "create a list of arrays of shape (6,7) of booleans containing all 69 possible winning coin positions"

    # create horizontal solutions (24)
    sol_set_h = []
    for line in range(6):
        for col in range(4):
            solution = np.zeros(shape=(6,7), dtype=bool)
            solution[line, col] = True
            solution[line, col+1] = True
            solution[line, col+2] = True
            solution[line, col+3] = True
            sol_set_h.append(solution)

    # create vertical solutions (21)
    sol_set_v = []
    for line in range(3):
        for col in range(7):
            solution = np.zeros(shape=(6,7), dtype=bool)
            solution[line, col] = True
            solution[line+1, col] = True
            solution[line+2, col] = True
            solution[line+3, col] = True
            sol_set_v.append(solution)

    # create diagonal top left to bot right solutions (12)
    sol_set_tlbr = []
    for line in range(3):
        for col in range(4):
            solution = np.zeros(shape=(6,7), dtype=bool)
            solution[line, col] = True
            solution[line+1, col+1] = True
            solution[line+2, col+2] = True
            solution[line+3, col+3] = True
            sol_set_tlbr.append(solution)

    # create diagonal top right to bot left solutions (12)
    sol_set_trbl = []
    for line in range(3):
        for col in range(4):
            solution = np.zeros(shape=(6,7), dtype=bool)
            solution[line, col+3] = True
            solution[line+1, col+2] = True
            solution[line+2, col+1] = True
            solution[line+3, col] = True
            sol_set_trbl.append(solution)

    return sol_set_h + sol_set_v + sol_set_tlbr + sol_set_trbl


BRIGHT_COLOR = (150,150,160)
YELLOW_HOVER = (150,150,50)
YELLOW_COIN = (255,255,0)
RED_HOVER = (150,50,50)
RED_COIN = (255,0,0)

WINNING_GRIDS = winning_grids()


if __name__ == "__main__":

    solution1 = np.array([[1,2],[3,4]])
    solution2 = np.array([[5,6],[7,8]])
    solution3 = np.array([[9,10],[11,12]])
    solution4 = np.array([[13,14],[15,16]])
    solution5 = np.array([[17,18],[19,20]])

    # solution_set_1 = np.array((solution1, solution2))
    # solution_set_2 = np.array((solution3, solution4))
    # solution_set_2 = np.vstack((solution_set_2, np.array([solution5])))

    solution_set_1 = [solution1, solution2]
    solution_set_2 = [solution3, solution4]
    solution_set_2.append(solution5)

    solutions = np.vstack((np.array(solution_set_1), np.array(solution_set_2)))

    # print(solution_set_1)
    # print(solution_set_2)
    # print(solutions)
    # print(solutions.shape)
    # for solution in solutions:
    #     print(solution)

    print(winning_grids())
