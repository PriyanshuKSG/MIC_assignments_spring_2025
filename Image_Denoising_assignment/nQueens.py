from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_n_queens(N):
    # Create a Pyomo model
    model = ConcreteModel()

    # Define indices
    model.rows = RangeSet(1, N)
    model.cols = RangeSet(1, N)

    # Define decision variables (binary)
    model.x = Var(model.rows, model.cols, domain=Binary)

    # Constraint: One queen per row
    model.row_constraint = ConstraintList()
    for i in model.rows:
        model.row_constraint.add(sum(model.x[i, j] for j in model.cols) == 1)

    # Constraint: One queen per column
    model.col_constraint = ConstraintList()
    for j in model.cols:
        model.col_constraint.add(sum(model.x[i, j] for i in model.rows) == 1)

    # Constraint: No two queens on main diagonals
    model.main_diag_constraint = ConstraintList()
    for k in range(-N+1, N):
        model.main_diag_constraint.add(sum(model.x[i, j] for i in model.rows for j in model.cols if i - j == k) <= 1)

    # Constraint: No two queens on anti-diagonals
    model.anti_diag_constraint = ConstraintList()
    for k in range(2, 2*N):
        model.anti_diag_constraint.add(sum(model.x[i, j] for i in model.rows for j in model.cols if i + j == k) <= 1)

    # Objective function (dummy)
    model.obj = Objective(expr=sum(model.x[i, j] for i in model.rows for j in model.cols), sense=maximize)

    # Solve the model
    solver = SolverFactory('glpk')  # Use 'cbc' if GLPK is not available
    solver.solve(model, tee=False)

    # Extract the solution
    board = [['.' for _ in range(N)] for _ in range(N)]
    for i in model.rows:
        for j in model.cols:
            if value(model.x[i, j]) == 1:
                board[i-1][j-1] = 'Q'

    # Print the board
    for row in board:
        print(" ".join(row))

# Solve for N=8
solve_n_queens(8)
