import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
import matplotlib.pyplot as plt

def rref(A_aug):
    A_sym = sp.Matrix(A_aug)
    rref_matrix, pivot_columns = A_sym.rref()
    return np.array(rref_matrix).astype(np.float64), pivot_columns

def solve_systems_rref(A, b):
    A_aug = np.hstack([A, b.reshape(-1, 1)])
    rref_matrix, pivot_columns = rref(A_aug)
    
    # Check for inconsistency
    for row in rref_matrix:
        if np.all(row[:-1] == 0) and row[-1] != 0:
            return None, None, None, None
    
    n = A.shape[1]
    free_variables = set(range(n)) - set(pivot_columns)
    
    if not free_variables:
        # If there are no free variables, the system is fully determined
        x_homogeneous = np.zeros(n)
        x_nonhomogeneous = np.linalg.lstsq(A, b, rcond=None)[0]
        return x_homogeneous, x_nonhomogeneous, [], None
    else:
        x_homogeneous = np.zeros(n)
        for idx in free_variables:
            x_homogeneous[idx] = 1
        x_nonhomogeneous = np.zeros(n)
        x_nonhomogeneous[list(free_variables)] = 1
        return x_homogeneous, x_nonhomogeneous, list(free_variables), pivot_columns

def format_solution(A, x_homogeneous, x_nonhomogeneous, free_variables, pivot_columns):
    solutions = []
    n = A.shape[1]
    
    if not free_variables:
        for i in range(n):
            solutions.append(f"x_{i} = {x_nonhomogeneous[i]:.4f}")
    else:
        for i in range(n):
            if i in pivot_columns:
                equation = f"x_{i} = {x_nonhomogeneous[i]:.4f}"
                for free_var in free_variables:
                    equation += f" + {A[:, free_var][i]:.4f}*x_{free_var}"
                solutions.append(equation)
            else:
                solutions.append(f"x_{i} = x_{i}")
    
    return solutions

def plot_solutions_2d(A, x_homogeneous, x_nonhomogeneous, free_variables):
    fig, ax = plt.subplots()
    
    t = np.linspace(-10, 10, 400)
    x_vals = np.zeros((2, len(t)))
    
    for i in range(2):
        if i in free_variables:
            x_vals[i] = t
        else:
            x_vals[i] = x_nonhomogeneous[i] + t * x_homogeneous[i]

    ax.plot(x_vals[0], x_vals[1], label='Solution space')
    ax.scatter(x_nonhomogeneous[0], x_nonhomogeneous[1], c='r', label='Particular solution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    st.pyplot(fig)

def plot_solutions_3d(A, x_homogeneous, x_nonhomogeneous, free_variables):
    fig = go.Figure()

    if len(free_variables) == 0:
        fig.add_trace(go.Scatter3d(x=[x_nonhomogeneous[0]], y=[x_nonhomogeneous[1]], z=[x_nonhomogeneous[2]],
                                   mode='markers', marker=dict(size=10, color='red'), name='Unique solution'))
    elif len(free_variables) == 1:
        t = np.linspace(-10, 10, 400)
        x_vals = np.array([x_nonhomogeneous + t_val * x_homogeneous for t_val in t])
        fig.add_trace(go.Scatter3d(x=x_vals[:, 0], y=x_vals[:, 1], z=x_vals[:, 2],
                                   mode='lines', line=dict(width=4), name='Line of solutions'))
    elif len(free_variables) >= 2:
        t1 = np.linspace(-10, 10, 20)
        t2 = np.linspace(-10, 10, 20)
        u = x_homogeneous
        v = np.zeros_like(u)
        v[free_variables[1]] = 1  # Second free variable direction
        points_x, points_y, points_z = [], [], []
        for c in t1:
            for d in t2:
                w = c * u + d * v
                x_vals = x_nonhomogeneous + w
                points_x.append(x_vals[0])
                points_y.append(x_vals[1])
                points_z.append(x_vals[2])
        fig.add_trace(go.Scatter3d(x=points_x, y=points_y, z=points_z,
                                   mode='markers', marker=dict(size=5, color='blue'), name='Plane of solutions'))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[-10, 10]),
    ))

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Matrix Input Application")

    st.write("Enter the augmented matrix values directly into the grid:")

    # Specify the size of the matrix
    rows = st.number_input("Number of rows", min_value=1, value=3)
    cols = st.number_input("Number of columns (including b)", min_value=2, value=4)

    # Initialize the matrix with zeros
    matrix_input = np.zeros((rows, cols)).tolist()

    # Create a grid options builder
    gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(matrix_input, columns=[str(i) for i in range(cols)]))
    gb.configure_default_column(editable=True)
    grid_options = gb.build()

    # Calculate the appropriate height for the grid
    grid_height = 40 + rows * 40  

    # Create an interactive data grid
    grid_response = AgGrid(pd.DataFrame(matrix_input, columns=[str(i) for i in range(cols)]), gridOptions=grid_options, height=grid_height, fit_columns_on_grid_load=True)

    # Button to trigger matrix parsing
    if st.button("Submit Matrix"):
        matrix_input = grid_response['data'].values.tolist()
        A = np.array([row[:-1] for row in matrix_input])
        b = np.array([row[-1] for row in matrix_input])
        valid_input = True

        # Parse the matrix input values
        try:
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except ValueError:
            valid_input = False
            st.error("Invalid input. Please enter numeric values.")

        if valid_input:
            # Display the entered matrix
            st.write("The entered matrix A and vector b are:")
            st.write("A:")
            st.write(A)
            st.write("b:")
            st.write(b)

            # Solve the systems using RREF
            x_homogeneous, x_nonhomogeneous, free_variables, pivot_columns = solve_systems_rref(A, b)

            if x_homogeneous is None:
                st.error("The system is inconsistent and has no solutions.")
            else:
                if free_variables:
                    st.write("The solution has free variables. The free variables are:")
                    for var in free_variables:
                        st.write(f"x_{var}")
                else:
                    st.write("There are no free variables in the solution.")

                # Format and display the solutions
                solutions = format_solution(A, x_homogeneous, x_nonhomogeneous, free_variables, pivot_columns)
                st.write("The solution to the system is:")
                for solution in solutions:
                    st.write(solution)

                # Plot the solutions if in ℝ2 or ℝ3
                if cols - 1 == 2:
                    plot_solutions_2d(A, x_homogeneous, x_nonhomogeneous, free_variables)
                elif cols - 1 == 3:
                    plot_solutions_3d(A, x_homogeneous, x_nonhomogeneous, free_variables)

if __name__ == "__main__":
    main()
