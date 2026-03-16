# mcp-scicomp

A Model Context Protocol (MCP) server that gives LLMs a scientific computing stack — probability, stochastic processes, ODEs, PDEs, SDEs, linear algebra, and optimization.

Built on NumPy, SciPy, SymPy, and Matplotlib, mcp-scicomp exposes 17 tools that let an LLM describe data, fit distributions, simulate stochastic processes, solve differential equations, decompose matrices, and run optimization — returning numeric results and plots in a single call.

## Quick Start

### Install

```bash
git clone https://github.com/Younes-Sdg/mcp-scicomp.git
cd mcp-scicomp
uv sync
```

### Claude Code

```bash
claude mcp add mcp-scicomp -- uv run --directory /path/to/mcp-scicomp mcp-scicomp
```

### Claude Desktop / Other MCP Clients

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "scicomp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-scicomp", "mcp-scicomp"]
    }
  }
}
```

## Tools

17 tools across 7 modules:

| Module | Tool | Description |
|--------|------|-------------|
| **Probability** | `describe_data` | Descriptive statistics and normality test for a numeric sample |
| | `fit_distribution` | Fit parametric distributions to a dataset and rank by AIC |
| | `hypothesis_test` | Frequentist hypothesis test on one or two samples |
| **Stochastic Processes** | `simulate_brownian_motion` | Simulate arithmetic or geometric Brownian motion paths |
| | `simulate_ornstein_uhlenbeck` | Simulate Ornstein-Uhlenbeck (mean-reverting) processes |
| | `simulate_poisson_process` | Simulate homogeneous Poisson counting processes |
| | `markov_chain_analysis` | Stationary distribution, ergodicity, and simulation of a DTMC |
| **ODEs** | `solve_ode` | Solve a system of ODEs as an initial value problem |
| | `phase_portrait` | Plot the 2D phase portrait of an autonomous ODE system |
| **PDEs** | `solve_pde_1d` | Solve a 1D time-dependent PDE via Method of Lines |
| | `solve_laplace_2d` | Solve the 2D Laplace equation via Gauss-Seidel iteration |
| **SDEs** | `simulate_sde` | Simulate dX = f(X,t)dt + g(X,t)dW with Euler-Maruyama, Milstein, or Heun |
| | `analyze_sde` | Symbolic stationary distribution, stability, moments, Kolmogorov operators |
| **Linear Algebra** | `eigenanalysis` | Eigenvalues, eigenvectors, and spectral properties of a square matrix |
| | `matrix_decomposition` | SVD, LU, QR, or Cholesky factorization |
| **Optimization** | `optimize` | Minimize, maximize, or find roots of a mathematical function |
| | `curve_fit_data` | Fit a parametric model to (x, y) data via nonlinear least squares |

## Example Prompts

```
Describe this dataset and test if it's normally distributed: [2.1, 3.4, 2.8, 3.1, ...]

Fit a distribution to the values in data/samples.csv and show me the best fit.

Simulate 50 paths of geometric Brownian motion with mu=0.05 and sigma=0.2 over 1 year.

Analyze this Markov chain transition matrix and find its stationary distribution:
[[0.7, 0.3], [0.4, 0.6]]

Solve the Lorenz system with sigma=10, rho=28, beta=8/3 from (1, 1, 1) for t in [0, 50].

Draw the phase portrait of the Van der Pol oscillator: dx/dt = y, dy/dt = mu*(1-x^2)*y - x.

Solve the 1D heat equation u_t = 0.01 * u_xx on [0, 1] with u(0)=0, u(1)=0,
and initial condition u(x,0) = sin(pi*x).

Analyze the Ornstein-Uhlenbeck SDE: dX = -theta*(X - mu)*dt + sigma*dW.
Find its stationary distribution and moment equations.

Find the eigenvalues and eigenvectors of [[4, 1], [2, 3]].

Minimize the Rosenbrock function f(x, y) = (1-x)**2 + 100*(y-x**2)**2 starting from (-1, -1).
```

## Data Input

Tools that accept datasets (e.g. `describe_data`, `fit_distribution`, `curve_fit_data`) support two input methods:

- **Inline lists** — pass values directly as a JSON array: `[1.2, 3.4, 5.6, ...]`
- **File paths** — pass a `file_path` pointing to a local file. Supported formats: CSV, TSV, Excel (`.xlsx`), JSON, and Parquet

When using file paths, specify `column` to select which column to use from tabular data.

## License

MIT
