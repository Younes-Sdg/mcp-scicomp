from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-scicomp")

import mcp_scicomp.tools.probability  # noqa: F401  — registers @mcp.tool() decorators
# from mcp_scicomp.tools import stochastic, ode, pde, sde, optimization, linalg
