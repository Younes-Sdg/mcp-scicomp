from mcp_scicomp.app import mcp  # noqa: F401

import mcp_scicomp.tools.probability  # noqa: F401
import mcp_scicomp.tools.stochastic  # noqa: F401
import mcp_scicomp.tools.ode  # noqa: F401
import mcp_scicomp.tools.pde  # noqa: F401
import mcp_scicomp.tools.sde  # noqa: F401
# import mcp_scicomp.tools.optimization  # noqa: F401
import mcp_scicomp.tools.linalg  # noqa: F401

if __name__ == "__main__":
    mcp.run()
