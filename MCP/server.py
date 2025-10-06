import asyncio
import json
from mcp.server.fastmcp import FastMCP
from agents import run_research
from deep_research_crew import create_research_discovery_crew, run_research_discovery
from femm_planner_crew import create_femm_planning_crew, run_femm_planning

# Create FastMCP instance
mcp = FastMCP("crew_research")

@mcp.tool()
async def crew_research(query: str) -> str:
    """Run CrewAI-based research system for given user query. Can do both standard and deep web search.

    Args:
        query (str): The research query or question.

    Returns:
        str: The research response from the CrewAI pipeline.
    """
    return run_research(query)

@mcp.tool()
async def discover_research(query: str) -> str:
    """Find 3-5 relevant research papers for user's electromagnetic/FEMM query.

    This tool integrates with your dual KG system (code + LaTeX) to extract relevant concepts,
    then uses deep web search to find academic papers that can be implemented in FEMM.

    Args:
        query (str): User's research interest (e.g., "optimize C-core actuator using CNN")

    Returns:
        str: JSON with 5 research options including summaries, relevance scores, and implementation complexity
    """
    try:
        result = run_research_discovery(query, max_options=5)
        return result
    except Exception as e:
        return f"Error in research discovery: {str(e)}"

@mcp.tool()
async def plan_implementation(selected_research_json: str) -> str:
    """Generate detailed FEMM + ML implementation plan for selected research.

    Takes a research option (from discover_research) and creates a comprehensive
    implementation plan including FEMM geometry, ML architecture, and execution steps.

    Args:
        selected_research_json (str): Full research option JSON from discover_research tool

    Returns:
        str: Detailed JSON plan with FEMM setup, ML architecture, execution steps, and resource requirements
    """
    try:
        result = run_femm_planning(selected_research_json)
        return result
    except Exception as e:
        return f"Error in implementation planning: {str(e)}"

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")


# add this inside ./.cursor/mcp.json
# {
#   "mcpServers": {
#     "crew_research": {
#       "command": "uv",
#       "args": [
#         "--directory",
#         "/Users/akshay/Eigen/ai-engineering-hub/Multi-Agent-deep-researcher-mcp-windows-linux",
#         "run",
#         "server.py"
#       ],
#       "env": {
#         "LINKUP_API_KEY": "your_linkup_api_key_here"
#       }
#     }
#   }
# }
