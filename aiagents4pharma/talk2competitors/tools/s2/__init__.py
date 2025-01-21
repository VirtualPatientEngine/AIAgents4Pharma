from tools.s2.display_results import display_results
from tools.s2.multi_paper_rec import get_multi_paper_recommendations
from tools.s2.search import search_tool
from tools.s2.single_paper_rec import get_single_paper_recommendations

# Export all tools in a list for easy access
s2_tools = [
    search_tool,
    display_results,
    get_single_paper_recommendations,
    get_multi_paper_recommendations,
]

__all__ = [
    "search_tool",
    "display_results",
    "get_single_paper_recommendations",
    "get_multi_paper_recommendations",
    "s2_tools",
]
