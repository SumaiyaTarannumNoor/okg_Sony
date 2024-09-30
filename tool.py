# BaseTool クラスのインポート
from langchain.agents import BaseTool

# 検索を行うツールを定義
class CustomSearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str) -> str:
        """Use the tool."""
        search = SerpAPIWrapper()
        return search.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")