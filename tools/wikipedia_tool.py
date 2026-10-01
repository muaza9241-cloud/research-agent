from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Wikipedia tool initialize karein
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)