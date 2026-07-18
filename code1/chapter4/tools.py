import os
from typing import Any, Dict
from dotenv import load_dotenv
from serpapi import SerpApiClient

load_dotenv()

def search(query: str) -> str:
    """基于SerpApi的网页搜索工具"""

    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")

    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误: SERPAPI_API_KEY 未再 .env 文件中配置"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 只能解析: 优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案, 则返回前三个有结果的摘要
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


class ToolExecutor:
    """一个工具执行器, 负责管理和执行工具"""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """向工具箱注册一个新工具"""
        # 检测工具是否已经存再
        if name in self.tools:
            print(f"警告: 工具 '{name}' 已存在, 将被覆盖")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册")

    def getTool(self, name: str) -> callable:
        """根据名称获取一个工具的执行函数"""
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )


# 工具初始化与使用示例
if __name__ == "__main__":
    # 初始化工具执行器
    tool_executor = ToolExecutor()

    # 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    # 打印可用工具
    print("\n--- 可用的工具 ---")
    print(tool_executor.getAvailableTools())

    # 智能体Action调用, 询问实时性问题
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"
    print(f"\n--- 执行 Action: Search['{tool_input}']")

    tool_function = tool_executor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("\n--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误: 未找到名为 '{tool_name}' 的工具")
