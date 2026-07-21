import asyncio
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str  # 经过llm理解后的用户需求总结
    search_query: str  # 优化后用于Tavily api的搜索查询
    search_results: str  # Tavily 查询结果
    final_answer: str  # 最终答案
    step: str  # 标记当前步骤
    check_pass: bool  # 检查是否通过


load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", ""),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", ""),
    temperature=0.7,
)

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def understand_query_node(state: SearchState) -> SearchState:
    """步骤1: 理解查询并生成搜索关键字"""
    user_message = state["messages"][0].content

    if state.get("check_pass", True):
        understand_prompt = f"""
    分析用户的查询: "{user_message}"
    请完成两个任务:
    1. 简洁总结用户想要了解什么
    2. 生成最适合搜索引擎的关键词(中英文均可,要精准)

    格式:
    理解: [用户需求总结]
    搜索词: [最佳搜索关键词]
    """

        response = llm.invoke([SystemMessage(content=understand_prompt)])
        llm_res: str = response.content
        print("a" * 10)
        search_query = _parse_query_res(user_message, llm_res)

        return SearchState(
            state,
            user_query=llm_res,
            search_query=search_query,
            step="understood",
            messages=[AIMessage(content=f"我将为您搜索: {search_query}")],
        )
    else:
        messages = state["messages"] + [
            HumanMessage(content="指定的搜索词不恰当,请重新思考一个更合适的搜索词,仍使用之前的格式回复"),
        ]
        print(f"b----{messages}---")
        response = llm.invoke(messages)
        llm_res: str = response.content
        search_query = _parse_query_res(user_message, llm_res)
        return SearchState(
            state,
            user_query=llm_res,
            search_query=search_query,
            step="understood",
            messages=[AIMessage(content=f"我将为您搜索: {search_query}")],
        )


def _parse_query_res(user_message: str, llm_res: str):
    # 解析LLM的输出, 提取搜索关键词
    search_query = user_message
    if "搜索词:" in llm_res:
        search_query = llm_res.split("搜索词:")[1].strip()
    return search_query


def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2: 使用Tavily API进行审视搜索"""
    search_query = state["search_query"]
    try:
        print(f"正在搜索: {search_query}")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )

        search_results = ""

        # Tavily返回的结果已经非常干净，可以直接使用
        if response.get("answer"):
            search_results = response["answer"]
        elif response.get("result", []):
            # 如果没有综合性回答, 则格式化原始结果
            formatted_results = []
            for result in response.get("result", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")
            search_results = "\n".join(formatted_results)
        else:
            search_results = "未搜索到结果"

        return SearchState(
            state,
            search_results=search_results,
            step="searched",
            messages=[AIMessage(content="搜索完成! 正在整理答案")],
        )

    except Exception as e:
        print("错误: {e}")
        return SearchState(
            state,
            search_results=f"搜索失败: {e}",
            step="search_failed",
            messages=[AIMessage(content="搜索遇到问题")],
        )


def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3: 基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        # 如果搜索失败,执行回退策略, 基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用, 请基于你的知识回答用户的问题: \n用户问题: {state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功,基于搜索结果生成答案
        answer_prompt = f"""
基于一下搜索结果为用户提供完整,准确的答案:
用户问题: {state["user_query"]}
搜索结果:\n{state["search_results"]}
请综合搜索结果,提供准确,有用的回答...
"""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return SearchState(
        state,
        final_answer=response.content,
        step="completed",
        messages=[AIMessage(content=response.content)],
    )


def human_node(state: SearchState) -> dict:

    if state.get("search_query"):
        # 如果有搜索词, 由人类决定搜索词是否准确
        print("你觉得搜索词恰当吗?请回复:\n")

        while True:
            human_answer = input()
            prompt = f"判断用户的输入表达的是肯定还是否定,亦或是不确定. 若表达肯定回复1, 若表达否定回复0, 若不确定回复2. \n问题: '你决定搜索词恰当吗?'\n用户的输入: {human_answer}"
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content

            if "0" in response_text:
                print("人类回复0\n")
                return {
                    "check_pass": False,
                    "step": "check_search_query_refuse",
                    "messages": [HumanMessage(content="搜索词不准确,请重新调整搜索词")],
                }
            elif "1" in response_text:
                print("人类回复1\n")
                return {
                    "check_pass": True,
                    "step": "check_search_query_pass",
                    "messages": [HumanMessage(content="搜索词准确,请执行搜索")],
                }
            else:
                print("无法理解您的输入,请重新输入:\n")
                continue
    else:
        return {}


def human_route(state: SearchState) -> str:
    if state.get("check_pass") == 1:
        return "search"
    else:
        return "understand"


def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("check_search_query", human_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 添加边
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "check_search_query")
    workflow.add_conditional_edges("check_search_query", human_route)
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


async def main():
    """主函数：运行智能搜索助手"""

    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return

    app = create_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    session_count = 0

    while True:
        user_input = input("🤔 您想了解什么: ").strip()

        if user_input.lower() in ["quit", "q", "退出", "exit"]:
            print("感谢使用！再见！👋")
            break

        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        # 初始状态
        initial_state = SearchState(
            messages=[HumanMessage(content=user_input)],
            user_query="",
            search_query="",
            search_results="",
            final_answer="",
            step="start",
        )

        try:
            print("\n" + "=" * 60)

            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")

            print("\n" + "=" * 60 + "\n")

            import json

            print("=== checkpoint state (JSON): ===")
            state = app.get_state(config)
            values = state.values if state else {}
            # 将 messages 中的对象转为可序列化的字符串
            if "messages" in values:
                values["messages"] = [
                    {"role": m.__class__.__name__, "content": m.content}
                    for m in values["messages"]
                ]
            print(json.dumps(values, ensure_ascii=False, indent=2))
            print("=" * 21)

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    asyncio.run(main())
