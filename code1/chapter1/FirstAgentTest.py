import os
from re import S

from dotenv import load_dotenv
from openai import OpenAI
from regex import F
from tavily import TavilyClient
from openai.types.chat import ChatCompletionMessageParam
import requests

# 加载环境变量
load_dotenv()


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"缺少必需的环境变量: {key}")
    return value


# 配置API秘钥
API_KEY = _require_env("API_KEY")
BASE_URL = _require_env("BASE_URL")
MODEL_ID = _require_env("MODEL_ID")
TAVILY_API_KEY = _require_env("TAVILY_API_KEY")

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""


# 定义工具
def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200成功
        response.raise_for_status()
        # 解析返回的json数据
        data = response.json()

        print(f"wttr.in响应的天气结果: {data}")

        # 提取天气情况
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        # 格式化为自然语言
        return f"{city}当前天气: {weather_desc}, 气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误: 查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误: 解析天气数据失败, 可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    if not TAVILY_API_KEY:
        return "错误: 未配置TAVILY_API_KEY"

    # 初始化Tavily客户端
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    # 构造一个精确查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐和理由"

    try:
        # 调用API, include_answer=True会返回一个综合性回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # Tavily返回的结果已经非常干净，可以直接使用
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答, 则格式化原始结果
        formatted_results = []
        for result in response.get("result", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉, 没有找到相关的旅游景点推荐"

        return "根据搜索, 为你找到以下信息: \n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误: 执行Tavily搜索时出现问题 - {e}"


# 将所有工具函数放入一个字典, 方便后续使用
available_tools = {"get_weather": get_weather, "get_attraction": get_attraction}


# OpenAI客户端
class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用LLM API来生成回应
        """
        try:
            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
            )
            answer = response.choices[0].message.content
            if answer is None:
                print("大语言模型未响应content")
                return "错误: 模型未返回文本内容"
            print("大语言模型响应成功")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误: 调用大语言模型服务时出错"


class TravelAssistant:
    """
    智能旅行助手类
    """

    def __init__(self):
        self.llm = OpenAICompatibleClient(
            model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL
        )
        self.prompt_history = []

    def reset(self):
        """重置对话历史"""
        self.prompt_history = []

    def add_user_message(self, message: str):
        """添加用户消息到历史"""
        self.prompt_history.append(f"用户请求: {message}")

    def add_assistant_message(self, message: str):
        """添加助手消息到历史"""
        self.prompt_history.append(message)

    def add_observation(self, observation: str):
        """添加观察结果到历史"""
        self.prompt_history.append(f"Observation: {observation}")


def display_conversation(history):
    """美观的显示对话历史"""
    print("\n" + "=" * 60)
    print("📝 对话历史")
    print("=" * 60)

    for i, message in enumerate(history, 1):
        if message.startswith("用户请求:"):
            print(f"\n👤 用户 [{i}]: {message[5:]}")
        elif message.startswith("Thought:"):
            print(f"\n🤔 思考 [{i}]: {message[8:].strip()}")
        elif message.startswith("Action:"):
            print(f"🛠️  行动 [{i}]: {message[7:].strip()}")
        elif message.startswith("Observation:"):
            print(f"📊 观察 [{i}]: {message[12:].strip()}")
        else:
            print(f"💬 消息 [{i}]: {message}")

    print("=" * 60 + "\n")

def parse_action(action_str):
    """解析行动字符串"""
    if action_str.startswitch("Finish"):
        