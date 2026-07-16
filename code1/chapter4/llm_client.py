import os
from typing import Dict, List
from openai.types.chat import ChatCompletionMessageParam

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    "Hello Agents"定制LLM客户端
    用于调用任何兼容OpenAI接口的服务,并默认使用流式响应
    """

    def __init__(
        self,
        model: str | None = None,
        apiKey: str | None = None,
        baseUrl: str | None = None,
        timeout: int | None = None,
    ):
        """初始化客户端. 优先使用传入的参数,如未提供,则从环境变量加载"""
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not self.model:
            raise ValueError("模型ID必须被提供或在.env文件中定义")
        if not apiKey:
            raise ValueError("API密钥必须被提供或在.env文件中定义")
        if not baseUrl:
            raise ValueError("服务地址必须被提供或在.env文件中定义")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(
        self, messages: List[Dict[str, str]], temperature: float = 0
    ) -> str | None:
        """调用LLM进行思考并返回其响应"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            assert self.model is not None
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# 客户端使用示例

if __name__ == "__main__":
    try:
        llm_client = HelloAgentsLLM()

        example_messages = [
            {"role": "system", "content": "你是一个出色的python代码专家"},
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用LLM ---")

        response_text = llm_client.think(example_messages)

        if response_text:
            print(f"\n\n--- 完整模型响应 ---\n{response_text}")

    except ValueError as e:
        print(e)
