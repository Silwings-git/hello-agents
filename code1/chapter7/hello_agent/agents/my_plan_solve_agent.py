# 默认规划器提示词模板
import ast
from typing import Any, Dict, List, Optional

from hello_agents import Config, HelloAgentsLLM, PlanAndSolveAgent

DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

DEFAULT_PROMOTS = {
    "panner": DEFAULT_PLANNER_PROMPT,
    "executor": DEFAULT_EXECUTOR_PROMPT,
}


class Planner:
    def __init__(self, llm_client: HelloAgentsLLM, planner_prompt: str | None = None):
        self.llm_client = llm_client
        self.planner_prompt = (
            planner_prompt if planner_prompt else DEFAULT_PLANNER_PROMPT
        )

    def plan(self, question: str) -> list[str]:
        """根据用户问题生成一个行动计划"""
        prompt = self.planner_prompt.format(question=question)

        # 为了生成计划,我们需要构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages)

        print(f"✅ 计划已生成:\n{response_text}")

        if response_text is None:
            print("❌ 方案响应信息为空")
            return []

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全的执行字符串,将其转换为python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    def __init__(self, llm_client: HelloAgentsLLM, executor_prompt: str | None = None):
        self.llm_client = llm_client
        self.executor_prompt = (
            executor_prompt if executor_prompt else DEFAULT_EXECUTOR_PROMPT
        )

    def execute(self, question: str, plan: list[str]) -> str:
        """根据计划,逐步执行并解决问题"""
        history = ""  # 用于存储历史步骤和结果的字符串

        response_text = ""

        print("\n--- 正在执行计划 ---")

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i + 1}/{len(plan)}: {step}")

            prompt = self.executor_prompt.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=i + 1,
            )

            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages) or ""

            response_text = "".join(response_text)

            # 更新历史记录, 为下一步做准备
            history += f"步骤 {i + 1}: {step}\n结果: {response_text}\n\n"

            print(f"\n✅ 步骤 {i + 1} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer


class MyPlanAndSolveAgent(PlanAndSolveAgent):

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            custom_prompts if custom_prompts else DEFAULT_PROMOTS,
        )

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)

    def run(self, input_text: str, **kwargs) -> str:

        print(f"\n--- 开始处理问题 ---\n问题: {input_text}")

        # 先规划
        plan = self.planner.plan(input_text)

        if not plan:
            print("\n--- 任务终止 ---\n无法生成有效的行动计划")

        # 再循环执行直到任务全部完成
        final_answer = self.executor.execute(input_text, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        return final_answer
