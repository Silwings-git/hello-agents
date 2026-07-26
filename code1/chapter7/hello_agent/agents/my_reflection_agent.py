from typing import Dict, Optional

from hello_agents import Config, HelloAgentsLLM, ReflectionAgent


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

class MyReflectionAgent(ReflectionAgent):

    def __init__(
            self,
                    name: str,
                    llm: HelloAgentsLLM,
                    system_prompt: Optional[str] = None,
                    config: Optional[Config] = None,
                    max_iterations: int = 3,
                    custom_prompts: Optional[Dict[str, str]] = None
    ):
        super().__init__(name,llm,system_prompt,config,max_iterations,custom_prompts if custom_prompts else DEFAULT_PROMPTS)
        print(f"✅ {name} 初始化完成，最大步数: {max_iterations}")


    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n--- 开始处理任务 ---\n任务: {input_text}")

        prompts = self.prompts

        # 初始执行
        initial_prompt = prompts["initial"].format(task=input_text)
        initial_result = super()._get_llm_response(initial_prompt,**kwargs)
        self.memory.add_record("execution",initial_result)

        # 迭代循环, 反思和优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            # 进行反思
            print(f"\n-> 正在进行反思...")
            last_execution = self.memory.get_last_execution()
            reflect_prompt = prompts["reflect"].format(task=input_text, content = last_execution)
            reflect_result = super()._get_llm_response(reflect_prompt,**kwargs)
            self.memory.add_record("reflect",reflect_result)

            if "无需改进" in reflect_result:
                print("\n✅ 反思认为回答已无需改进，任务完成。")
                break

            # 进行优化
            print("\n-> 正在进行优化...")
            refine_prompt = prompts["refine"].format(task = input_text,last_attempt = last_execution, feedback=reflect_result)
            refine_result = super()._get_llm_response(refine_prompt,**kwargs)
            self.memory.add_record("execution",refine_result)

        final_answer = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终回答:\n{final_answer}")
        return final_answer