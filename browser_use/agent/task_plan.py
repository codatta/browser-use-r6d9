import asyncio
import json
from langchain_openai import ChatOpenAI

# 初始化 OpenAI 模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 系统提示消息，指导代理如何将任务描述转换为逐步执行的操作
init_message = [{"role": "system",
                 "content": """
You are an automated browser-operating agent. Convert my task descriptions into execution sequences for easier processing by other agents.

Basic Principles You Must Follow:
1. If completing the task requires additional information, it should be considered an invalid task, and the reason should remind me of what information needs to be provided.
2. If the task cannot be completed via a browser, it should be considered an invalid task, and the reason should explicitly state this. For example, “Call 12345678 for me.”

Response Requirements: 
1. Must be a JSON object containing the following keys:
1) is_valid_task: Judge the validity of the task based on the previous Basic Principles. You must be very certain that the task is invalid before determining it as invalid.
2) invalid_reason: If the task is invalid, provide the reason. For example, specify what additional information is needed or explain why the task is unrelated to browser operations.
3) task_plan: A detailed execution plan for the task.
2. The return requirement is a string content that can be converted into JSON by code, and please do not provide me with JSON content in markdown format.
            """}]


# 生成任务执行计划的异步函数
async def get_task_plan(raw_task: str) -> str:
    """
    根据任务描述生成逐步执行的计划。

    :param raw_task: 用户输入的任务描述。
    :return: 生成的任务执行步骤。
    """
    messages = init_message + [{"role": "user", "content": raw_task}]
    response = await get_openai_response(messages)
    print(response)
    return json.loads(response)


# 调用 OpenAI 模型并获取响应的异步函数
async def get_openai_response(messages) -> str:
    """
    发送消息历史记录到 OpenAI 模型，并获取响应。

    :param messages: 包含系统指令和用户输入的消息列表。
    :return: AI 生成的响应文本。
    """
    response = await llm.ainvoke(messages)  # 使用异步方式调用 OpenAI
    return response.content if response else ""


# 运行主逻辑
if __name__ == '__main__':
    raw_task = "In docs.google.com write my Papa a quick thank you for everything letter."
    content = asyncio.run(get_task_plan(raw_task))
    print(content)
