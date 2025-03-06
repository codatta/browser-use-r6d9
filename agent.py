import json

from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from browser_use import Browser, BrowserConfig
from browser_use.agent.task_plan import get_task_plan
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from browser_use.browser.context import BrowserContext, BrowserContextConfig

llm = ChatOpenAI(model="gpt-4o", temperature=0)
browser = Browser(
    config=BrowserConfig(
        headless=False,
        # NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig()
    )
)
controller = Controller()


async def main():
    task = "打开淘宝和京东"
    # task = "Find a one-way flight from Beijing to Tokyo on 2 April 2025 on Google Flights. Return me the cheapest option."
    # task = f"打开x.com主页，获取前10条推文及其作者、点赞等数据。如果需要登录，账号是:{os.environ['TWITTER_NAME']}，密码是:{os.environ['TWITTER_PASSWORD']}"

    # task = f"我想给我3岁女儿在淘宝买个裤子，我想要销量最多的那个加到购物车里。如果没登录的话先登录下，账号是:{os.environ['TAOBAO_NAME']}，密码是:{os.environ['TAOBAO_PASSWORD']}"
    # task = f"我想在亚马逊买一本关于Deep Learning的畅销书，好评低于4.7的不考虑，帮我加到购物车。如果没登录的话先登录下，账号是:{os.environ['AMAZON_NAME']}，密码是:{os.environ['AMAZON_PASSWORD']}"

    # print(task)
    task_plan = await get_task_plan(task)
    print(json.dumps(task_plan, indent=2))

    is_valid_task = task_plan.get("is_valid_task")
    invalid_reason = task_plan.get("invalid_reason")
    task_plan = task_plan.get("task_plan")

    if is_valid_task:
        if task_plan:
            agent = Agent(
                task=task_plan,
                llm=llm,
                controller=controller,
                browser=browser,
                max_actions_per_step=1
            )
            result = await agent.run()
            print(result)
        else:
            print("System error!")
    else:
        print(f"Task is invalid, reason: {invalid_reason}")


asyncio.run(main())
