# ==========================================
# ADVANCED OFFLINE LANGCHAIN AGENT (FINAL VERSION)
# ==========================================

import os
import json
import math
import platform
from typing import Annotated

from transformers import pipeline
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline

# ==========================================
# 1. LOAD OFFLINE MODEL
# ==========================================

# 🔥 Best for CPU (stable)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 🔥 Uncomment if you have GPU (better performance)
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# ==========================================
# 2. MEMORY (CONVERSATION)
# ==========================================

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==========================================
# 3. OFFLINE DATA
# ==========================================

conversion_data = {
    ("USD", "INR"): 83.2,
    ("INR", "USD"): 0.012,
    ("EUR", "INR"): 90.5,
    ("INR", "EUR"): 0.011
}

# ==========================================
# 4. TOOLS (REAL WORLD)
# ==========================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Get conversion factor between currencies"""
    return conversion_data.get((base_currency, target_currency), "Rate not available")


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Convert currency using rate"""
    return base_currency_value * conversion_rate


@tool
def calculator(expression: str) -> str:
    """Solve math expressions like '45*12+10'"""
    try:
        return str(eval(expression))
    except:
        return "Calculation error"


@tool
def system_info(_: str = "") -> str:
    """Get system info"""
    return f"OS: {platform.system()}, CPU: {platform.processor()}"


@tool
def file_reader(filename: str) -> str:
    """Read a local file"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "File not found"


@tool
def word_count(text: str) -> str:
    """Count words"""
    return f"Word count: {len(text.split())}"


@tool
def basic_ai_helper(query: str) -> str:
    """Fallback general AI response"""
    return f"I think the answer is: {query}"

# ==========================================
# 5. TOOL LIST
# ==========================================

tools = [
    get_conversion_factor,
    convert,
    calculator,
    system_info,
    file_reader,
    word_count,
    basic_ai_helper
]

# ==========================================
# 6. AGENT SETUP
# ==========================================

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ==========================================
# 7. INTERACTIVE LOOP (LIKE CHATGPT)
# ==========================================

def run_agent():
    print("\n🤖 Offline AI Agent Ready (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = agent_executor.invoke({"input": user_input})

        print("\nAI:", response["output"])
        print("-" * 50)


# ==========================================
# 8. TEST RUN
# ==========================================

if __name__ == "__main__":
    run_agent()