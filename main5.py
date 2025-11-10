
# from langchain_core.tools import tool
# from langchain_core.messages import (
#     AnyMessage,
#     SystemMessage,
#     ToolMessage,
#     HumanMessage,
#     AIMessage,
# )

# from typing_extensions import TypedDict, Annotated
# import operator
# from typing import Literal
# from langgraph.graph import StateGraph, START, END
# from langchain_ollama import ChatOllama  

# # ---------------- Model Setup ----------------
# model = ChatOllama(
#     model="phi3:latest",   # ✅ Phi-3 model
#     temperature=0.2
# )

# # ---------------- Define Tools ----------------
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply a and b."""
#     return a * b

# @tool
# def add(a: int, b: int) -> int:
#     """Add a and b."""
#     return a + b

# @tool
# def divide(a: int, b: int) -> float:
#     """Divide a by b."""
#     return a / b

# @tool
# def subtract(a: int, b: int) -> int:
#     """Subtract b from a."""
#     return a - b

# tools = [add, multiply, divide, subtract]
# tools_by_name = {tool.name: tool for tool in tools}


# model_with_tools = model

# # ---------------- Define State ----------------
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], operator.add]
#     llm_calls: int

# # ---------------- Define LLM Node ----------------
# def llm_call(state: MessagesState):
#     """LLM decides whether to call a tool or respond directly."""
#     messages = state["messages"]

#     # Include conversation context + system prompt
#     system_prompt = SystemMessage(
#         content="""
# You are a highly intelligent Math Expert that remembers the previous conversation naturally.
# Always interpret references like 'it', 'that', or 'previous value' based on prior messages.

# ✅ Responsibilities:
# - Understand and accurately solve any mathematical question.
# - Use the conversation context to resolve ambiguous words like 'it'.
# - Show steps only if the user explicitly requests them.
# - Support arithmetic, algebra, geometry, trigonometry, calculus, statistics, and unit conversions.

# 🧠 Rules:
# 1. Never refuse a math question.
# 2. If unclear, politely ask for clarification.
# 3. Always verify your result before responding.
# 4. Use concise and clear responses.

# ⚙️ Output Format:
# **Answer:** <numeric or algebraic result>
# """
#     )

#     # ✅ Use normal model invocation
#     response = model_with_tools.invoke([system_prompt] + messages)
#     return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

# # ---------------- Tool Node ----------------
# def tool_node(state: dict):
#     """Performs the tool call and updates the conversation state."""
#     result_msgs = []
#     last_msg = state["messages"][-1]

#     if hasattr(last_msg, "tool_calls"):
#         for tool_call in last_msg.tool_calls:
#             tool = tools_by_name[tool_call["name"]]
#             observation = tool.invoke(tool_call["args"])
#             result_msgs.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

#     return {"messages": result_msgs}

# # ---------------- Conditional Logic ----------------
# def should_continue(state: MessagesState) -> Literal["tool_node", END]:
#     """Route between tool execution and end response."""
#     last_msg = state["messages"][-1]
#     if hasattr(last_msg, "tool_calls") and getattr(last_msg, "tool_calls", None):
#         return "tool_node"
#     return END

# # ---------------- Build Graph ----------------
# builder = StateGraph(MessagesState)
# builder.add_node("llm_call", llm_call)
# builder.add_node("tool_node", tool_node)
# builder.add_edge(START, "llm_call")
# builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
# builder.add_edge("tool_node", "llm_call")
# agent = builder.compile()

# # ---------------- Run Interactive CLI ----------------
# if __name__ == "__main__":
#     print("\n🤖 Phi-3 + LangGraph Memory-Based Math Agent")
#     print("Type 'exit' or 'no' to quit.\n")

#     state = {"messages": [], "llm_calls": 0}

#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() in ["exit", "no", "quit"]:
#             print("Goodbye 👋")
#             break

#         # Keep conversation history
#         state["messages"].append(HumanMessage(content=user_input))
#         result_state = agent.invoke({"messages": state["messages"], "llm_calls": state["llm_calls"]})

#         # Append model responses to message history
#         for msg in result_state["messages"]:
#             state["messages"].append(msg)
#             if hasattr(msg, "content"):
#                 print("Result:", msg.content)

#         print("----------------\n")



from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
import re

model = ChatOllama(model="phi3:latest", temperature=0.2)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add a and b."""
    return a + b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

tools = [add, multiply, divide, subtract]
tools_by_name = {tool.name: tool for tool in tools}

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    last_result: float  

def llm_call(state: MessagesState):
    """LLM reasoning node that uses conversation memory."""
    messages = state["messages"]
    last_result = state.get("last_result", None)

    
    system_prompt = SystemMessage(
    content=f"""
You are a precise and context-aware Math Expert.

🧠 Context:
The last computed result was {last_result if last_result is not None else "None"}.

✅ Behavior:
- When the user says things like "add by 4", "multiply it by 2", or "subtract that by 3",
  automatically substitute the last result in place of 'it', 'that', or 'previous value' — then compute.
- If no previous result exists, politely ask for clarification once.
- When a valid previous result exists, NEVER ask or explain — just give the final numeric answer.
- Do not show reasoning, steps, or re-explain what 'it' refers to.

⚙️ Output Format:
Always respond in one line, like this:
Answer: <number>

📘 Capabilities:
Handle arithmetic, algebra, trigonometry, geometry, calculus, statistics, and conversions.

🚫 Forbidden:
- Do not write "Please specify" or "assuming".
- Do not repeat the question.
- Do not include history or commentary.
- Only return the computed numeric result in the specified format.
"""
)



    response = model.invoke([system_prompt] + messages)
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

def tool_node(state: dict):
    """Executes a tool and returns the result."""
    result_msgs = []
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls"):
        for tool_call in last_msg.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result_msgs.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

    return {"messages": result_msgs}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decides whether to call a tool or finish."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and getattr(last_msg, "tool_calls", None):
        return "tool_node"
    return END

builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()



def extract_number(text):
    """
    Prefer a number that follows 'Answer:' if present.
    Otherwise return the last numeric token in the text.
    """
    if not text:
        return None

    # Try to find a number right after '**Answer:**' or 'Answer:'
    ans_match = re.search(r"Answer:\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", text, flags=re.IGNORECASE)
    if ans_match:
        try:
            return float(ans_match.group(1))
        except:
            pass
    nums = re.findall(r"[+-]?\d+\.?\d*(?:e[+-]?\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except:
        return None

if __name__ == "__main__":
    print("\n🤖 Phi-3 + LangGraph Context-Aware Math Expert (with Memory)")
    print("Type 'exit' to quit.\n")

    state = {"messages": [], "llm_calls": 0, "last_result": None}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "no", "quit"]:
            print("Goodbye 👋")
            break

        
        temp_state = {
            "messages": state["messages"] + [HumanMessage(content=user_input)],
            "llm_calls": state["llm_calls"],
            "last_result": state["last_result"],
        }

        result_state = agent.invoke(temp_state)

        last_ai = None
        for msg in reversed(result_state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break
        if last_ai and hasattr(last_ai, "content"):
            final_text = last_ai.content.strip()
            print("Result:", final_text)
            num = extract_number(final_text)

            if num is not None:
                state["last_result"] = num
        else:
            for msg in result_state["messages"]:
                if hasattr(msg, "content"):
                    print("Result:", msg.content.strip())

        state["messages"].append(HumanMessage(content=user_input))
        if last_ai:
            state["messages"].append(last_ai)
        state["llm_calls"] = result_state["llm_calls"]

        print("----------------\n")



