import operator
import os
import uuid
from typing import Annotated, TypedDict, Union

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent
from langchain.callbacks import tracing_v2_enabled
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt.tool_executor import ToolExecutor

load_dotenv()

def create_search_reasoning_graph():
    # os.environ["TAVILY_API_KEY"] = get_secret()['tavily']['api_key_free_tier_nb']
    # os.environ["LANGCHAIN_API_KEY"] = get_secret()['langchain']['langsmith_secret_nb']
    # os.environ["LANGCHAIN_TRACING_V2"] = "false"

    tools = [
        TavilySearchResults(max_results=1)
    ]

    system = """You are a helpful assistant."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
            ("placeholder", "{chat_history}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    # Construct the OpenAI Functions agent
    agent_runnable = create_openai_functions_agent(llm, tools, prompt)

    class AgentState(TypedDict):
        # The input string
        input: str
        # The list of previous messages in the conversation
        chat_history: list[BaseMessage]
        # The outcome of a given call to the agent
        # Needs `None` as a valid type, since this is what this will start as
        agent_outcome: Union[AgentAction, AgentFinish, None]
        # List of actions and corresponding observations
        # Here we annotate this with `operator.add` to indicate that operations to
        # this state should be ADDED to the existing values (not overwrite it)
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    # This a helper class we have that is useful for running tools
    # It takes in an agent action and calls that tool and returns the result
    tool_executor = ToolExecutor(tools)

    # Define the agent
    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    def parse_as_str(data):
        chain = StrOutputParser()
        agent_outcome = chain.invoke(data)
        return {"agent_outcome": agent_outcome}

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        output = tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    # Define logic that will be used to determine which conditional edge to go down
    def should_continue(data):
        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(data["agent_outcome"], AgentFinish):
            return "end"
        # Otherwise, an AgentAction is returned
        # Here we return `continue` string
        # This will be used when setting up the graph to define the flow
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)
    # workflow.add_node("string_output", parse_as_str)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent")
    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )
    # workflow.add_edge("string_output", END)
    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("action", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    # app = workflow.compile()
    return workflow.compile()


if __name__ == "__main__":
    import json

    graph = create_search_reasoning_graph()

    with open("search_graph_mermaid_image.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())

    with open("search_graph_output.json", "w") as f:
        json.dump(graph.get_graph().to_json(), f)
    prompt = "whats the weather in sf?"
    inputs = {"input": prompt, "chat_history": []}
    # inputs = {"input": prompt}
    response = graph.invoke(inputs)
    print(f"RESPONSE: {response}")
    print("SUCCESS")
