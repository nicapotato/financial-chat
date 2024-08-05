from typing import List, Any, Union

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langserve import add_routes, CustomUserType
# from openbb import obb
from dotenv import load_dotenv

import os
import warnings
import pandas as pd

# from app.chains.agent import create_anthropic_agent_graph
from app.chains.seach_graph import create_search_reasoning_graph

warnings.filterwarnings("ignore")

load_dotenv()

# obb.account.login(pat=os.environ.get("OPENBB_TOKEN"), remember_me=True)
# obb.user.credentials.tiingo_token = os.environ.get("TIINGO_API_KEY")
# obb.user.credentials.fmp_api_key = os.environ.get("FMP_API_KEY")
# obb.user.credentials.intrinio_api_key = os.environ.get("INTRINIO_API_KEY")
# obb.user.credentials.fred_api_key = os.environ.get("FRED_API_KEY")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

app = FastAPI(
    title="Financial Chat",
    version="1.0",
    description="The Trading Dude Abides",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

graph = create_search_reasoning_graph()
llm = ChatOpenAI(model="gpt-4o")

class AgentInput(BaseModel):
    # {"messages": ("user", user_input)}
    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
        extra={"widget": {"type": "chat", "input": "messages"}},
    )

class AgentInputSearchGraph(BaseModel):
    # {"messages": ("user", user_input)}
    input: str = Field(
        ...,
        description="Single Query no chat memory",
        # extra={"widget": {"input": "input"}},
    )
    chat_history: list = Field(
        ...,
        description="Chat History",
        # extra={"widget": {"input": "input"}},
    )

class AgentOutput(BaseModel):
    output: Any


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}


add_routes(
    app,
    llm,
    path="/openai",
)


class Foo(CustomUserType):
    bar: int


def func(foo: Foo) -> int:
    """Sample function that expects a Foo type which is a pydantic model"""
    assert isinstance(foo, Foo)
    return foo.bar


# Note that the input and output type are automatically inferred!
# You do not need to specify them.
# runnable = RunnableLambda(func).with_types( # <-- Not needed in this case
#     input_type=Foo,
#     output_type=int,
#
add_routes(app, RunnableLambda(func), path="/foo")

def func_2(x: Any) -> int:
    """Mistyped function that should accept an int but accepts anything."""
    return x + 1


add_routes(app,
           RunnableLambda(func_2).with_types(
               input_type=int,
           ),
           path='/func'
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assisstant named Cob."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm


class MessageListInput(BaseModel):
    """Input for the chat endpoint."""
    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
        extra={"widget": {"type": "chat", "input": "messages"}},
    )


add_routes(
    app,
    chain.with_types(input_type=MessageListInput),
    path="/chat_langchain_basic",
)

add_routes(
    app,
    graph.with_types(input_type=AgentInputSearchGraph),
    path="/search_reasoning",
)

add_routes(
    app,
    graph,
    path="/search_reasoning_v2",
    input_type=AgentInputSearchGraph,
    output_type=AgentOutput,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
