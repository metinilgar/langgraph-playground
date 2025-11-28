"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from turtle import mode
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from typing import Literal


model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.5
)


@dataclass
class WriterState:
    draft : str # Güncel tweet
    critique : str = ""
    revision_number : int = 0

async def writer_node(state: WriterState) -> WriterState:
    """Write a tweet about the article."""
    draft = state.draft
    critique = state.critique

    if critique:
        prompt = f"You are a writer. Update an existing tweet based on the criticism. Criticism: {critique}"
    else:
        prompt = "You are a writer. Write a short but impactful tweet on the topic."


    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=draft),
    ]
    
    response = await model.ainvoke(messages)

    return {"draft": response.content}   


async def editor_node(state: WriterState) -> WriterState:
    """Critique the article."""
    messages = [
        SystemMessage(content="""
Review the following tweet draft.
Rules: It must be less than 280 characters, contain no emojis or hashtags, and be clear and effective. 
If the tweet is good enough, just write “YES” to approve it; do not comment. If it is not good enough, explain why it needs to be corrected.
        """),
        HumanMessage(content=state.draft),
    ]

    response = await model.ainvoke(messages)

    
    return {"critique": response.content, "revision_number": state.revision_number + 1}

def approve(state: WriterState) -> Literal["writer_node", END]:
    """Approve the tweet."""
    if state.critique.lower().startswith("yes"):
        return END
    elif state.revision_number > 4:
        return END
    
    return "writer_node"     

# Define the graph
graph = (
    StateGraph(WriterState)
    .add_node(writer_node)
    .add_node(editor_node)
    .add_edge(START, "writer_node")
    .add_edge("writer_node", "editor_node")
    .add_conditional_edges(
        "editor_node",
        approve,
        ["writer_node", END]
    )
    .compile(name="New Graph")
)
