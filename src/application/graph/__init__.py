"""
Application Layer - LangGraph State Machine
Implements the conversation flow using LangGraph
"""

from operator import add
from typing import Annotated, List, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.domain import ConversationState, RetrievedSection
from src.infrastructure import AzureAISearchVectorStore, OpenAILLM, get_settings


class GraphState(TypedDict):
    """
    LangGraph graph state
    Represents the mutable state during flow execution
    """

    helpdesk_id: int
    project_name: str
    messages: Annotated[List[dict], add]
    current_query: str
    retrieved_context: str
    sections_retrieved: List[dict]
    clarification_count: int
    handover_to_human_needed: bool
    agent_response: str
    is_clarification: bool


class ConversationGraph:
    """
    Orchestrates the conversation flow using LangGraph
    Pattern: State Machine - manages conversation states and transitions
    Principle: Single Responsibility - responsible only for orchestration
    """

    def __init__(self):
        """Initializes the conversation graph"""
        self.settings = get_settings()
        self.vector_store = AzureAISearchVectorStore()
        self.llm = OpenAILLM()
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Builds the conversation state graph

        Flow:
        1. retrieve_context: Retrieves context from vector store
        2. generate_response: Generates agent response
        3. check_clarification: Checks if it's a clarification and updates counter
        4. END: Finishes
        """
        workflow = StateGraph(GraphState)

        # Define the nodes (functions)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("check_clarification", self._check_clarification)

        # Define the edges (flow)
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "check_clarification")
        workflow.add_edge("check_clarification", END)

        # Compile the graph with checkpointer for memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _retrieve_context(self, state: GraphState) -> GraphState:
        """
        Node 1: Retrieves context from vector store
        """
        query = state["current_query"]

        # Search for relevant sections
        sections = self.vector_store.similarity_search(query, k=5)

        # Format the context
        context = "\n\n".join(
            [f"[Score: {section.score:.4f}]\n{section.content}" for section in sections]
        )

        # Update the state
        state["retrieved_context"] = context
        state["sections_retrieved"] = [
            {"score": s.score, "content": s.content} for s in sections
        ]

        return state

    def _generate_response(self, state: GraphState) -> GraphState:
        """
        Node 2: Generates agent response using the LLM
        """
        user_message = state["current_query"]
        context = state["retrieved_context"]

        # Prepare conversation history (exclude current message)
        history = state.get("messages", [])[:-1] if state.get("messages") else []

        # Generate response
        response, is_clarification = self.llm.generate_response(
            user_message=user_message,
            context=context,
            conversation_history=history,
            clarification_count=state["clarification_count"],
            max_clarifications=self.settings.max_clarifications,
        )

        # Update the state
        state["agent_response"] = response
        state["is_clarification"] = is_clarification

        return state

    def _check_clarification(self, state: GraphState) -> GraphState:
        """
        Node 3: Checks if it was a clarification and updates counter
        """
        if state["is_clarification"]:
            state["clarification_count"] += 1

            # Check if limit exceeded
            if state["clarification_count"] >= self.settings.max_clarifications:
                state["handover_to_human_needed"] = True
                # Add message informing about handover
                handover_msg = "\n\n⚠️ I will forward your ticket to a human specialist who can help you better."
                state["agent_response"] += handover_msg

        # Add agent response to messages
        state["messages"].append({"role": "agent", "content": state["agent_response"]})

        return state

    def process_conversation(
        self, conversation: ConversationState
    ) -> ConversationState:
        """
        Processes a complete conversation through the graph

        Args:
            conversation: Current conversation state

        Returns:
            Updated conversation with agent response
        """
        # Get the last user message
        last_message = conversation.messages[-1]

        # Prepare initial graph state
        initial_state: GraphState = {
            "helpdesk_id": conversation.helpdesk_id,
            "project_name": conversation.project_name,
            "messages": [
                {"role": msg.role.value.lower(), "content": msg.content}
                for msg in conversation.messages
            ],
            "current_query": last_message.content,
            "retrieved_context": "",
            "sections_retrieved": [],
            "clarification_count": conversation.clarification_count,
            "handover_to_human_needed": conversation.handover_to_human_needed,
            "agent_response": "",
            "is_clarification": False,
        }

        # Execute the graph
        config: RunnableConfig = {
            "configurable": {"thread_id": str(conversation.helpdesk_id)}
        }
        final_state = self.graph.invoke(initial_state, config)

        # Update conversation with results
        conversation.add_agent_message(final_state["agent_response"])
        conversation.clarification_count = final_state["clarification_count"]
        conversation.handover_to_human_needed = final_state["handover_to_human_needed"]

        # Add retrieved sections
        sections = [
            RetrievedSection(score=s["score"], content=s["content"])
            for s in final_state["sections_retrieved"]
        ]
        conversation.add_retrieved_sections(sections)

        return conversation
