from typing import List, Literal

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from vector_store import VectorStore

# NOTE: This is not being used as of now
# TODO: integrate its functionality with VectorStore class

load_dotenv(override=True)

# Data models
class RouteQuery(BaseModel):
    """Routes a user query to the most relevant datasource."""

    datasource: Literal["vectorstore"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of graph.

    Attributes:
        question: question
        generation: LLM generated completion
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

class RAGSystem:
    """
    This class contains the RAG system for routing and grading user queries, and generating appropriate completions from those documents.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retriever = vector_store.db_client.as_retriever()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.setup_router()
        self.setup_grader()
        self.setup_generator()
        
    def setup_router(self):
        # Setting up Router for routing queries to vector store
        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        system = """You are an expert at routing a user question to a vectorstore.
        The vectorstore contains documents related to quotes and their meanings.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        self.question_router = route_prompt | structured_llm_router
        
    def setup_grader(self):
        # Setting up document grader for grading relevance of retrieved documents
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        self.retrieval_grader = grade_prompt | structured_llm_grader
        
    def setup_generator(self):
        prompt = hub.pull("rlm/rag-prompt")

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        self.rag_chain = prompt | llm | StrOutputParser()


    def retrieve(self, state):
        """
        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}


    def generate(self, state):
        """
        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def route_question(self, state: RouteQuery):
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        question = state["question"]
        input_data = {"question": question, "datasource": "vectorstore"}
        source = self.question_router.invoke(input_data)
        if source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            print("---TRYING TO ROUTE QUESTIONS TO WEB SEARCH---")
            return "web-search"

