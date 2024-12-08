import re
import json
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from langchain.memory import ConversationBufferMemory

import operator
from typing_extensions import TypedDict
from typing import List, Annotated

from bs4 import BeautifulSoup
import requests


# LLM model name
MODEL = "llama3.2"
# Model Creativity
TEMPERATURE = 0
# Model for embeddings
EMBEDDINGS_MODEL = "nomic-embed-text-v1.5"

class Chatbot:
    def __init__(self, 
                 model=MODEL, 
                 temperature=TEMPERATURE, 
                 embeddings_model=EMBEDDINGS_MODEL,
                 search_retrieval=3):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.llm_json = ChatOllama(model=model, temperature=temperature, format="json")
        self.vectorstore = self.set_vectorstore(embeddings=embeddings_model)
        self.retriever = self.set_retriever(search_retrieval)
        self.graph = self.set_workflow()
        self.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        ### === Instructions === ###
        ### Router
        self.router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

        The vectorstore contains documents related to Metanoia and IT consulting.

        Use the vectorstore for questions on these topics. Else, use web-search.

        Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""


        ### Retrieval Grader
        self.doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

        self.doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

        This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

        Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


        ### Generate
        self.rag_prompt = """You are an assistant for question-answering tasks. 

        Here is the context to use to answer the question:

        {context} 

        Think carefully about the above context. 

        Now, review the user question:

        {question}

        Provide an answer to this questions using only the above context. 

        Use three sentences maximum and keep the answer concise.

        Answer:"""


        ### Hallucination Grader
        self.hallucination_grader_instructions = """You are a teacher grading a quiz. 

        You will be given FACTS and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

        self.hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""


        ### Answer Grader
        self.answer_grader_instructions = """You are a teacher grading a quiz. 

        You will be given a QUESTION and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) The STUDENT ANSWER helps to answer the QUESTION

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

        self.answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

    
    class GraphState(TypedDict):
        """
        Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
        """

        question: str  # User question
        generation: str  # LLM generation
        web_search: str  # Binary decision to run web search
        max_retries: int  # Max number of retries for answer generation
        answers: int  # Number of answers generated
        loop_step: Annotated[int, operator.add]
        documents: List[str]  # List of retrieved documents
    
    def set_vectorstore(self, embeddings=EMBEDDINGS_MODEL):
        """
        Setup the vectorstore.

        Args:
            docs (list[Document]): List of documents
        """

        # Add to vectorDB
        embeddings = NomicEmbeddings(model=embeddings, inference_mode="local", device="nvidia")
        
        vectorstore = Qdrant.from_existing_collection(
            path="./KnowledgeBase",
            embedding=embeddings,
            collection_name="KB"
        )
        return vectorstore

    def add_docs(self, docs):
        """

        Args:
            docs (list[Document]): List of documents
        """

        # Add documents to the vectorstore
        self.vectorstore.add_documents(docs)

    def set_retriever(self, k=3):
        """

        Args:
            k (int): number of documents to retrieve
        """

        # Create retriever
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve_KB(self, query):
        """
        Get the response from the chatbot.

        Args:
            query (str): User query

        Returns:
            str: Chatbot response
        """
        return self.retriever.invoke(query)
    
    # Post-processing
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    ### === Workflow === ###    
    def set_workflow(self):
        workflow = StateGraph(self.GraphState)

        ### Nodes
        workflow.add_node("websearch", self.web_search)  # web search
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        ### edges
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
                "max retries": END,
            },
        )

        graph = workflow.compile()
        return graph
    
    ### === Node Functions === ###
    def retrieve(self, state):
        """
        Retrieve documents from vectorstore

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Write retrieved documents to documents key in state
        documents = self.retriever.invoke(question)
        return {"documents": documents}
    
    def generate(self, state):
        """
        Generate answer using RAG on retrieved documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        loop_step = state.get("loop_step", 0)

        # RAG generation
        docs_txt = self.format_docs(documents)
        rag_prompt_formatted = self.rag_prompt.format(context=docs_txt, question=question)
        generation = self.llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        return {"generation": generation, "loop_step": loop_step + 1}
    
    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question
        If any document is not relevant, we will set a flag to run web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        web_search = state["web_search"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            doc_grader_prompt_formatted = self.doc_grader_prompt.format(
                document=d.page_content, question=question
            )
            result = self.llm_json.invoke(
                [SystemMessage(content=self.doc_grader_instructions)]
                + [HumanMessage(content=doc_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                if web_search:
                    web_search = "Yes"
                continue
        if filtered_docs or not web_search:
            web_search = "No"
        return {"documents": filtered_docs, "web_search": web_search}
    
    # Web Search
    def search_url(query, num_results=3):
        """
        Get search results from Google

        Args:
            query (str): The search query

        Returns:
            urls (list): List of URLs from the search
        """

        question = query
        search_url = 'https://www.google.com/search'

        headers = {
            'Accept' : '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
        }
        parameters = {'q': question}

        content = requests.get(search_url, headers = headers, params = parameters).text
        soup = BeautifulSoup(content, 'html.parser')

        search = soup.find(id = 'search')
        links = search.find_all('a')
        
        return [link.get('href') for link in links if link.get('href', '').startswith('https://')][:num_results]

    def web_search(self, state):
        """
        Web search based based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to documents
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])

        # Web search
        docs = load_doc_url(self.search_url(question))
        for doc in docs:
            documents.append(doc)
        return {"documents": documents}
    
    ### === Edge Functions === ###
    def route_question(self, state):
        """
        Route question to web search or RAG

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("---ROUTE QUESTION---")
        route_question = self.llm_json.invoke(
            [SystemMessage(content=self.router_instructions)]
            + [HumanMessage(content=state["question"])]
        )
        source = json.loads(route_question.content)["datasource"]
        if source == "websearch":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or add web search

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
        
    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

        hallucination_grader_prompt_formatted = self.hallucination_grader_prompt.format(
            documents=self.format_docs(documents), generation=generation.content
        )
        result = self.llm_json.invoke(
            [SystemMessage(content=self.hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            # Test using question and generation from above
            answer_grader_prompt_formatted = self.answer_grader_prompt.format(
                question=question, generation=generation.content
            )
            result = self.llm_json.invoke(
                [SystemMessage(content=self.answer_grader_instructions)]
                + [HumanMessage(content=answer_grader_prompt_formatted)]
            )
            grade = json.loads(result.content)["binary_score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] <= max_retries:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
            else:
                print("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    
    def show_graph(self):
        """
        Display the graph
        """
        return display(Image(self.graph.get_graph().draw_mermaid_png()))
    
    ### === Chat History Functions === ###
    def update_chat_history(self,user_message, bot_message):
        """
        Function to update the chat history after each query
        
        Args:
            user_message (str): The user question
            bot_message (str): The bot response
        """
        # Save context to memory after each question/response
        self.chat_memory.save_context({"input": user_message}, {"output": bot_message})
        # Optionally, print the updated chat history for debugging
        #print(f"Updated Chat History:\n{chat_memory.load_memory_variables({})['chat_history']}")

    def get_chat_history(self):
        """
        Function to get the chat history
        
        Returns:
            list: The chat history
        """
        return self.chat_memory.load_memory_variables({})["chat_history"]

    def print_chat_history(self,):
        """
        Function to print the chat history
        
        Args:
            chat_memory (ConversationBufferMemory): The chat memory
        """
        chat_history = self.get_chat_history()
        for entry in chat_history:
            print(f"{entry.type}: {entry.content}\n")

    def ask_question(self, question, max_retries=1, web_search=True):
        """
        Ask a question and get an answer
        
        Args:
            question (str): The user question
            max_retries (int): Maximum number of retries for answer generation
            web_search (bool): True to enable web search even if there are some relevant documents, False to disable
        
        Returns:
            str: Answer to the question
        """
        inputs = {"question": question, "max_retries": max_retries, "web_search": web_search}
        events = []

        # Iterate through the graph, streaming events
        for event in graph.stream(inputs, stream_mode="values"):
            events.append(event)

        # Get the final answer
        bot_message = events[-1]["generation"].content
        self.update_chat_history(question, bot_message)
        
        # Return the final generated answer
        return bot_message

def load_doc_url(urls, chunk_size=500, chunk_overlap=100):
        """
        Get and clean content from URLs. Split the content into chunks.

        Args:
            urls (list[str]): List of URLs
            chunk_size (int): Size of the chunks
            chunk_overlap (int): Overlap between chunks
        
        Returns:
            list[Document]: List of splitted documents
        """
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Remove unnecessary spaces from the page content
        for doc in docs_list:
            doc.page_content = re.sub(r"(\n|\t| )+", r"\1", doc.page_content)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs_list)

if __name__ == "__main__":
    chatbot = Chatbot()
    #chatbot.add_docs([Document(page_content="Dummy Document")])
    #print(chatbot.retrieve_KB("Dummy"))

    graph = chatbot.graph
    chatbot.show_graph()
    inputs = {"question": "Tell me about metanoia IT and it's customers",
              'web_search':True}
    chatbot.ask_question(**inputs)
    chatbot.print_chat_history()