# Chat Bot with Retrieval-Augmented Generation (RAG)

This included jupyter notebook file (chatbot.ipynb), UI using react, and backend using python.

## Installation
### Backend
1. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run backend server:
    ```
    python backend.py
    ```

### Frontend
1. Navigate to frontend directory
    ```
    cd frontend
    ```
2. Install the required dependencies:
    ```
    npm install
    ```
3. Start the frontend development server:
    ```
    npm start
    ```

P.S. You can choose vector space between in memory or local storage by adjusting in **Data Preparation and Knowledge Base** section.
#### In Memory:
```python
vectorstore = Qdrant.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    location=":memory:",
    collection_name="KB"
)
```
#### In local storage:
```python
vectorstore = Qdrant.from_documents(
    documents=doc_splits,
    embedding=embeddings,
    path="./KnowledgeBase",
    collection_name="KB"
)
```

## Usage
Once both the backend and frontend servers are running, open your web browser and navigate to http://localhost:3000. You should see the chatbot interface.

- Turn on/off web search button to enable/disable web search even if there are some relevant documents
- Type your message in the input field and press Enter or click the "Send" button.
- The chatbot will process your message and respond based on the session context and retrieved information from the vectorstore or web search.

> **Note:** the UI is cloned and modified from https://github.com/ nydasco/rag_based_chatbot
