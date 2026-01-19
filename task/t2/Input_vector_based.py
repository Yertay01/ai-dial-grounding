import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

#TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. You will receive a context containing relevant user information retrieved based on the user's question
2. Answer the user's question using ONLY the information provided in the context
3. If the context doesn't contain enough information to answer the question, say so
4. Be concise but thorough in your response
5. Present the information in a clear and organized manner
"""

#TODO:
# Should consist retrieved context and user question
USER_PROMPT = """## CONTEXT:
{context}

## QUESTION:
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    #TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    result = ["User:"]
    for key, value in user.items():
        result.append(f"  {key}: {value}")
    return "\n".join(result)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        #TODO:
        # 1. Get all users (use UserClient)
        # 2. Prepare array of Documents where page_content is `format_user_document(user)` (you need to iterate through users)
        # 3. call `_create_vectorstore_with_batching` (don't forget that its async) and setup it as obj var `vectorstore`
        
        # 1. Get all users
        user_client = UserClient()
        users = user_client.get_all_users()
        
        # 2. Prepare array of Documents
        documents = [Document(page_content=format_user_document(user)) for user in users]
        
        # 3. Create vectorstore with batching
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        #TODO:
        # 1. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        # 2. Iterate through document batches and create array with tasks that will generate FAISS vector stores from documents:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.afrom_documents
        # 3. Gather tasks with asyncio
        # 4. Create `final_vectorstore` via merge of all vector stores:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.merge_from
        # 6. Return `final_vectorstore`
        
        # 1. Split documents into batches
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # 2. Create tasks for each batch
        tasks = [
            FAISS.afrom_documents(batch, self.embeddings)
            for batch in batches
        ]
        
        # 3. Gather tasks
        vectorstores = await asyncio.gather(*tasks)
        
        # 4. Merge all vectorstores
        final_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            final_vectorstore.merge_from(vs)
        
        # 5. Return final vectorstore
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        #TODO:
        # 1. Make similarity search:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.similarity_search_with_relevance_scores
        # 2. Create `context_parts` empty array (we will collect content here)
        # 3. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then print score and content
        # 4. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        
        # 1. Make similarity search
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        
        # 2. Create context_parts array
        context_parts = []
        
        # 3. Iterate through results
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Score: {relevance_score:.4f}")
            print(f"Content: {doc.page_content}\n")
        
        # 4. Return joined context
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        # TODO: Make augmentation for USER_PROMPT via `format` method
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        #TODO:
        # 1. Create messages array with:
        #       - system prompt
        #       - user prompt
        # 2. Generate response
        #    https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html#langchain_openai.chat_models.azure.AzureChatOpenAI.invoke
        # 3. Return response content
        
        # 1. Create messages array
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]
        
        # 2. Generate response
        response = self.llm_client.invoke(messages)
        
        # 3. Return response content
        return response.content


async def main():

    #TODO:
    # 1. Create AzureOpenAIEmbeddings
    #    embedding model 'text-embedding-3-small-1'
    #    I would recommend to set up dimensions as 384
    # 2. Create AzureChatOpenAI
    
    # 1. Create AzureOpenAIEmbeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="text-embedding-3-small-1",
        dimensions=384
    )
    
    # 2. Create AzureChatOpenAI
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            #TODO:
            # 1. Retrieve context
            # 2. Make augmentation
            # 3. Generate answer and print it
            
            # 1. Retrieve context
            context = await rag.retrieve_context(user_question)
            
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            
            # 3. Generate answer and print it
            answer = rag.generate_answer(augmented_prompt)
            print(f"\nAnswer: {answer}\n")


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce