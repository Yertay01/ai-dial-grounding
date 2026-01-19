import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


# Pydantic models for structured output
class HobbiesExtraction(BaseModel):
    """Model for extracting hobbies and associated user IDs"""
    hobbies: dict[str, list[int]] = Field(
        description="Dictionary mapping hobby names to lists of user IDs who have that hobby"
    )


class HobbiesSearchWizard:
    """
    HOBBIES SEARCHING WIZARD
    Searches users by hobbies and provides their full info in JSON format.
    
    Features:
    1. Embeds only user `id` and `about_me` to reduce context window
    2. Adaptive vector store updates (remove deleted users, add new ones on each request)
    3. Uses LLM for Named Entity Extraction (NEE) of hobbies
    4. Output grounding: validates user IDs and fetches full user info
    5. Returns JSON grouped by hobbies
    """
    
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore: Optional[Chroma] = None
        self.parser = PydanticOutputParser(pydantic_object=HobbiesExtraction)
        
    async def __aenter__(self):
        print("🔎 Loading all users and initializing vector store...")
        # Cold start: load all users and create vector store
        await self._initialize_vectorstore()
        print("✅ Vector store is ready.")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _format_user_document(self, user: dict[str, Any]) -> str:
        """Format user data for embedding - only ID and about_me to reduce context"""
        return f"User ID: {user['id']}\nAbout: {user.get('about_me', 'No information')}"
    
    async def _initialize_vectorstore(self):
        """Cold start: Create vector store with all current users"""
        users = self.user_client.get_all_users()
        
        # Create documents with only id and about_me
        documents = []
        for user in users:
            doc = Document(
                page_content=self._format_user_document(user),
                metadata={"user_id": user['id']}
            )
            documents.append(doc)
        
        # Create Chroma vectorstore
        self.vectorstore = await Chroma.afrom_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="user_hobbies"
        )
        
        print(f"Initialized vector store with {len(documents)} users")
    
    async def _update_vectorstore(self):
        """
        Adaptive update: sync vector store with current users in the service.
        Removes deleted users and adds new ones.
        """
        print("🔄 Updating vector store...")
        
        # Get current users from service
        current_users = self.user_client.get_all_users()
        current_user_ids = {user['id'] for user in current_users}
        
        # Get existing IDs in vector store
        existing_data = self.vectorstore.get()
        existing_ids = set()
        if existing_data and existing_data.get('metadatas'):
            existing_ids = {meta['user_id'] for meta in existing_data['metadatas'] if meta}
        
        # Find deleted and new users
        deleted_ids = existing_ids - current_user_ids
        new_user_ids = current_user_ids - existing_ids
        
        # Remove deleted users
        if deleted_ids:
            # Delete by filtering metadata
            ids_to_delete = []
            if existing_data and existing_data.get('ids') and existing_data.get('metadatas'):
                for idx, meta in enumerate(existing_data['metadatas']):
                    if meta and meta.get('user_id') in deleted_ids:
                        ids_to_delete.append(existing_data['ids'][idx])
            
            if ids_to_delete:
                self.vectorstore.delete(ids=ids_to_delete)
                print(f"  Removed {len(deleted_ids)} deleted users")
        
        # Add new users
        if new_user_ids:
            new_users = [user for user in current_users if user['id'] in new_user_ids]
            new_documents = []
            for user in new_users:
                doc = Document(
                    page_content=self._format_user_document(user),
                    metadata={"user_id": user['id']}
                )
                new_documents.append(doc)
            
            await self.vectorstore.aadd_documents(new_documents)
            print(f"  Added {len(new_user_ids)} new users")
        
        if not deleted_ids and not new_user_ids:
            print("  No updates needed")
    
    async def _retrieve_relevant_users(self, query: str, k: int = 20) -> list[dict]:
        """Retrieve relevant user documents based on hobby query"""
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, 
            k=k
        )
        
        retrieved_users = []
        for doc, score in results:
            print(f"  Score: {score:.4f} | {doc.page_content[:100]}...")
            retrieved_users.append({
                'content': doc.page_content,
                'user_id': doc.metadata.get('user_id'),
                'score': score
            })
        
        return retrieved_users
    
    def _extract_hobbies_with_llm(self, user_context: str, query: str) -> dict[str, list[int]]:
        """
        Use LLM for Named Entity Extraction (NEE) to identify hobbies and associated user IDs.
        Returns format: {"hobby_name": [user_id1, user_id2, ...]}
        """
        system_prompt = """You are an expert at extracting hobbies from user profiles and grouping users by their hobbies.

Your task:
1. Analyze the provided user profiles in the context
2. Extract hobbies/interests that are relevant to the user's query
3. Group user IDs by the hobbies they have
4. Return ONLY user IDs that actually appear in the provided context
5. Use descriptive hobby names (e.g., "hiking", "rock climbing", "photography")

IMPORTANT: Only include user IDs that are explicitly mentioned in the context. Do not make up or hallucinate user IDs."""

        user_prompt = f"""## USER QUERY:
{query}

## USER PROFILES CONTEXT:
{user_context}

{self.parser.get_format_instructions()}

Extract hobbies relevant to the query and group user IDs by hobby."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm_client.invoke(messages)
        
        try:
            parsed = self.parser.parse(response.content)
            return parsed.hobbies
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response.content}")
            return {}
    
    async def _output_grounding(self, hobbies_dict: dict[str, list[int]]) -> dict[str, list[dict]]:
        """
        Output grounding: Verify user IDs exist and fetch full user information.
        Returns: {"hobby": [full_user_info_dict, ...]}
        """
        result = {}
        
        for hobby, user_ids in hobbies_dict.items():
            full_users = []
            for user_id in user_ids:
                try:
                    # Fetch full user info and verify user exists
                    user_data = await self.user_client.get_user(user_id)
                    full_users.append(user_data)
                except Exception as e:
                    print(f"  Warning: Could not fetch user {user_id}: {e}")
                    # Skip users that don't exist (output grounding verification)
                    continue
            
            if full_users:
                result[hobby] = full_users
        
        return result
    
    async def search_users_by_hobbies(self, query: str) -> dict[str, list[dict]]:
        """
        Main method: Search users by hobbies based on natural language query.
        
        Returns: {"hobby_name": [full_user_info, ...]}
        """
        print(f"\n🔍 Searching for: {query}")
        
        # Step 1: Update vector store (adaptive grounding)
        await self._update_vectorstore()
        
        # Step 2: Retrieve relevant users from vector store
        print("📚 Retrieving relevant users...")
        relevant_users = await self._retrieve_relevant_users(query)
        
        if not relevant_users:
            print("No relevant users found")
            return {}
        
        # Step 3: Prepare context for LLM
        user_context = "\n\n".join([user['content'] for user in relevant_users])
        
        # Step 4: Extract hobbies with LLM (NEE - Named Entity Extraction)
        print("🤖 Extracting hobbies with LLM...")
        hobbies_dict = self._extract_hobbies_with_llm(user_context, query)
        print(f"  Extracted hobbies: {list(hobbies_dict.keys())}")
        
        # Step 5: Output grounding - verify IDs and fetch full user info
        print("✅ Performing output grounding (fetching full user info)...")
        result = await self._output_grounding(hobbies_dict)
        
        return result


async def main():
    """Main function to run the Hobbies Search Wizard"""
    
    # Initialize embeddings and LLM
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="text-embedding-3-small-1",
        dimensions=384
    )
    
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o",
        temperature=0
    )
    
    async with HobbiesSearchWizard(embeddings, llm_client) as wizard:
        print("\n" + "="*60)
        print("🎯 HOBBIES SEARCHING WIZARD")
        print("="*60)
        print("\nQuery examples:")
        print("  - I need people who love to go to mountains")
        print("  - Find users interested in photography and art")
        print("  - Who likes outdoor activities?")
        print("\nType 'quit' or 'exit' to stop.\n")
        
        while True:
            user_query = input("> ").strip()
            if user_query.lower() in ['quit', 'exit']:
                break
            
            if not user_query:
                continue
            
            # Search users by hobbies
            result = await wizard.search_users_by_hobbies(user_query)
            
            # Display results
            print("\n" + "="*60)
            print("📊 RESULTS:")
            print("="*60)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("No users found matching the query")
            print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())



