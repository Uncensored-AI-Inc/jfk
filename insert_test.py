# %%
import uuid
import requests
import json
from supabase import create_client


# %% Configuration
SUPABASE_URL = "http://127.0.0.1:54321"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
TEI_URL = "http://localhost:8011/embed"


# %%
def test_single_document_insertion():
    """Test inserting a single document into Supabase."""

    print("Starting single document insertion test...")

    # 1. Initialize Supabase client
    print("Initializing Supabase client...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 2. Create a sample document
    print("Creating sample document...")
    sample_text = "This is a test document about JFK for testing Supabase insertion."
    sample_metadata = {
        "source": "test_script",
        "filename": "test_document",
        "file_path": "test/path.md",
        "element_type": "Text",
        "element_id": 1,
    }

    # 3. Get embedding for the sample document
    print("Getting embedding from the TEI service...")
    try:
        response = requests.post(
            TEI_URL,
            json={"inputs": sample_text},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            print(f"Error from TEI service: {response.status_code} - {response.text}")
            return

        embedding_nested = response.json()

        # The embedding is coming back as a nested array [[values]] - we need to flatten it
        if (
            isinstance(embedding_nested, list)
            and len(embedding_nested) > 0
            and isinstance(embedding_nested[0], list)
        ):
            embedding = embedding_nested[0]  # Take the first element of the outer array
        else:
            embedding = embedding_nested  # Just in case it's already flat

        print(f"Got embedding with {len(embedding)} dimensions")

        # Print a few values to verify it's a proper embedding and is now flattened
        print(f"Sample embedding values (flattened): {embedding[:5]}...")

    except Exception as e:
        print(f"Error getting embedding: {e}")
        return

    # 4. Prepare document for insertion
    document_id = str(uuid.uuid4())
    document = {
        "id": document_id,
        "content": sample_text,
        "metadata": sample_metadata,
        "embedding": embedding,
    }

    # 5. Insert document into Supabase
    print(f"Inserting document with ID: {document_id}...")

    try:
        # First, let's try a simple query to see if the table exists
        try:
            print("Checking if jfk_documents table exists...")
            table_query = (
                supabase.table("jfk_documents").select("id").limit(1).execute()
            )
            print(f"Table exists. Sample query returned: {table_query.data}")
        except Exception as table_error:
            print(f"Error checking table: {table_error}")
            print(
                "This might indicate the table doesn't exist or has a different name."
            )
            return

        # Try to insert the document
        print("Attempting to insert document...")
        insert_response = supabase.table("jfk_documents").insert(document).execute()

        # Print the entire response for debugging
        print(
            f"Insert response data: {insert_response.data if hasattr(insert_response, 'data') else 'No data'}"
        )

        # Check if document was inserted
        print("Verifying insertion...")
        verify_query = (
            supabase.table("jfk_documents")
            .select("id", "content")
            .eq("id", document_id)
            .execute()
        )

        if hasattr(verify_query, "data") and verify_query.data:
            print("SUCCESS: Document was inserted and retrieved successfully!")
            print(f"Retrieved document: {verify_query.data[0]}")
        else:
            print("ERROR: Document was not found after insertion.")
            print(f"Verification query response: {verify_query}")

    except Exception as e:
        print(f"Error during insertion or verification: {e}")


if __name__ == "__main__":
    test_single_document_insertion()
