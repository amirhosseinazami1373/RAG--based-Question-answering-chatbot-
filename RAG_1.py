import os
import requests
import deeplake
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Streamlit app title
st.title("RAG-based Question Answering with Deep Lake & OpenAI")

# Step 1: Ask for API keys from the user
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
deeplake_token = st.text_input("Enter your Deep Lake Token", type="password")

if openai_api_key and deeplake_token:
    # Set the API keys
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Step 2: Download and save the text file
    file_url = "https://sherlock-holm.es/stories/plain-text/stud.txt"
    response = requests.get(file_url)

    if response.status_code != 200:
        st.error(f"Failed to download the file. Status code: {response.status_code}")
    else:
        file_path = 'downloaded_example.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        st.success(f"File downloaded and saved to {file_path}")

        # Step 3: Load and process the text file using LangChain's TextLoader
        loader = TextLoader(file_path)
        documents = loader.load()

        # Step 4: Split the document into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        st.write(f"Document split into {len(docs)} chunks.")

        # Step 5: Initialize the OpenAI embeddings
        embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")

        # Step 6: Load Deep Lake dataset
        activeloop_org_id = "YOUR_ORGANIZATION_NAME"
        activeloop_dataset_name = "YOUR_DATASET_NAME"
        dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
        db = deeplake.load(dataset_path, token=deeplake_token)
        st.success("Connected to Deep Lake dataset.")

        # Step 7: Query input from the user
        query = st.text_input("Enter your query:")

        if query:
            # Embed the query using OpenAI embeddings
            embedded_query = embeddings.embed_query(query)

            # Compute cosine similarity between the query and stored embeddings
            cosine_sim_matrix = cosine_similarity([embedded_query], db["embedding"].numpy())

            # Get the most relevant result
            most_relevant_index = int(cosine_sim_matrix.argmax())

            # Retrieve the most relevant text from the dataset
            try:
                most_relevant_text = db["text"][most_relevant_index].numpy()
                if isinstance(most_relevant_text, np.ndarray):
                    most_relevant_text = most_relevant_text.item()
            except Exception as e:
                st.error(f"Error retrieving the most relevant result: {str(e)}")
                most_relevant_text = ""

            # Define the prompt template
            template = """
            You are a helpful chatbot assisting readers with their questions based on the provided article.
            Use only the information below to answer the question. Do not invent or hallucinate any information.

            Context:
            {chunks_formatted}

            Question: {query}

            Answer:
            """

            prompt = PromptTemplate(
                input_variables=["chunks_formatted", "query"],
                template=template
            )

            # Format the context to inject into the prompt
            chunks_formatted = most_relevant_text if isinstance(most_relevant_text, str) else "\n\n".join(most_relevant_text)
            prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

            # Generate the answer using the LLM
            llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)
            try:
                answer = llm(prompt_formatted)
                st.write("Answer:", answer)
            except Exception as e:
                st.error(f"Error generating the answer: {str(e)}")
else:
    st.warning("Please enter both OpenAI API Key and Deep Lake Token.")
