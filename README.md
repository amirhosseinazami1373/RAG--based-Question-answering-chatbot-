# RAG-based question answering chatbot using Deep Lake, LangChain, and GPT-3.5

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that answers user questions by retrieving the most relevant information from a document and generating natural language responses using **GPT-3.5 Turbo**.

The system leverages **Deep Lake** as a vector database to store document embeddings, **LangChain** for seamless LLM integration, and **OpenAI’s GPT-3.5 Turbo** for advanced language understanding. It follows a robust pipeline to convert documents into vector embeddings and find the most relevant chunks through **cosine similarity** to provide precise and context-aware answers to user queries.

---

## Technologies Used

| ![Deep Lake](https://camo.githubusercontent.com/ca89695434d3a14babec7e7a8bbb25f2749d8bb1906225b7fe2f50684ad54ba6/68747470733a2f2f692e706f7374696d672e63632f72736a63576333532f646565706c616b652d6c6f676f2e706e67) | ![Activeloop](https://avatars.githubusercontent.com/u/34816118?s=200&v=4) | ![LangChain](https://raw.githubusercontent.com/langchain-ai/.github/main/profile/logo-dark.svg#gh-light-mode-only) | ![OpenAI GPT-3.5](https://github.com/user-attachments/assets/8985f5fe-dbb5-4692-969e-6141d2721feb) |
|:--:|:--:|:--:|:--:|
| [Deep Lake](https://deeplake.ai/) | [Activeloop](https://activeloop.ai/) | [LangChain](https://www.langchain.com/) | [GPT-3.5](https://beta.openai.com/docs/models/gpt-3-5) |

---

## Features

- **Document Handling:** Download and split large text files into smaller chunks.
- **Vector Embeddings:** Use OpenAI embeddings to transform text chunks.
- **Deep Lake Database Integration:** Store and manage vector embeddings.
- **GPT-3.5 Answer Generation:** Retrieve relevant text chunks and generate context-aware answers.
- **Cosine Similarity Matching:** Quickly find the most relevant content for user queries.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/amirhosseinazami1373/RAG--based-Question-answering-chatbot.git
    cd your-repo
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up API keys:**
   - OpenAI API key: Get your key from [OpenAI](https://beta.openai.com/signup/).
   - Deep Lake API key: Create a Deep Lake account at [Activeloop](https://activeloop.ai/).

4. **Add your API keys as environment variables:**
    ```bash
    export OPENAI_API_KEY="your-openai-api-key"
    export DEEPLAKE_API_KEY="your-deeplake-api-key"
    ```

---

## Usage

1. **Run the Streamlit app:**
    ```bash
    streamlit run RAG_1.py
    ```

2. **Steps in the application:**
   - **Document Processing:** The app downloads the specified text document and splits it into smaller chunks.
   - **Embeddings Generation:** Each chunk is converted into vector embeddings using OpenAI's `text-embedding-ada-002` model.
   - **Storing Embeddings:** The embeddings are uploaded to a **Deep Lake** vector database.
   - **Querying:** Users enter their queries in the app, which converts the query to a vector and uses **cosine similarity** to find the most relevant chunk.
   - **Answer Generation:** The retrieved chunk is fed to **GPT-3.5** to generate the final answer.

---

## Example Query

```text
**Query:** "What clues did Sherlock Holmes find at the crime scene?"
**Response:** "Holmes discovered footprints that indicated a person with a peculiar gait..."
