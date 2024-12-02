# Bedrock-Llama-Index Docs Chat

This project is a simple **Retrieval-Augmented Generation (RAG)** example that leverages **AWS Bedrock** and **Llama-Index** for document-based question answering. The application indexes any documents provided in the `data` folder and allows users to interact with the indexed content via a chat interface.

![Demo](demo.gif)

## Features

- **Document Indexing**: Automatically indexes all documents placed in the `data` folder.
- **Chat Interface**: Streamlit-powered web application for user-friendly interaction.
- **RAG Workflow**: Combines retrieval techniques with **Claude Sonnet** via AWS Bedrock to provide context-aware responses.
- **Dynamic Document Support**: Indexes and processes documents in various formats like PDFs and text files.

## Getting Started

### Prerequisites

1. **Python**: Ensure you have Python 3.8 or later installed.
2. **AWS Credentials**: Valid AWS credentials with access to AWS Bedrock.
3. **Dependencies**: Install required Python libraries using `requirements.txt`.

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd bedrock-llama-index-docs-chat-main
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Index Documents

Place your documents in the `data` folder. Supported formats include:

- PDFs (e.g., `Amazon Bedrock - User Guide.pdf`)
- Text files (e.g., `sample.txt`)

Run the indexing script:

```bash
python create_index.py
```

This will preprocess the documents and create an index file for efficient querying.

#### 2. Start the Chat Application

Launch the Streamlit application to interact with your indexed documents:

```bash
streamlit run chat_app.py
```

Once launched, open the URL provided by Streamlit in your web browser. The chat interface allows you to ask questions about the documents you’ve indexed.

### Architecture

The project implements a simple RAG approach:

1. **Document Retrieval**: Retrieves relevant context from indexed documents.
2. **Augmented Generation**: Passes the retrieved context to Claude Sonnet (via AWS Bedrock) to generate responses.
3. **Streamlit Interface**: Provides a seamless user interface for interaction.

### Example Workflow

1. Add a PDF guide or a text file to the `data` folder.
2. Run `create_index.py` to preprocess and index the documents.
3. Use the chat interface to ask questions like:
   - "What are the key features of Amazon Bedrock?"
   - "How does Claude Sonnet handle document queries?"

### Customization

- **Data Folder**: Add your own documents to the `data` directory.
- **Model Configuration**: Modify `chat_app.py` to customize the Bedrock model parameters or endpoint.
