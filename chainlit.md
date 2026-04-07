# AI Chatbot Assistant 🤖

This app is a Retrieval-Augmented Generation (RAG) chatbot built with Chainlit, LlamaIndex and LangGraph. It can answer questions using either your local documents or web sources.

## How It Works
- The project uses local Ollama models as selectable LLMs.
- Upload local files and ask questions about their content.
- You can also upload temporary session files for a single chat session.
- Session files are cleaned up automatically when the chat ends or the app restarts, unless you choose to persist them.
- The chatbot answers using either local documents or web information, depending on the selected RAG mode.
- You can set the RAG mode to `AI Decides`, `Local`, or `Web`.
- Sources are shown when available to support transparency and traceability.
- Responses are grounded in retrieved sources to reduce LLM hallucinations.

## Usage
1. Ask a question about your local files or a topic that requires web information.
2. Upload session files if you want to include additional temporary documents.
3. Review the cited sources in the response.
4. Change settings like temperature or model to improve performance.