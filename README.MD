# Delta ChatBot

A Streamlit application for asking questions about uploaded PDF and plain-text documents. It combines document loading, text chunking, embeddings, Chroma retrieval, and a conversational LangChain chain.

## Start locally

The original project uses Python 3.10.10 and Poetry 1.4.0. Dependencies are recorded in [`pyproject.toml`](pyproject.toml) and `poetry.lock`.

```bash
git clone https://github.com/ataturhan21/Delta-ChatBot.git
cd Delta-ChatBot
poetry install --no-root
poetry run streamlit run "delta-chatbot/app.py"
```

Run from the repository root so that `static/` assets resolve. On Windows, [`start.bat`](start.bat) sets the working directory for you. The application is run directly from its source directory; `--no-root` installs its dependencies without packaging that directory.

## Use the application

1. Choose an OpenAI or Hugging Face model in the sidebar.
2. Enter the matching provider credential in the masked API-key field.
3. Upload PDF or UTF-8 TXT files. The uploader accepts the formats handled by the document loader.
4. Ask questions in the chat and expand the source-document panel to inspect retrieved passages.

Each upload is written to an isolated temporary directory while it is parsed, then the temporary files are removed. Text is split into chunks of 256 characters with an overlap of 32 before retrieval.

Embedding and language-model calls use the selected external provider. The original model list and API integrations are preserved; provider access and compatibility need verification before deployment. The application uses process-level provider settings and is intended for a local, single-user session.

## Project map

| Path | Purpose |
| --- | --- |
| [`delta-chatbot/app.py`](delta-chatbot/app.py) | Upload processing, retrieval, model selection, and chat UI |
| [`delta-chatbot/modules/utils.py`](delta-chatbot/modules/utils.py) | Page configuration and shared styling |
| [`static/`](static/) | Background images and CSS |
| [`pyproject.toml`](pyproject.toml) | Original dependency constraints |

## Development

Keep credentials outside version control. When changing the upload pipeline, check valid PDFs and text files, unsupported extensions, duplicate upload names, and temporary-file cleanup. A live model response also depends on provider credentials and service availability.

## License

[GNU General Public License v3.0](LICENSE).
