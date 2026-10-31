import os
import tempfile
import time
import uuid
from collections import OrderedDict

import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from modules.utils import add_bg_from_local, set_page_config

STREAMING_INTERVAL = 0.01


if "messages" not in st.session_state:
    st.session_state.messages = OrderedDict()
if "chain" not in st.session_state:
    st.session_state.chain = None


def is_api_key_valid(model_host: str, api_key: str) -> bool:
    """
    Check if the provided API key is valid for the specified model host.

    Parameters:
        model_host (str): The name of the model host. Possible values are "openai" or "huggingface".
        api_key (str): The API key to be validated.

    Returns:
        bool: True if the API key is valid for the specified model host; False otherwise.
    """
    if api_key is None:
        st.sidebar.warning(
            f"Please enter a valid {model_host.title()} API Key!", icon="⚠"
        )
        return False
    elif model_host == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning("Please enter a valid OpenAI API Key!", icon="⚠")
        return False
    elif model_host == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning(
            "Please enter a valid HuggingFace API Key!", icon="⚠"
        )
        return False
    else:
        return model_host in {"openai", "huggingface"}


def create_vector_store_retriever(
    model_host, chunked_documents, api_key
) -> Chroma:
    """
    Create a vector store retriever for the given model host and chunked documents.

    Parameters:
        model_host (str): The model host for vector embeddings. Choose between "openai"
                          for OpenAI Embeddings or "huggingface" for Hugging Face Hub
                          Embeddings.
        chunked_documents (list): A list of text chunks obtained from documents. Each
                                  chunk should be a string representation of a document's
                                  part.

    Returns:
        vector_store.retriever: A retriever object capable of performing document retrieval
                                using vector embeddings.
    """
    embeddings = (
        OpenAIEmbeddings(openai_api_key=api_key)
        if model_host == "openai"
        else HuggingFaceHubEmbeddings(huggingfacehub_api_token=api_key)
    )
    vector_store = Chroma.from_documents(
        chunked_documents,
        embeddings,
        collection_name=f"delta-{uuid.uuid4().hex}",
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


def create_llm(model, api_key) -> ChatOpenAI | HuggingFaceHub:
    """
    Create a language model (LLM) for chat-based applications based on the specified model.

    Parameters:
        model (str): The model name or repository ID of the language model to be used.

    Returns:
        Union[ChatOpenAI, HuggingFaceHub]: A language model instance suitable for chat-based applications.
    """
    return (
        ChatOpenAI(
            model_name=model.split("/")[1],
            openai_api_key=api_key,
            temperature=0,
        )
        if model.startswith("openai")
        else HuggingFaceHub(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature": 0.1,
                "max_length": 4096,
            },
        )
    )


def create_main_prompt() -> str:
    """
    Create the main prompt for the chatbot to respond to user queries about TOBB ETÜ (TOBB University).

    Returns:
        str: The main prompt template for the chatbot.
    """
    return """
    <|SYSTEM|>#
    You are a bot that lets your user talk to the documents he has uploaded.
    - Answer using the information in the documents provided and your chat history.
    - If the answers are not in the documentation and your chat history, you should say "Sorry, I could not find the answer to this question
    in your uploaded files.".
    <|USER|>
    Now the user asks you a question. Using your knowledge of the context and chat history provided to you, answer this question
    clearly and precisely.

    QUESTION: {question}
    CONTEXT:
    {context}

    ANSWER: <|ASSISTANT|>
    """


def create_retrieval_qa(
    llm, prompt_template, retriever
) -> ConversationalRetrievalChain:
    """
    Create a Conversational Retrieval Chain for question-answering based on the provided components.

    Parameters:
        llm (Union[ChatOpenAI, HuggingFaceHub]): The language model used for conversational responses.
        prompt_template (str): The main prompt template for the chatbot to respond to user queries.
        retriever (vector_store.retriever.Retriever): The retriever object for document retrieval.

    Returns:
        langchain.ConversationalRetrievalChain: A Conversational Retrieval Chain for question-answering.

    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    combine_docs_chain_kwargs = {"prompt": PROMPT}
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        memory=memory,
        return_source_documents=True,
    )


def prepare_documents(uploaded_files):
    """Load supported uploads and remove temporary files after parsing."""
    documents = []
    loaders = {".pdf": PyPDFLoader, ".txt": TextLoader}

    with tempfile.TemporaryDirectory() as temp_dir:
        for index, uploaded_file in enumerate(uploaded_files):
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            if extension not in loaders:
                raise ValueError(f"Unsupported document format: {extension}")

            # Generated names avoid collisions and keep uploads inside temp_dir.
            temp_file_path = os.path.join(
                temp_dir, f"upload_{index}{extension}"
            )
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())

            loader_class = loaders[extension]
            if extension == ".txt":
                loader = loader_class(temp_file_path, encoding="utf-8")
            else:
                loader = loader_class(temp_file_path)
            documents.extend(loader.load())

    character_splitter = CharacterTextSplitter(
        chunk_size=256, chunk_overlap=32
    )
    return character_splitter.split_documents(documents)


def clean_chain():
    st.session_state.chain = None
    st.session_state.messages = OrderedDict()
    return


def main():
    set_page_config()

    background_img_path = os.path.join("static", "background", "Bot.png")
    sidebar_background_img_path = os.path.join(
        "static", "background", "Lila Gradient.png"
    )
    page_markdown = add_bg_from_local(
        background_img_path=background_img_path,
        sidebar_background_img_path=sidebar_background_img_path,
    )
    st.markdown(page_markdown, unsafe_allow_html=True)

    style_path = os.path.join("static", "style.css")
    with open(style_path) as s:
        st.markdown(f"<style>{s.read()}</style>", unsafe_allow_html=True)

    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 60px;'>
         📝 Delta - Document ChatBot
         </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 20px;'>
        You can upload your files and start to chat with them
        </h1>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<center><h1>ChatBot Configurations</h1></center> <br>",
        unsafe_allow_html=True,
    )

    model = st.sidebar.selectbox(
        "Choose a LLM",
        [
            "<Select>",
            "openai/gpt-3.5-turbo",
            "meta-llama/Llama-2-70b-chat-hf",
            "google/flan-t5-xxl",
            "databricks/dolly-v2-3b",
            "Writer/camel-5b-hf",
            "Salesforce/xgen-7b-8k-base",
            "tiiuae/falcon-40b",
            "bigscience/bloom",
        ],
        on_change=clean_chain,
    )
    if model == "<Select>":
        st.sidebar.warning("Choose a model")
        _, center_war_col, _ = st.columns([3, 5, 3])
        center_war_col.warning(
            "Please make the necessary configurations for the ChatBot from the left side panel."
        )
        return
    else:
        api_key = st.sidebar.text_input(
            f"Enter {model} API key",
            type="password",
            on_change=clean_chain,
        )
        model_host = "openai" if model.startswith("openai") else "huggingface"
        if is_api_key_valid(model_host, api_key):
            st.sidebar.success("API key was received successfully")
        else:
            _, center_war_col, _ = st.columns([3, 5, 3])
            center_war_col.warning(
                "Please make the necessary configurations for the ChatBot from the left side panel."
            )
            return

    uploaded_files = st.file_uploader(
        "You can upload any number of documents you want to chat with",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        on_change=clean_chain,
    )
    if len(uploaded_files) == 0:
        return

    if not st.session_state.chain:
        with st.spinner("Documents are being processed"):
            chunked_documents = prepare_documents(uploaded_files)
            # chunked_documents = load_multiple_documents_and_split(temp_files)
            retriever = create_vector_store_retriever(
                model_host, chunked_documents, api_key
            )

        with st.spinner("ChatBot is being prepared"):
            llm = create_llm(model, api_key)
            prompt_template = create_main_prompt()
            st.session_state.chain = create_retrieval_qa(
                llm, prompt_template, retriever
            )

    for user_message, assistant_message in st.session_state.messages.items():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_message)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(assistant_message)
            # sound_file = BytesIO()
            # tts = gTTS(assistant_message, lang="tr")
            # tts.write_to_fp(sound_file)
            # st.audio(sound_file)

    text_input = st.chat_input(
        placeholder="Chat by typing ✍️",
        key="text_box",
        max_chars=100,
    )
    # voice_input = speech2text()
    # user_input = voice_input or text_input
    user_input = text_input

    try:
        if user_input:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            with st.spinner("Question is being answered"):
                result = st.session_state.chain({"question": user_input})
                response = result["answer"]

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()

                llm_output = ""
                for i in range(len(response)):
                    llm_output += response[i]
                    message_placeholder.write(f"{llm_output}▌")
                    time.sleep(STREAMING_INTERVAL)
                message_placeholder.write(llm_output)
                with st.expander(label="Click to see the source documents"):
                    sources = "".join(
                        f"{str(idx + 1)} - {source.page_content}<hr>"
                        for idx, source in enumerate(
                            result["source_documents"]
                        )
                    )
                    st.markdown(sources, unsafe_allow_html=True)
                    # sound_file = BytesIO()
                    # tts = gTTS(llm_output, lang="tr")
                    # tts.write_to_fp(sound_file)
                    # st.audio(sound_file)

            if user_input not in st.session_state.messages:
                assistant_message = llm_output
                st.session_state.messages[user_input] = assistant_message

    except Exception as e:
        _, center_err_col, _ = st.columns([1, 8, 1])
        center_err_col.error(
            "\n Your question could not be answered. Please try to ask another question. Thank you ;]"
        )
        print(f"An error occurred: {type(e).__name__}")
        print(e)
        print("-------------------")


if __name__ == "__main__":
    main()
