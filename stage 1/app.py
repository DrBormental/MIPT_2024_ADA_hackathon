import shutil
import os
import gradio as gr

import torch
from uuid import uuid4
from huggingface_hub.file_download import http_get
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from llama_cpp import Llama


SYSTEM_PROMPT = "Ты — помощник ТревелЛайн, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_model(
    directory: str = ".",
    model_name: str = "model-q4_K.gguf",
    model_url: str = "https://huggingface.co/IlyaGusev/saiga2_13b_gguf/resolve/main/model-q4_K.gguf"
):
    final_model_path = os.path.join(directory, model_name)
    
    print("Downloading all files...")
    if not os.path.exists(final_model_path):
        with open(final_model_path, "wb") as f:
            http_get(model_url, f)
    os.chmod(final_model_path, 0o777)
    print("Files downloaded!")
    # Используйте устройство CUDA, если доступно
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Llama(
        model_path=final_model_path,
        n_ctx=2000,
        n_parts=1,
        device=device  # Передача устройства в модель
    )
    
    print("Model loaded!")
    return model


EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
MODEL = load_model()


def get_uuid():
    return str(uuid4())


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]


def get_message_tokens(model, role, content):
    content = f"{role}\n{content}\n</s>"
    content = content.encode("utf-8")
    return model.tokenize(content, special=True)


def get_system_tokens(model):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    return get_message_tokens(model, **system_message)


def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text


def upload_files(files, file_paths):
    file_paths = [f.name for f in files]
    return file_paths

    
def build_index(file_paths, db, chunk_size, chunk_overlap, file_warning):
    documents = [load_single_document(path) for path in file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    print("Documents after split:", len(documents))
    fixed_documents = []
    for doc in documents:
        doc.page_content = process_text(doc.page_content)
        if not doc.page_content:
            continue
        fixed_documents.append(doc)
    print("Documents after processing:", len(fixed_documents))

    texts = [doc.page_content for doc in fixed_documents]
    embeddings = EMBEDDER.encode(texts, convert_to_tensor=True)
    db = {"docs": texts, "embeddings": embeddings}
    print("Embeddings calculated!")
    
    file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
    return db, file_warning


def retrieve(history, db, retrieved_docs, k_documents):
    retrieved_docs = ""
    if db:
        last_user_message = history[-1][0]
        query_embedding = EMBEDDER.encode(last_user_message, convert_to_tensor=True)
        scores = cos_sim(query_embedding, db["embeddings"])[0]
        top_k_idx = torch.topk(scores, k=k_documents)[1]
        top_k_documents = [db["docs"][idx] for idx in top_k_idx]
        retrieved_docs = "\n\n".join(top_k_documents)
    return retrieved_docs

    
def user(message, history, system_prompt):
    new_history = history + [[message, None]]
    return "", new_history


def bot(
    history,
    system_prompt,
    conversation_id,
    retrieved_docs,
    top_p,
    top_k,
    temp
):
    model = MODEL
    if not history:
        return

    tokens = get_system_tokens(model)[:]

    for user_message, bot_message in history[:-1]:
        message_tokens = get_message_tokens(model=model, role="user", content=user_message)
        tokens.extend(message_tokens)
        if bot_message:
            message_tokens = get_message_tokens(model=model, role="bot", content=bot_message)
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    if retrieved_docs:
        last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {last_user_message}"
    message_tokens = get_message_tokens(model=model, role="user", content=last_user_message)
    tokens.extend(message_tokens)

    role_tokens = model.tokenize("bot\n".encode("utf-8"), special=True)
    tokens.extend(role_tokens)
    generator = model.generate(
        tokens,
        top_k=top_k,
        top_p=top_p,
        temp=temp
    )

    partial_text = ""
    for i, token in enumerate(generator):
        if token == model.token_eos():
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
        history[-1][1] = partial_text
        yield history


with gr.Blocks(
    theme=gr.themes.Soft()
) as demo:
    db = gr.State(None)
    conversation_id = gr.State(get_uuid)
    favicon = '<img src="https://cdn.midjourney.com/b88e5beb-6324-4820-8504-a1a37a9ba36d/0_1.png" width="48px" style="display: inline">'
    gr.Markdown(
        f"""<h1><center>TravelLine - центр автоматизированной поддержки</center></h1>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            file_output = gr.File(file_count="multiple", label="Загрузка файлов")
            file_paths = gr.State([])
            file_warning = gr.Markdown(f"Фрагменты ещё не загружены!")

        with gr.Column(min_width=200, scale=3):
            with gr.Tab(label="Параметры нарезки"):
                chunk_size = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=250,
                    step=50,
                    interactive=True,
                    label="Размер фрагментов",
                )
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=30,
                    step=10,
                    interactive=True,
                    label="Пересечение"
                )


    with gr.Row():
        k_documents = gr.Slider(
            minimum=1,
            maximum=10,
            value=2,
            step=1,
            interactive=True,
            label="Кол-во фрагментов для контекста"
        )
    with gr.Row():
        retrieved_docs = gr.Textbox(
            lines=6,
            label="Извлеченные фрагменты",
            placeholder="Появятся после задавания вопросов",
            interactive=False
        )
    with gr.Row():
        with gr.Column(scale=5):
            system_prompt = gr.Textbox(label="Системный промпт", placeholder="", value=SYSTEM_PROMPT, interactive=False)
            chatbot = gr.Chatbot(label="Диалог").style(height=400)
        with gr.Column(min_width=80, scale=1):
            with gr.Tab(label="Параметры генерации"):
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    interactive=True,
                    label="Top-p",
                )
                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=30,
                    step=5,
                    interactive=True,
                    label="Top-k",
                )
                temp = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.1,
                    interactive=True,
                    label="Temp"
                )

    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Отправить сообщение",
                placeholder="Отправить сообщение",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Отправить")
                stop = gr.Button("Остановить")
                clear = gr.Button("Очистить")

    # Upload files
    upload_event = file_output.change(
        fn=upload_files,
        inputs=[file_output, file_paths],
        outputs=[file_paths],
        queue=True,
    ).success(
        fn=build_index,
        inputs=[file_paths, db, chunk_size, chunk_overlap, file_warning],
        outputs=[db, file_warning],
        queue=True
    )

    # Pressing Enter
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, db, retrieved_docs, k_documents],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            conversation_id,
            retrieved_docs,
            top_p,
            top_k,
            temp
        ],
        outputs=chatbot,
        queue=True,
    )

    # Pressing the button
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot, system_prompt],
        outputs=[msg, chatbot],
        queue=False,
    ).success(
        fn=retrieve,
        inputs=[chatbot, db, retrieved_docs, k_documents],
        outputs=[retrieved_docs],
        queue=True,
    ).success(
        fn=bot,
        inputs=[
            chatbot,
            system_prompt,
            conversation_id,
            retrieved_docs,
            top_p,
            top_k,
            temp
        ],
        outputs=chatbot,
        queue=True,
    )

    # Stop generation
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )

    # Clear history
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=1)
demo.launch(show_error=True,server_name='0.0.0.0')
