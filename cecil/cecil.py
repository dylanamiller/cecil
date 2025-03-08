import os
import subprocess
import tempfile
from typing import Any

from omegaconf import OmegaConf
from PyPDF2 import PdfReader

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython import get_ipython

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

from cecil.state.agent_state import AgentState
from cecil.utils.system_prompt import SYSTEM_PROMPT



@magics_class
class Cecil(Magics):
    def __init__(self):
        super().__init__()
        self._graph = None
        self._models = None
        self._state = None

    @line_magic
    def initialize(self, line: dict[str, Any]) -> None:
        if line.get("model", None):
            models = OmegaConf.create({
                model_fcn: OllamaLLM(model=model_name) \
                    for model_name, model_fcn in line.models.items()
            })
            models.rag = OllamaLLM(model=line.rag_model)
        else:
            models = OmegaConf.create({})
            llm = OllamaLLM(model="deepseek-r1:70b")
            models.model = llm
            models.rag = llm
        self._models = models
        state = AgentState(
            input="",
            system_prompt=SYSTEM_PROMPT + line.get("system_prompt", ""),
            messages=[],
            model_called=[],
            agent_outcome=None,
            intermediate_steps=[""" add tools here """],
        )
        graph_builder = StateGraph(state)
        graph_builder.set_entry_point(START)
        graph_builder.add_node("chat", self._chat)
        graph_builder.set_finish_point(END)
        tools = []
        self._graph = graph_builder.compile()
        subprocess.run([
            "docker",
            "compose",
            "-f",
            "~/projects/Perplexica",
            "up",
            "-d",
        ], check=False)
        # rag
        self._chunk_size = line.get("chunk_size", None)
        self._chunk_overlap = line.get("chunk_overlap", None)
        self._chroma_path = line.get("chroma_path", None)

    def _chat(self, state: AgentState) -> str:
        messages = state['messages'] + state["system_prompt"]
        model = state['model_called']
        response = self._models[model].invoke(messages)
        return {"messages": [response]}

    def _query(self, query: str, model: str | None = None) -> str:
        if model is None:
            model = "model"
        return self._models[model].invoke(query)

    def _query_rag(self, query: str) -> str:
        db = Chroma(
            persist_directory=self._chroma_path, embedding_function=self._get_embedding_fcn()
        )
        results = db.similarity_search_with_score(query, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        prompt = prompt_template.format(context=context_text, question=query)
        return self._models.rag.invoke(prompt)

    def _add_to_chroma(self, chunks: list[Document]) -> None:
        db = Chroma(
            persist_directory=self._chroma_path, embedding_function=self._get_embedding_fcn()
        )
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        new_chunks = []
        new_chunk_ids = []
        last_page_id = None
        current_page_idx = 0
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}#"
            if last_page_id == current_page_id:
                current_page_idx += 1
            else:
                current_page_idx = 0
            last_page_id = current_page_id
            uid = current_page_id + str(current_page_idx)
            chunk.metadata["id"] = uid
            if uid not in existing_ids:
                new_chunk_ids.append(uid)
                new_chunks.append(chunk)
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()

    def _split_docs(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def _get_embedding_fcn(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model="nomic-embed-test")

    @line_magic
    def chat(self, line: dict[str, str]) -> str:
        if not line.get("model_called", None):
            line["model_called"] = "model"
        return self._graph.invoke(**line)

    @line_magic
    def query(self, line: dict[str]) -> str:
        return self._query(line[0], line[1])

    @line_magic
    def load_pdf(self, line: os.PathLike) -> None:
        document_loader = PyPDFDirectoryLoader(line)
        document = document_loader.load()

    @line_magic
    def pdfquery(self, line: str) -> str:
        return self._model.rag.query(line)

    @line_magic
    def search(self, line):
        return f"Hello, {line}!"

    @cell_magic
    def optimize(self, line, cell):
        return cell * int(line)

    @cell_magic
    def context(self, line, cell):
        return cell * int(line)

    @cell_magic
    def explain(self, line, cell):
        return cell * int(line)


# Registering the magics
ipython = get_ipython()
ipython.register_magics(Cecil)