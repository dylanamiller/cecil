import os
import subprocess
import tempfile
from typing import Any

from omegaconf import OmegaConf
from PyPDF2 import PdfReader

from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython import get_ipython

from embedchain import BotAgent
from embedchain.vector_stores import Weaviate
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END

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
        vector_store = Weaviate()
        temp_dir = tempfile.mkdtemp()
        if line.get("model", None):
            models = OmegaConf.create({
                model_fcn: OllamaLLM(model=model_name) \
                    for model_name, model_fcn in line.models.items()
            })
            rag_llm = OllamaLLM(model=line.rag_model)
            models.rag = BotAgent(llm=rag_llm, vector_store=vector_store, temp_dir=temp_dir)
        else:
            models = OmegaConf.create({})
            llm = OllamaLLM(model="deepseek-r1:70b")
            models.model = llm
            models.rag = BotAgent(llm=llm, vector_store=vector_store, temp_dir=temp_dir)
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

    def _chat(self, state: AgentState) -> str:
        messages = state['messages'] + state["system_prompt"]
        model = state['model_called']
        response = self._models[model].invoke(messages)
        return {"messages": [response]}

    def _query(self, query: str, model: str | None = None) -> str:
        if model is None:
            model = "model"
        return self._models[model].invoke(query)

    @line_magic
    def chat(self, line: dict[str, str]) -> str:
        if not line.get("model_called", None):
            line["model_called"] = "model"
        return self._graph.invoke(**line)

    @line_magic
    def query(self, line: dict[str]) -> str:
        return self._query(line[0], line[1])

    @line_magic
    def store_pdf(self, line: os.PathLike) -> None:
        reader = PdfReader(line)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name
        # Add the PDF file to the knowledge base
        self._model.rag.add_source(temp_file_path)

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