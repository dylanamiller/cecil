# description

`cecil` is an ai assistant for jupyter notebooks. intended to be used as magin functions, `cecil` can be queried about entire cells
or specific queries can be made. `cecil` can work directly on  the code, or use `perplexica` to search the web for more information
if needed. `cecil` is also capable of acting as a rag application, parsing and storing pdfs to be used in later questions. this is useful
if you are doing research on a specific paper or set of papers.

## examples

%%optimize <could ask specific question>
to offer any optimizations on code cell

%%context <could add specific question>
to look up supporting context for code cell, may use web, may use rag

%query <question you want to ask>

%pdf <path to pdf to store for rag purposes>

%pdfquery <use rag in answer of question>

%search <ask to search web>


```
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython import get_ipython

@magics_class
class MyMagics(Magics):
    @line_magic
    def hello(self, line):
        return f"Hello, {line}!"

    @cell_magic
    def repeat(self, line, cell):
        return cell * int(line)

# Registering the magics
ipython = get_ipython()
ipython.register_magics(MyMagics)
```