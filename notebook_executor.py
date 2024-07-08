import nbformat
from nbconvert import PythonExporter
from nbclient import NotebookClient
#import requests
from nbformat import read, write
from nbconvert.preprocessors import ExecutePreprocessor
import io
import json
from fastapi import FastAPI, HTTPException

def run_notebook(notebook_content: dict) -> str:
    """Izvrši sadržaj beležnice i vrati rezultat kao JSON string.
    """
    # Pretvorite JSON sadržaj beležnice u string
    notebook_str = json.dumps(notebook_content)
    # Pročitajte beležnicu iz JSON stringa
    nb = read(io.StringIO(notebook_str), as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    # Izvršite beležnicu
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Greška pri izvršavanju beležnice: {e}")
    
    # Pretvorite beležnicu u string
    output = io.StringIO()
    write(nb, output)
    return output.getvalue()