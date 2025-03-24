import uvicorn
from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import gradio as gr
from .translation import Translator

from .app.app import demo

class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "rus_Cyrl"
    tgt_lang: str = "tyv_Cyrl"
    by_sentence: bool = True
    preprocess: bool = True


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
translator = Translator()

@app.get("/interface")
async def interface():
    with open("templates/interface.html", "r") as file:
            html_content = file.read()
    return HTMLResponse(content=html_content)


@app.post("/translate")
def translate(request: TranslationRequest):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    print(request)
    output = translator.translate(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
        preprocess=request.preprocess,
    )
    return {"translation": output}


@app.get("/translator")
def translate(tyv: Union[str, None] = None, rus: Union[str, None] = None):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    output = translator.translate(rus, src_lang='rus_Cyrl', tgt_lang='tyv_Cyrl')
    return {"translation": output}


@app.get("/list-languages")
def list_languages():
    """Show the mapping of supported languages: from their English names to their 8-letter codes."""
    return translator.languages

CUSTOM_PATH = "/"
app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
demo.baseTrans = translator
