import gradio as gr
from pydantic import BaseModel
from typing import Union
from src.translation import Translator

dir = True
baseTrans = None

def changeDir():
    global dir
    dir = not dir
    
    topTextView.label = "На русском" if dir else "Тывалап"
    output.label = "На тувинском" if dir else "Орус дылда"
    submitButton.label = "Перевести/Очулдурар"
    return [
        gr.Button("Поменять на Tyv/Rus" if dir else "Rus/Tyv өскертир"),
        gr.Textbox(label= "На русском" if dir else "Тывалап", lines=4, placeholder="Введите текст" if dir else "Сөзүглелден киириңер"),
        gr.Textbox(label= "На тувинском" if dir else "Орус дылга", lines=4),
        gr.Button("Перевести" if dir else "Очулдурар", variant="primary")
    ]

class ModelInterface(BaseModel):
    rus: str
    dir: str

class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "rus_Cyrl"
    tgt_lang: str = "tyv_Cyrl"
    by_sentence: bool = True
    preprocess: bool = True

def translate(request: TranslationRequest):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    print(request)
    global demo
    translator = demo.baseTrans

    output = translator.translate(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
        preprocess=request.preprocess,
    )
    return output
    
def goTranslate(text):
    global dir
    model = TranslationRequest(
        text = text,
        src_lang = "rus_Cyrl" if dir else "tyv_Cyrl",
        tgt_lang = "tyv_Cyrl" if dir else "rus_Cyrl"
    )
    payload = translate(request=model)
    return payload

with gr.Blocks(theme=gr.themes.Default(primary_hue="green", secondary_hue="yellow").set(background_fill_primary_dark="#FFFFFF", block_background_fill_dark="white", body_text_color="blue", body_background_fill_dark="#FFFFFF", button_primary_text_color_dark="blue", button_primary_background_fill_dark="lightred", button_primary_text_color_hover_dark="white", button_secondary_text_color_dark="blue", button_secondary_background_fill_hover_dark="lightgreen", body_text_color_dark="black", input_background_fill_dark="white", input_placeholder_color_dark="gray", button_secondary_background_fill_dark="white", block_title_text_color_dark="blue", block_title_text_color="blue", background_fill_primary="#FFFFFF", loader_color_dark="#DDFFDD")) as demo:
    
    
    topTextView = gr.Textbox(label= "На русском" if dir else "Тывалап", lines=4, placeholder="Введите текст")
    output = gr.Textbox(label="На тувинском", lines=4)

    with gr.Row():
        direction_btn = gr.Button("Поменять на Tyv/Rus")
        submitButton = gr.Button("Перевести/Очулдурар", variant='primary')
    direction_btn.click(fn=changeDir, outputs=[direction_btn, topTextView, output, submitButton], api_name="changeDir")
    submitButton.click(fn=goTranslate, inputs=topTextView, outputs=output, api_name="goTranslate")

    gr.Markdown("""
    This translator is powered by the neural network model https://huggingface.co/slone/nllb-rus-tyv-v2-extvoc based on NLLB-200.
    """)


if __name__ == "__main__":
    demo.launch()
