import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import gradio as gr

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

def text_to_speech(text):
    # Preprocess the text
    inputs = processor(text=text, return_tensors="pt")
    
    with torch.no_grad():
        spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        waveform = vocoder(spectrogram)
    
    # Save the waveform to a temporary file
    output_file = "output.wav"
    sf.write(output_file, waveform.squeeze().numpy(), samplerate=22050)
    
    return output_file


iface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.Audio(type="filepath"),
    title="Text-to-Speech with SpeechT5",
    description="Convert text to speech using Microsoft's SpeechT5 model.",
    allow_flagging="never",  
    cache_examples=True,     
)

# Launch the interface
iface.launch()
