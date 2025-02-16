import os
import gradio as gr
from transformers import AutoTokenizer
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingfacehub_api_token:
    raise ValueError("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")


model_name = "HuggingFaceH4/starchat-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)


llm = HuggingFaceHub(
    repo_id=model_name,
    huggingfacehub_api_token=huggingfacehub_api_token,
    model_kwargs={
        "min_length": 10,        
        "max_new_tokens": 50,     
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 50,
        "top_p": 0.95,
        "eos_token_id": 49155,
    },
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="{chat_history}\nYou: {input}\nAI:"
)

llm_chain = LLMChain(
    prompt=prompt_template,
    llm=llm,
    memory=memory,
)

def chat_interface(user_input, history):
    if not user_input.strip():
        return history, history

    raw_reply = llm_chain.predict(input=user_input)
    if "AI:" in raw_reply:
        reply = raw_reply.split("AI:")[-1].strip()
    else:
        reply = raw_reply.strip()

    history.append((user_input, reply))
    return history, history


css = """
/* Align odd chat messages (user messages) to the left */
.gr-chatbot .chat-message:nth-child(2n-1) {
    justify-content: flex-start;
    text-align: left;
}
/* Align even chat messages (AI messages) to the right */
.gr-chatbot .chat-message:nth-child(2n) {
    justify-content: flex-end;
    text-align: right;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# AI Assistant")
    chatbot = gr.Chatbot(label="Chat History")
    user_input_box = gr.Textbox(lines=2, placeholder="Enter your message here...", label="Your Message:")
    send_button = gr.Button("Send")
    state = gr.State([])

    send_button.click(
        fn=chat_interface,
        inputs=[user_input_box, state],
        outputs=[chatbot, state],
    )

demo.launch()
