from dotenv import load_dotenv
import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Define Janvi's personality
system_prompt = """
You are Janvi.
Answer questions logically.
You will speak from your point of view. You will share personal things from your life
even when the user doesn’t ask for it. For example, if the user asks about the theory of
relativity, you will share your personal experiences with it.
Answer in 2-6 sentences.
You will have a sense of humor.
"""

# Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# Create LangChain runnable
chain = prompt | llm

# Store full conversation state
chat_history_messages = []

# Function to handle chat
def chat_with_janvi(user_message, chat_display):
    global chat_history_messages
    chat_history_messages.append(HumanMessage(content=user_message))
    response = chain.invoke({"input": user_message, "history": chat_history_messages})
    chat_history_messages.append(AIMessage(content=response.content))
    
    # Append OpenAI-style messages
    chat_display.append({"role": "user", "content": user_message})
    chat_display.append({"role": "assistant", "content": response.content})
    return chat_display, ""

# GUI
with gr.Blocks(title="Chat with Janvi", theme=gr.themes.Soft()) as page:
    gr.Markdown("## 💬 Chat with Janvi\nWelcome to your personal conversation with Janvi Gaba!")

    chatbot = gr.Chatbot(type='messages', show_label=False, render=False)
    textbox = gr.Textbox(placeholder="Type your message and press Enter", show_label=False, render=False)
    clear = gr.Button("🧹 Clear Chat")

    with gr.Row():
        chatbot.render()
    with gr.Row():
        textbox.render()
    
    textbox.submit(chat_with_janvi, inputs=[textbox, chatbot], outputs=[chatbot, textbox])
    clear.click(lambda: ([], chat_history_messages.clear()), outputs=chatbot)

# Launch app
if __name__ == "__main__":
    page.launch()

