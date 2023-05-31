from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# ConversationBufferMemoryからメッセージの履歴を取得
history = ConversationBufferMemory.chat_memory.messages

# OpenAIのAPIが受け付ける形式に変換
formatted_history = []
for message in history:
    if isinstance(message, HumanMessage):
        formatted_history.append({"role": "user", "content": message.content})
    elif isinstance(message, AIMessage):
        formatted_history.append({"role": "assistant", "content": message.content})
