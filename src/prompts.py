from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant.

Use ONLY the context below.

If the answer is not in the context, say:
"I could not find this in the HR policy documents provided."

Context:
{context}

Question:
{user_query}

Answer:
""")