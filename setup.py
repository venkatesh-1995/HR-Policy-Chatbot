from setuptools import find_packages, setup

setup(
    name="HR-Policy Chatbot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-community>=0.1.0",
        "langchain-groq>=0.1.0",
        "langchain-pinecone>=0.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.16.0",
        "python-dotenv>=1.0.0",
        "ipykernel>=6.0.0",
        "pypdf>=2.0.0",
        "langchain-huggingface>=0.1.0",
        "openai>=1.0.0",
        "sentence-transformers>=2.0.0",
        "typing_extensions>=4.0.0",
        "datasets>=2.0.0",
    ],
    author="venkatesh",
    author_email="ovenkatramana01@gmail.com",
    description="HR Policy Chatbot",
    python_requires=">=3.9"
)