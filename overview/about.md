Multimodal RAG: lets understand it with the simple example:

If I use Langchain's PypdfLoader and chunk it and create embeddings of it and store it in a vector store I would not be able to pass Image,charts,tables level context to the LLM(LIKE GPT-4 OR CLAUDE SONNET OPUS whatever) even though those models are good with visual understanding along with text.

For a company using RAG they want to cover context from charts , images, tables,text and so on.. 

This is where Multimodal RAG comes into play. It is a technique that allows you to create a RAG system that can understand and answer questions about images, charts, tables, and text.

![alt text](image.png)