#The main RAG pipeline.

# For Crawling
import asyncio
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy

# For Chunking and Embedding
from embedding_function import OllamaEmbeddingFunction
import re

#For Deleting
import chromadb

#For Generation
from langchain_core.output_parsers import StrOutputParser

# For Retrieval Grader
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

#TODO: Embed these in the docker file
EMBEDDING_MODEL = 'bge-base-en-v1.5'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


client = chromadb.PersistentClient(path="./my_chroma_data")
CURRENT_BOOKS = [cli.name for cli in client.list_collections()]

# Gather Data
async def savePDF(pdf_url):
    # Initialize the PDF crawler strategy
    pdf_crawler_strategy = PDFCrawlerStrategy()

    # PDFCrawlerStrategy is typically used in conjunction with PDFContentScrapingStrategy
    # The scraping strategy handles the actual PDF content extraction
    pdf_scraping_strategy = PDFContentScrapingStrategy()
    run_config = CrawlerRunConfig(scraping_strategy=pdf_scraping_strategy)

    async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as crawler:

        print(f"Attempting to process PDF: {pdf_url}")
        result = await crawler.arun(url=pdf_url, config=run_config)

        if result.success:
            
            title = result.metadata.get('title')
            
            if not title:
                title = 'Unknown_Title'
                
            title = re.sub(r'[^a-zA-Z0-9._-]', '', title)
            CURRENT_BOOKS.append(title)
            
            print(f"Successfully processed PDF: {result.url}")
            print(f"Metadata Title: {title}")
            
            if result.markdown and hasattr(result.markdown, 'raw_markdown'):
                #Get Chunks and Metadata
                texts, metadatas = await chunk_markdown(result.markdown, title)
                
                #Create Chroma Instance
                vector_store = Chroma(
                    embedding_function= OllamaEmbeddingFunction(),
                    collection_name=title,
                    persist_directory = "./my_chroma_data"
                )
                
                #Save to Chroma
                vector_store.add_texts(texts=texts, metadatas=metadatas)

                print(f'Added {len(texts)} chunks to the database')
            else:
                print("No markdown (text) content extracted.")
                
        else:
            print(f"Failed to process PDF: {result.error_message}")

# Process Data into chunks
async def chunk_markdown(markdown, title, max_chunk_size=1000):
    
    texts = []
    metadatas = []
    Chroma.from_texts
    
    sections = re.split(r'(?=^#{1,3} )', markdown.raw_markdown, flags=re.MULTILINE)
    
    for section in sections:
        heading_match = re.match(r'^(#{1,3}) (.*)', section.strip())
        heading = heading_match.group(2).strip() if heading_match else "No Heading"
        
        sentences = [s.strip() for s in section.split(".") if s.strip()]
        
        buffer = ""
        for sentence in sentences:
            # If current paragraph fits in the buffer, append it
            if len(buffer) + len(sentence) + 2 <= max_chunk_size:
                buffer += sentence
            else:
                texts.append((buffer + sentence).strip())

                buffer = sentence
                
        # Commit remaining buffer
        if buffer.strip():
            texts.append(buffer.strip())
            metadatas.append({"heading": heading, "title": title})
        
        '''path = "view.md"
        
        with open(path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts, 1):
                f.write(f"### Chunk {i}\n\n")
                f.write(text.strip() + "\n\n")
                f.write("---\n\n")
        print(f"✅ Saved {len(texts)} chunks to '{path}'")'''
        
    return texts, metadatas

# Delete book from VECTOR_DB
async def delete_book(title):
    
    global CURRENT_BOOKS
    
    if title not in CURRENT_BOOKS:
        print(f'Book titled "{title}" not found in the database.')
        return
        
    client.delete_collection(name=title)
    
    CURRENT_BOOKS.remove(title)
    print(f'Book titled "{title}" has been deleted from the database.')

# Retrieval Grader
async def grade_retrieval(query, title, retriever):
    
    ''' vectorstore = Chroma(
        client=client, # Use the client if persistent/remote
        collection_name=title,
        embedding_function=OllamaEmbeddingFunction()
        )
    
    retriever = vectorstore.as_retriever(similarity_top_k=1)'''
    
    llm = ChatOllama(model=LANGUAGE_MODEL, format='json', temperature=0)

    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains anything related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    
    retrieval_grader = prompt | llm | JsonOutputParser()

    docs = retriever.invoke(query)
    doc_txt = docs[1].page_content if len(docs) > 1 else "No document retrieved"
        
    return retrieval_grader.invoke({"question": query, "document": doc_txt}), doc_txt

async def generate_answer(query, title):

    vectorstore = Chroma(
        client=client, # Use the client if persistent/remote
        collection_name=title,
        embedding_function=OllamaEmbeddingFunction()
        )
    
    retriever = vectorstore.as_retriever(similarity_top_k=1)

    llm = ChatOllama(model=LANGUAGE_MODEL, temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a knowledgeable assistant.
        Use the provided context to answer the question. If you don't know the answer, just say you don't know. 
        Do not try to make up an answer.
        Provide your answer in markdown format. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context}
        Question: {question}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "context"],
    )
    
    grade = await grade_retrieval(query, title, retriever)
    
    relavent = grade[0]['score'] == 'yes'
    
    if not relavent:
        print("The retrieved document was not relevant to the question. Cannot provide an answer.")
        return False
    
    doc_txt = grade[1]
    
    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke({"question": query, "context": doc_txt})
    print(generation)
    return generation