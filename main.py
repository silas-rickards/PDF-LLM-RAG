from rag import savePDF, generate_answer, delete_book
import asyncio
import chromadb


client = chromadb.PersistentClient(path="./my_chroma_data")
CURRENT_BOOKS = [cli.name for cli in client.list_collections()]


async def main():
    
    pdf_link = "Insert PDF Link Here"
    
    await savePDF(pdf_link)
    
    #query = input("What is your question? ")
    
    #await generate_answer(query, CURRENT_BOOKS[0])
    
    #await delete_book(CURRENT_BOOKS[0])

if __name__ == "__main__":
    asyncio.run(main())