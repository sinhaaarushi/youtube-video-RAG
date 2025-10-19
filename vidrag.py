#!/usr/bin/env python3
"""
VidRAG - RAG-powered YouTube Q&A System

A Python application that uses Retrieval-Augmented Generation (RAG) to answer questions
about YouTube videos by extracting transcripts, creating embeddings, and using AI to
provide contextual answers.

Author: Mehak Verma
Tech Stack: Python, LangChain, OpenAI, FAISS, youtube-transcript-api
"""

import os
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


class VidRAG:
    """
    A RAG system for YouTube video Q&A using LangChain and OpenAI.
    """
    
    def __init__(self, openai_api_key: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the VidRAG system.
        
        Args:
            openai_api_key: OpenAI API key for embeddings and chat
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Initialize prompt template
        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant that answers questions about YouTube videos.
            Answer ONLY from the provided transcript context.
            If the context is insufficient to answer the question, just say you don't know.
            Be concise and accurate in your responses.

            Context: {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=['context', 'question']
        )
        
        self.vector_store = None
        self.retriever = None
        self.main_chain = None
    
    def extract_transcript(self, video_id: str, languages: List[str] = ["en"]) -> str:
        """
        Extract transcript from a YouTube video.
        
        Args:
            video_id: YouTube video ID (not full URL)
            languages: List of preferred languages
            
        Returns:
            Extracted transcript as string
            
        Raises:
            TranscriptsDisabled: If no captions available
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            print(f"✅ Successfully extracted transcript ({len(transcript)} characters)")
            return transcript
        except TranscriptsDisabled:
            print("❌ No captions available for this video.")
            raise
        except Exception as e:
            print(f"❌ Error extracting transcript: {e}")
            raise
    
    def create_vector_store(self, transcript: str) -> None:
        """
        Create vector store from transcript.
        
        Args:
            transcript: Video transcript text
        """
        print("🔄 Creating text chunks...")
        chunks = self.text_splitter.create_documents([transcript])
        print(f"✅ Created {len(chunks)} chunks")
        
        print("🔄 Generating embeddings and creating vector store...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        # Create main chain
        self._create_chain()
        print("✅ Vector store created successfully")
    
    def _create_chain(self) -> None:
        """Create the main RAG chain."""
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        
        parallel_chain = RunnableParallel({
            'context': self.retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        self.main_chain = parallel_chain | self.prompt | self.llm | parser
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question about the video.
        
        Args:
            question: Question to ask
            
        Returns:
            AI-generated answer
        """
        if not self.main_chain:
            raise ValueError("Vector store not initialized. Please process a video first.")
        
        print(f"🤔 Question: {question}")
        print("🔄 Processing...")
        
        answer = self.main_chain.invoke(question)
        print(f"💡 Answer: {answer}")
        return answer
    
    def process_video(self, video_id: str) -> None:
        """
        Process a YouTube video for Q&A.
        
        Args:
            video_id: YouTube video ID
        """
        print(f"🎥 Processing video: {video_id}")
        
        # Extract transcript
        transcript = self.extract_transcript(video_id)
        
        # Create vector store
        self.create_vector_store(transcript)
        
        print("✅ Video processing complete! You can now ask questions.")


def main():
    """Main function to run VidRAG interactively."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("You can create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Initialize VidRAG
    vidrag = VidRAG(api_key)
    
    print("🎬 Welcome to VidRAG - YouTube Q&A System!")
    print("=" * 50)
    
    # Get video ID from user
    video_id = input("Enter YouTube video ID: ").strip()
    if not video_id:
        print("❌ Please provide a valid video ID")
        return
    
    try:
        # Process video
        vidrag.process_video(video_id)
        
        # Interactive Q&A
        print("\n💬 Ask questions about the video (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                vidrag.ask_question(question)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error processing video: {e}")


if __name__ == "__main__":
    main()
