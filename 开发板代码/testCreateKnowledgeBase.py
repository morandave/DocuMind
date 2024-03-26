# coding: utf-8
import os
import glob
from typing import List
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from langchain_community.document_loaders import (
    CSVLoader, EverNoteLoader, PDFMinerLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredEPubLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredODTLoader, UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from config import CONFIG

# 设置目录和embedding基础变量
embeddings_model_name = CONFIG['embedding_model']
chunk_size = CONFIG['chunk_size']
chunk_overlap = CONFIG['chunk_overlap']
k = CONFIG['k']

# Custom document loaders 自定义文档加载
class MyElmLoader(UnstructuredEmailLoader):
    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}), ".doc": (UnstructuredWordDocumentLoader, {}), ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}), ".eml": (MyElmLoader, {}), ".epub": (UnstructuredEPubLoader, {}), ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}), ".odt": (UnstructuredODTLoader, {}), ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}), ".pptx": (UnstructuredPowerPointLoader, {}), ".txt": (TextLoader, {"encoding": "utf8"}),
    ".xls": (UnstructuredExcelLoader, {}), ".xlsx": (UnstructuredExcelLoader, {}),
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split into chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def main(input_dir: str):
    global source_directory, output_dir
    source_directory = os.path.join(CONFIG['doc_source'], input_dir)
    # 创建输出文件夹的路径
    output_dir = os.path.join(CONFIG['db_source'], input_dir)
    
    # Create embeddings
    print("Creating new vectorstore")
    texts = process_documents()
    print(f"Creating embeddings. May take some minutes...")
    embedding_function = SentenceTransformerEmbeddings(model_name=embeddings_model_name)
    db = Chroma.from_documents(texts, embedding_function, persist_directory=output_dir)
    db.persist()
    db = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create knowledge base.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory.')
    args = parser.parse_args()
    main(args.input_dir)