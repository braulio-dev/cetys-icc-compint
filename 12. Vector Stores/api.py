import torch

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np

# same vector store as the notebook
class Document:
    def __init__(self, text: str, metadata: dict[str, str]):
        self.text = text
        self.metadata = metadata

class SearchResult:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document

class VectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[Document]):
        # add all documents
        self.documents.extend(documents)

        # get their content
        texts = [doc.text for doc in documents]

        # get their embeddings
        new_embeddings = self.embedding_model.encode(texts)

        # replace new embeddings with old ones or add them
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # get the query embedding
        query_embedding = self.embedding_model.encode(query)

        # compute cosine similarity scores between the query embedding and all document embeddings
        # according to docuemntation, [0] gives the per-document score vector
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # get the top k scores and their indices
        top = torch.topk(scores, k=min(top_k, len(self.documents)))
        
        # return a search result for each top score and its corresponding document
        return [SearchResult(float(score), self.documents[i]) for score, i in zip(top.values, top.indices)]
    
class FilteredVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.documents: list[Document] = []
        self.embeddings = None

    def add_documents(self, documents: list[Document]):
        self.documents.extend(documents)
        texts = [doc.text for doc in documents]
        new_embeddings = self.embedding_model.encode(texts)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self,
               query: str,
               top_k: int = 5,
               metadata_filter: dict[str, str] | None = None) -> list[SearchResult]:
        # get the query embedding
        query_embedding = self.embedding_model.encode(query)

        # compute cosine similarity scores between the query embedding and all document embeddings
        # according to docuemntation, [0] gives the per-document score vector
        scores = util.cos_sim(query_embedding, self.embeddings)[0]

        if metadata_filter is None:
            # if no filter is provided, just return the top K results as usual
            # same code as VectorStore
            top = torch.topk(scores, k=min(top_k, len(self.documents)))
            return [SearchResult(float(score), self.documents[i]) for score, i in zip(top.values, top.indices)]

        # order them by score because we're gonna iterate 1 by 1 and select until we have K
        # matches on the filter
        order = torch.argsort(scores, descending=True)

        # iterate
        results = []
        for i in order:
            # get the doc
            doc = self.documents[i]

            # compare each key-value in the filter with the document's metadata
            # if they ALL() match then we can assume the document matches and we add uit
            if all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                results.append(SearchResult(float(scores[i]), doc))
            
            # if we've got top K then we can return
            if len(results) == top_k:
                break

        return results

# Actual API
app = Flask(__name__)

# Same as the notebook, we create the model and the store we will use
#
# Since we will have a POST /documents/search endpoint that accepts a metadata filter
# we need to use the FilteredVectorStore instead of the regular one
model = SentenceTransformer("all-MiniLM-L6-v2")
store = FilteredVectorStore(model)

# Instructions state *we* will define the metadata schema
REQUIRED_METADATA = ["title", "author", "category"]

# Since we will have a GET /documents/{id} endpoint, we need to store the documents in a dict with an id as key
documents = {}
next_id = 1

# ===========================
# Utility Functions
# ===========================
def chunk_text(text):
    """
    This clamps the text in documents to 500 characters.
    1. If it's less than or equal to 500, we can pass it as is.
    2. If it's more than 500, we split it into chunks of 400 (according to the exercise instructions)
    """
    if len(text) <= 500:
        return [text]
    return [text[i:i + 400] for i in range(0, len(text), 400)]

# ============================
# API Endpoints
# ============================
@app.route("/documents", methods=["POST"])
def create_document():
    """
    Creates a new document.
    """

    global next_id # this lets us use next_id from outside the function

    # get payload
    data = request.get_json()
    text = data.get("text")
    metadata = data.get("metadata")

    # validate payload
    # 1. ensure the text is not None
    # 2. ensure the metadata is a dict
    # 3. ensure the metadata has exactly the keys in REQUIRED_METADATA (we need sorted because otherwise it compares the order of entries)
    if (text is None or not isinstance(metadata, dict)
            or sorted(metadata.keys()) != sorted(REQUIRED_METADATA)):
        return jsonify({"error": f"metadata must have exactly these keys: {REQUIRED_METADATA}"}), 400


    # now we can create the document
    document_id = next_id
    next_id += 1
    documents[document_id] = {"id": document_id, "text": text, "metadata": metadata}

    # if it needs chunking let's do it. According to the instructions:
    # 1. we append the original ID to each fragment
    # 2. each fragment is a new document in the store
    chunks = []
    for fragment in chunk_text(text):
        chunk_metadata = dict(metadata)
        chunk_metadata["original_id"] = document_id
        chunks.append(Document(fragment, chunk_metadata))
    store.add_documents(chunks)

    # returen results
    return jsonify({"id": document_id}), 201


@app.route("/documents/<int:document_id>", methods=["GET"])
def get_document(document_id):
    """
    Retrieves a document by its ID.
    """

    # i didnt know if this was meant to be done in the vector store itself but
    # i made a document dict that maps document_id to the corresponding document
    # for an O(1) lookup.
    document = documents.get(document_id)

    # if it doesnt exist, return 404, otherwise return the document
    if document is None:
        return jsonify({"error": "not found"}), 404
    
    return jsonify(document)


@app.route("/documents/search", methods=["POST"])
def search_documents():
    """
    Searches for documents based on a query and optional metadata filter.
    """

    # For search, it's pretty simple. we just use our FilteredVectorStore's search method and pass the query, top_k and metadata_filter from the payload.
    
    # get payload
    data = request.get_json()
    query = data.get("query")
    top_k = data.get("top_k", 5)
    metadata_filter = data.get("metadata_filter")

    # validate payload
    # 1. query is required
    # 2. top_k is optional (defaults to 5)
    # 3. metadata_filter is optional, but if provided it must be a dict
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        return jsonify({"error": "metadata_filter must be a dict"}), 400
    if query is None:
        return jsonify({"error": "query is required"}), 400
    
    # ran into this problem:
    # if the store is empty, there's an error in search because of empty embeddings
    # 1. just make sure it exists 
    if not store.documents:
        return jsonify([])

    # all input is validated, we can call store.search
    results = store.search(query, top_k=top_k, metadata_filter=metadata_filter)

    # for all the results, we just build the results. according to instructions:
    # 1. porcentaje de similutd
    # 2. el texto
    # 3. los metadatos
    response = []
    for r in results:
        response.append({
            "similarity": round(r.score * 100, 2),
            "text": r.document.text,
            "metadata": r.document.metadata,
        })
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
