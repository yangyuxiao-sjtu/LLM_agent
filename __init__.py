from sentence_transformers import SentenceTransformer

token_used = 0
sentence_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
if __name__ == "__main__":
    import sys

    sys.path.append("/mnt/sda/yuxiao_code/hlsm")
