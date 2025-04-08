import re

class FixedTokenChunker:
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size

    def tokenize(self, text: str):

        return re.findall(r'\S+', text)

    def chunk_text(self, text: str):
        tokens = self.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk = " ".join(tokens[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks

