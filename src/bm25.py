"""Okapi BM25 lexical score (issue text as query, PR text as document)."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    def __init__(self, k1=1.2, b=0.75): self.k1, self.b = k1, b

    def fit(self, docs):
        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", lowercase=True, max_features=50000).fit(docs)
        D = self.cv.transform(docs); n = D.shape[0]; df = np.asarray((D > 0).sum(0)).ravel()
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)); self.avgdl = D.sum(1).mean()
        return self

    def score(self, queries, docs):
        Q = (self.cv.transform(queries) > 0).astype(float)
        D = self.cv.transform(docs).astype(float).tocsr(); dl = np.asarray(D.sum(1)).ravel()
        D.data = D.data * (self.k1 + 1) / (D.data + self.k1 * (1 - self.b + self.b * np.repeat(dl, np.diff(D.indptr)) / self.avgdl))
        D = D.multiply(self.idf[None, :]).tocsr()
        return np.asarray(Q.multiply(D).sum(1)).ravel()
