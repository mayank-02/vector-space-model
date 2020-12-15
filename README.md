# Vector Space Model

The program `vsm.py` implements a toy search engine to illustrate the vector space model using TF-IDF for documents.

The program asks you to enter a search query, and then returns all documents from the corpus matching the query, in decreasing order of cosine similarity, according to the vector space model.

The document corpus consists of just four documents, which are product descriptions of popular books, taken from amazon.com.

## Getting Started

- Install Python 3.6+
- Install all pip requirements from the `requirements.txt`:

```bash
$ python3 -m pip install -r requirements.txt
```

- To download stopwords used for the model, open your terminal or command prompt and enter following commands:

```bash
$ python3
>>> import nltk
>>> nltk.download('stopwords')
```

## Usage

```bash
$ python3 vsm.py
Search query >> lord of the ring
------------------------------------------
| Score | Document                       |
------------------------------------------
| 0.731 | corpus/lotr.txt                |
| 0.118 | corpus/the_hobbit.txt          |
------------------------------------------
```

### Queries

It supports free-form queries

### Corpus

You can point `CORPUS` in `vsm.py` to your own corpus to use vector space model on it.

## Authors

[Mayank Jain](https://github.com/mayank-02)

## License

MIT
