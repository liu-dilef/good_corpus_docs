# good_corpus_docs

Simple model to discriminate between good and bad documents of a corpus. The model is based on a Random Forest classifier and discriminates mainly textual documents from noisy files (e.g. forms, table data, symbols, etc.).

The file **rf_model.joblib** is the serialization of a *scikit-learn* model.

The file **gcd.py** contains the Python code to use the model.
