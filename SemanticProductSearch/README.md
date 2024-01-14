This project is about using sentence transformers in semantic product search. Dataset used in this project is from Kaggle (https://www.kaggle.com/c/home-depot-product-search-relevance).
For only evaluation eval.py can be used with a pickled model that takes a batch of queries and products as input and produces relevance predictions between 1-3. 
There are several training configurations in the main.py that produce such models. Commented out lines at the end of the main.py are some examples of training commands.
