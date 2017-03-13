# CNER - Chinese Named Entity Recognition

This is a python (Keras and TensorFlow in particular) implementation of NER tagging, specializing in Chinese. The model structure is very close to what's described in 
"Named Entity Recognition with Bidirectional LSTM-CNNs" by Chiu, Jason P. C.; Nichols, Eric and "Natural Language Processing (almost) from Scratch" by 
Collobert, Ronan et al. 

The model is trained (for now) on PKU's labeled corpus of People's Daily published in Jan. 1998 ("?????????:??·????·??", Journal of Chinese Language and Computing, 13 (2) 121-158). This corpus has a rich set of labels; only "nr", "nrf" and "nrg" for persons, "ns" for locations, and "nt" for organizations are used for this model. 

The performance of the model is slightly worse than the Stanford NLP's NER module for Chinese. The average FB1 on test data for the Person/Location/Organization categories is around 55% (FB1 (Person) ~ 70%, FB1 (Location) ~56% and FB1 (Organization) ~ 23%). 

## Usage