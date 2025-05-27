# NEI-Marker

In Natural Language Processing (NLP) Natural Entity Identification(NEI) is one of the common problem. The entity is referred to as the part of the text that is interested in. In NLP, NEI is a method of extracting the relevant information from a large corpus and classifying those entities into predefined categories such as location, organization, name and so on. Todat we will be implementing Natural Entity Identification using CRF. It uses IOB scheme to do NEI. Information about lables:

Prefix
B = Beginning of an entity
I = Inside an entity
Labels
MISC= Miscellanous enitity
PER = Person
ORG = Organisation
LOC = Location
O = Outside Entity(Not in an entity
Total Words Count = 23623

The NEI marker is a tool designed to process a given sentence and assign NEI tags to its words or tokens. NEI tags typically classify parts of the sentence into specific categories such as persons, location or organisations.) (or a similar set depending on the tagging scheme). The tagging helps in identifying and labeling key components in the sentence for further natural language processing tasks such as entity recognition, sentiment analysis, or information extraction.

The HMM_Pos_Tagging.py file contains a hmm model that is used to get the pos tags of a sentence. These pos tags are used a features in the crf model.

The NEI_marker.py file contains the implementation of the NEI marker. It uses the conll2003 dataset that contains 23623 words, their pos and NEI tags.  It pre-processes the data by formatting the dataset in a desirable manner and extracts its features. It also trains a CRF model to do NEI marking.

The GUI.py file implements a GUI to do a demo of the NEI marking. It imports hmm_modle from the HMM_Pos_tagging.py file to get the pos tags of the sentence and then it imports extract features function in the NEI_marker.py file and uses it on the the sentence given by the user. It shows the predicted NEI tags by the crf model in the output box.
