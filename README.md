# Sentiment Classification using simple FNN

## Description

This is a sentiment classification using nlp for my course. The project utilizes a 
simple feedforward neural network (FNN) to classify the sentences into negative (0)
or positive (1) class.

## On dataset

- train.txt: is used for training, the pattern within the training data is `[label] [sentence]` 
- dev.txt: is used for testing, the pattern within the testing data is `[label] [sentence]` 
- glove file: contains multiple word embeddings for words, the pattern within the glove file is `[word] [embeddings]`

## Preparing data
1. Data is prepared by (first) loading the glove file into a dictionary which will be used to map words.
2. Training & dev files are split by lines, normalized into lower case and all special symbols were removed.
3. The lines are then tokenized using WhiteSpaceTokenizer()

## TextDataset & Dataloader
A custom dataset was created using torch.utils.data.dataset. 
Its functions include `__len__` and `__getitem__`
The `__getitem__` function utilizes the glove file to map inputs into embeddings, and return the torch version of both inputs and features.
The dataloader was also used to load train and dev data for training & evaluation.

## Model & Training
The model has a simple architecture: first, inputs are being passed into a Linear layer
which transform the inputs by multiplying it with a matrix, and uses a ReLU function
to activates it.
The next layer uses another Linear function to transform the hidden layer outputs, and uses a
sigmoid function to map it to the correct output.

## Evaluation & results
The training model was optimized using Adam optimizer with a learning rate of 1e-4.

## Further improvement
1. Collater could be used to ensure the width of training is covering all sentences length
2. Further text preprocessing to remove stop words, recognize Named Entity
3. Loading training data in batch size