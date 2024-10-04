# Toxic Comment Classification using LSTM

This project implements a Long Short-Term Memory (LSTM) model to classify toxic comments into various categories such as toxic, severe toxic, obscene, threat, insult, and identity hate. The dataset consists of over 150,000 comments with binary labels for each class. The LSTM architecture is designed to capture long-term dependencies and effectively classify sequential data.

## Introduction
The primary goal of this project is to develop a deep learning-based model capable of identifying toxic comments from user-generated content. The model classifies comments into six categories: toxic, severe toxic, obscene, threat, insult, and identity hate. LSTM was chosen for its ability to retain long-term dependencies in sequential data, which is crucial for accurate classification in natural language processing tasks.

## Dataset
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Model Overview and Architecture

The LSTM model addresses the limitations of conventional Recurrent Neural Networks (RNNs) by preserving long-range dependencies in sequential input, such as text data. LSTM models are widely used for tasks like speech recognition, time-series prediction, and natural language processing.

### LSTM Components:
1. **Cell State (Ct)**: Serves as long-term memory, carrying information across the entire sequence.
2. **Forget Gate (ft)**: Determines which information should be retained or discarded from the cell state.
3. **Input Gate (it)**: Selects which new data should be added to the cell state.
4. **Output Gate (ot)**: Decides the output data based on the current cell state.

### Model Layers:
- **Embedding Layer (nn.Embedding)**: Converts word indices into dense vectors of fixed size.
- **LSTM Layer (nn.LSTM)**: Processes sequential information and retains long-term dependencies.
- **Linear Layer (nn.Linear)**: Transforms the LSTM output to the required output dimension.
- **Sigmoid Activation (nn.Sigmoid)**: Compresses the final output to values between 0 and 1 for binary classification.

## Dataset

The dataset contains 159,751 comments, each labeled with binary values (0 or 1) across six categories: toxic, severe toxic, obscene, threat, insult, and identity hate. The dataset is preprocessed to make it suitable for deep learning, including tokenization, stopword removal, and padding of sequences.

## Model Parameters and Hyperparameters

- **Embedding Dimension (embedding_dim)**: 200
- **Hidden Dimension (hidden_dim)**: 128
- **Vocabulary Size (vocab_size)**: 65,068
- **Output Dimension (output_dim)**: 6 (corresponding to the six classes)
- **Number of Epochs (num_epochs)**: 5
- **Batch Size**: 32

## Loss Function and Optimizer

- **Loss Function**: Binary Cross-Entropy Loss (`nn.BCELoss`) is used to compute the error between the predicted probabilities and the true labels for each class.
- **Optimizer**: Adam Optimizer (`optim.Adam`) is used to update model parameters by adjusting learning rates based on gradient descent.

## Preprocessing

Several preprocessing steps were implemented to prepare the text data for model training:
1. **Tokenization and Lowercase**: Comments are tokenized and converted to lowercase.
2. **Stopwords Removal**: Common stopwords like "the," "is," and "and" are removed using the NLTK library.
3. **Developing Vocabulary**: A dictionary of words is created based on frequency, filtering out words that occur fewer than five times.
4. **Converting Text to Sequences**: Each comment is transformed into a sequence of word indices.
5. **Padding Sequences**: Comments shorter than 100 words are padded to ensure uniform input lengths.

## Experiments and Results

### Summary Statistics
The dataset contains 159,751 comments, with six binary labels for each class. There were no null values, but significant preprocessing was required to convert text into a format suitable for deep learning models.

### Model Training
- Initial experiments used smaller sample sizes (500 rows) to refine the model architecture and evaluate performance.
- Model training was conducted using Google Colab Pro and HPC lab systems to handle the large dataset.
- Batch size was reduced to 32 for efficient training on large datasets.
- The final model achieved significant improvement by calculating individual accuracies for each class, allowing a better evaluation of the model's performance.
