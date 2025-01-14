from typing import Iterable
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    """
    This SkipGram implementation is based on Mikolovs et al., 2013 
    paper "Efficient Estimation of Word Representations in Vector Space".
    """
    
    @classmethod
    def create_from_checkpoint(cls, checkpoint_path: str, vocab_map: dict, embedding_dim: int, context_size: int = 1, unknown_token: bool = True) -> "SkipGram":
        """Creates a SkipGram model from a checkpoint.

        :param checkpoint_path: Checkpoint path
        :type checkpoint_path: str
        :param vocab_map: Vocabulary to be embedded
        :type vocab_map: dict
        :param embedding_dim: Dimension of the latent space which represents the vocabulary.
        :type embedding_dim: int
        :param context_size: Context of the input word for the SkipGram model, defaults to 1
        :type context_size: int, optional
        :param unknown_token: Due to memory issues of missing words in the coprus, 
        having a representation for a new or unknown word is usefull.
        If not wanted set it to False, defaults to True
        :type unknown_token: bool, optional
        :return: A SkipGram model to embed the vocabulary
        :rtype: SkipGram
        """
        skipgram = cls(vocab_map, embedding_dim, context_size, unknown_token)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        skipgram.load_state_dict(checkpoint["model"])

        return skipgram

    def __init__(self, vocab_map: dict, embedding_dim: int, context_size: int = 1, unknown_token: bool = True) -> None:
        """
        Creates a SkipGram model for a given vocabulary. 
        Vocabulary is the NLP term, but can be extended to any kind of tokens like products, actions.

        :param vocab_map: Vocabulary to be embedded
        :type vocab_map: dict
        :param embedding_dim: Dimension of the latent space which represents the vocabulary.
        :type embedding_dim: int
        :param context_size: Context of the input word for the SkipGram model, defaults to 1
        :type context_size: int, optional
        :param unknown_token: Due to memory issues of missing words in the coprus, 
        having a representation for a new or unknown word is usefull.
        If not wanted set it to False, defaults to True
        :type unknown_token: bool, optional
        """
        super(SkipGram, self).__init__()

        self.vocab_map = vocab_map
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.unknown_token = unknown_token

        self.vocab_size = len(vocab_map) + 1 if self.unknown_token else len(vocab_map)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
        self.linear_out = nn.ModuleList([nn.Linear(self.embedding_dim, self.vocab_size)]*self.context_size)

    def forward(self, inputs: torch.Tensor) -> list:
        """Forward function for the SkipGram model.

        :param inputs: A tensor with ids.
        :type inputs: torch.Tensor
        :return: Linear layer output in a list of context size.
        :rtype: list
        """
        embeddings = self.embedding(inputs)
        return [linear(embeddings) for linear in self.linear_out]
    
    def embed(self, sequence: torch.tensor) -> torch.LongTensor:
        """Embed a given sequence.

        :param sequence: A sequence of ids that should be embedded.
        :type sequence: torch.tensor
        :return: An embedded sequence with the same shape as the input sequence.
        :rtype: torch.tensor
        """
        return self.embedding(sequence)
    
    def map_words(self, words: Iterable) -> list:
        """Maps word tokens to the coresponding id for the one hot encoding.

        :param words: Words of a Iterable 
        :type words: Iterable
        :return: A list of ids for the given words
        :rtype: list
        """
        assert isinstance(words, Iterable), f"Parameter words must be an iterable, but type {type(words)} is not iterable."
        return [self.vocab_map.get(word, self.vocab_size - 1) for word in words]
    
    def update_embedding(self, new_vocab_map: dict) -> None:
        """
        Updates the embedding weights for a new vocabulary dictionary. 
        Necessary if new vocabulary is added to the relevant corpus.
        This function is necessary for incremental/online learning.
        If self.unknown_token == True new vocab get the weights for the unknown_token 
        else the weigths are randomly initialised.

        :param new_vocab_map: The new vocabulary map which maps the vocabulary to ids.
        :type new_vocab_map: dict
        """
        # get the new vocabulary size
        new_vocab_size = len(new_vocab_map) + 1 if self.unknown_token else len(new_vocab_map)
        # create new embedding and linear layers that matches the new vocabulary size
        new_embedding = nn.Embedding(new_vocab_size, self.embedding_dim, sparse=True)
        new_linear_out = nn.ModuleList([nn.Linear(self.embedding_dim, new_vocab_size)]*self.context_size)

        # Shallow copy of the newly created embedding and linear layers.
        new_embedding_weights_tensor = new_embedding.state_dict()["weight"]
        new_linear_weights_tensor = list(new_linear_out.state_dict().values())

        # Copy old weights
        # first value is the embedding weights tensor
        # rest is linear layer weights and bias in following order [linear0.weight, linear0.bias, linear1.weight,...]
        old_skipgram_weights_tensor = list(self.state_dict().values()) 
        
        # Iteration over all items in the new vocabulary.
        # The key represents the word and the value the word index in the new embedding
        for word, word_id in new_vocab_map.items():
            # check if the new word is in the old vocabulary map or unknown token is true.
            if word in self.vocab_map or self.unknown_token:
                
                old_word_id = self.vocab_map.get(word, self.vocab_size - 1)
                # set the appropiated embedding weights for the new word
                new_embedding_weights_tensor[word_id] = old_skipgram_weights_tensor[0][old_word_id]

                # for all linear layers set the appropriate embedding weights for the new word
                for i, layer in enumerate(old_skipgram_weights_tensor[1:]):
                    new_linear_weights_tensor[i][word_id] = layer[old_word_id]
        
        # update unknown token if its true.
        if self.unknown_token:
            # set the appropriated embedding weights for the unknown token
            # Note: The unknown token is ALWAYS the last id of the embedding model -> last id = vocab_size - 1 
            new_embedding_weights_tensor[new_vocab_size - 1] = old_skipgram_weights_tensor[0][self.vocab_size - 1]

            # for all linear layers set the appropiated embedding weights for the unknown token
            for i, layer in enumerate(old_skipgram_weights_tensor[1:]):
                new_linear_weights_tensor[i][new_vocab_size - 1] = layer[self.vocab_size - 1]
        
        # update the class attributes
        self.vocab_map = new_vocab_map
        self.vocab_size = new_vocab_size
        self.embedding = new_embedding
        self.linear_out = new_linear_out
    
    def fit(self, dataloader: Iterable, optimizers: list, device = torch.device("cpu")) -> float:
        """Trains the SKipGram model with a given dataloader.

        :param dataloader: A arbitrary dataloader that iterates over the batches.
        The batch consists of features and targets.
        Features is of type torch.Tensor and have the shape (batch_size).
        Targets is of type torch.Tensor and have the shape (batch_size, contex_size)
        :type dataloader: Iterable
        :param optimizers: List of optimizer to optimize the weights.
        If sparse embedding is used a sparse optimizer have to used additionally.
        :type optimizer: list
        :param device: Device where to compute the training process, defaults to torch.device("cpu")
        :type device: torch.device, optional
        :return: Mean loss over all batches.
        :rtype: float
        """
        train_loss = []

        # Change the model into train mode if necessary.
        self.train()
        # Move the model to the device which should compute the training
        self.to(device)

        # iterates over the batches provided by the datalaoder.
        for batch in dataloader:
            features, targets = batch

            # move features and targets to the used device
            features = features.long().to(device)
            targets = targets.long().to(device)

            # set the gradients to zero.
            for optimizer in optimizers:
                optimizer.zero_grad()

            # make a prediction
            out = self(features)
            
            # compute the loss for each output
            # CrossEntropyLoss is used for multiclass classification tasks 
            # and is a combination of softmax activation and negative log-likelihood loss
            loss = sum([nn.CrossEntropyLoss()(o, targets[:,i]) for i, o in enumerate(out)])

            # backpropagation
            loss.backward()

            # update the weigths
            for optimizer in optimizers:
                optimizer.step()

            # add the loss of this batch 
            train_loss.append(loss.item())
        
        return sum(train_loss)/len(train_loss)

if __name__ == "__main__":
    pass
