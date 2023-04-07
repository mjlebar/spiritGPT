import torch 
import torch.nn as nn
from torch.nn import functional as F


#set hyperparameters
scale = 1
block_size = 8 * scale #length of context before character prediction
batch_size = 4 * scale #number of examples simultaneously processed
embed_dim = 32 * scale #embedding dimension
dropout = 0.2 #probability of a neuron dropping out
num_head = 4 * scale
num_blocks = 4 * scale
learning_rate = 3e-4
epochs = 5000
eval_epochs = 200

#### Process text

#read in the file
f = open('phg.txt', 'r', encoding='UTF-8')
text = f.read()
f.close()

#make the vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

#make the encoding and decoding (text to number, number to text)
encoder = {ch:i for i, ch in enumerate(vocab)}
decoder = {i:ch for i, ch in enumerate(vocab)}
def encode_string(string): 
    return [encoder[char] for char in string]
def decode_string(string):
    return ''.join([decoder[char] for char in string])

#split into training and validation sets
data = torch.tensor(encode_string(text), dtype=torch.long)
portion_train = int(0.9*len(data))
training_set=data[:portion_train]
val_set = data[portion_train:]


#make up the batches (including padding + start & end tokens)
def get_batch(type):
    data = training_set if type == 'train' else val_set
    indices = torch.randint(len(data)-block_size, (batch_size, ))
    inputs = torch.stack([data[index: index+block_size] for index in indices])
    targets = torch.stack([data[index+1: index+block_size+1] for index in indices])
    return inputs, targets



###set up NN architecture

#head class
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('triangle', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, block, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        sa_weights = k @ q.transpose(1, 2) * channels ** -0.5 
        sa_weights = sa_weights.masked_fill(self.triangle[:block, :block]==0, float('-inf'))
        #note to self - why do we do softmax here?
        sa_weights = F.softmax(sa_weights, dim = 2)
        #self-attention weights
        out = sa_weights @ v
        return out



#multi-head layer class
class MultiHeadLayer(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #creates the heads
        self.projection = nn.Linear(embed_dim, embed_dim) 
        #projects the results of self-attention back into feed forward
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

#layernorm class - pytorch

#feedfoward class
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layers(x)

#block class
class Block(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.self_attention = MultiHeadLayer(num_heads, embed_dim//num_heads)
        self.feed_forward = FeedForward(embed_dim)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        #addition is to have a residual connectiion... here we do add and norm before self attention or feed forward, unlike original paper. Andrej says it's better and I believe him
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
    

#model class:
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, embed_dim)
        #embeds the vocabulary
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        #embeds the positions
        self.blocks = nn.Sequential(*[Block(4, embed_dim) for _ in range(num_blocks)])
        self.layer_norm_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, input, targets = None):
        batches, blocks = input.shape
        character_embedding = self.vocab_embedding(input)
        position_embedding = self.pos_embedding(torch.arange(blocks))
        logits = self.output(self.layer_norm_final(self.blocks(character_embedding + position_embedding)))

        if targets == None:
            loss = None
        else:
            logits = logits.view(batches*blocks, -1)
            targets = targets.view(batches*blocks)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_text(self, input, length):
        for _ in range(length):
            input_for_prediction = input[:, -block_size:]
            logits, loss = self(input_for_prediction)
            logits = logits[:, -1, : ]
            probabilities = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probabilities, 1)
            input = torch.cat((input, next_char), dim=1)
        return input
       
#loss estimate function - this is for the output every ~500 epochs
with torch.no_grad():
    def estimate_loss():
        out_loss = {}
        model.eval()
        for type in ['train', 'validation']:
            losses = torch.zeros(eval_epochs)
            for i in range(eval_epochs):
                inputs, targets = get_batch(type)
                logits, in_loss = model(inputs, targets)
                losses[i] = in_loss.item()
            out_loss[type] = losses.mean()
        model.train()
        return out_loss

###train
model = Model()

#initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#loop through batches input and targets from train:
for epoch in range(epochs):
    #get predictions and loss from model
    inputs, targets = get_batch('train')
    logits, loss = model(inputs, targets)
    
    #zero the gradient
    optimizer.zero_grad()
    #backpropagate
    loss.backward()
    #take a step
    optimizer.step()

    #every 500 steps or the second to last step: output training loss vs validation loss
    if epoch % 500 == 0 or epoch == epochs-1:
        losses = estimate_loss()
        print(f'Epoch {epoch}: Training loss {losses["train"]:.2f}, Validation loss {losses["validation"]:.2f}')


###output:
#use the model's predictions to output text
start = torch.zeros((1, 1), dtype=torch.long)
print(decode_string(model.generate_text(start, 500)[0].tolist()))
open('generated.txt', 'w').write(decode_string(model.generate_text(start, 10000)[0].tolist()))