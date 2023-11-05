
import torch
from torch import nn
torch.manual_seed(0)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) # concat them
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # energy shape: (N, heads, query_len, key_len) table with attention on
        # each word from target to input
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        # since value_len == key_len i use l for both
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim,
        ) # flatten last 2 dimensions
        
        out = self.fc_out(out)
        return out
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
            self,
            scr_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(scr_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )    
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length,  = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # x B, Seq_len, vocab_size
        # pos B, Seq_len, n_embd
        for layer in self.layers:
            # since we are in encoder and values, queries and keys are the same
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    # valule and key are from encoder
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size, 
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            print(x.shape, enc_out.shape)
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x) 
        return out
        
        
class Transformer(nn.Module):
    def __init__(
            self,
            scr_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=63,
            num_layers=6,
            forward_expansion=4,
            heads=9,
            dropout=0,
            device="cuda",
            max_length=128
    ):
        super().__init__()
        
        self.encoder = Encoder(
            scr_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # (N, 1, 1, src_length)
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # return src_mask.to(self.device)
        return None
    
    def make_trg_mask(self, trg):
        N, trg_length = trg.shape
        trg_mask = torch.tril(torch.ones((trg_length, trg_length))).expand(
            N, 1, trg_length, trg_length
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    #trg = torch.full((x.shape[0], 1), SOS_TOKEN, dtype=torch.long).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 40
    trg_vocab_size = 40
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    
    x = torch.randint(0, 40, size=(2, 20)).to(
        device
    )
    trg = torch.randint(0, 40, size=(2, 32)).to(device)
    out = model(x, trg)
    print (out.shape)
    # def inference(input_ids):
    #     max_length = 30  # Adjust as needed
    #     decoder_input_ids = torch.full((input_ids.shape[0], 1), 1, dtype=torch.long)
    #     for step in range(max_length):
    #         with torch.no_grad():
    #             logits = model(input_ids, decoder_input_ids)[:, -1, :]
    #         predicted_token = torch.argmax(logits, axis=-1)
    #         decoder_input_ids = torch.cat((decoder_input_ids, predicted_token.unsqueeze(1)), axis=-1)
            
    #         if torch.all(predicted_token == 7):
    #             break

    #     print(decoder_input_ids)
        
    # inference(x1)
    # print("\n\n")
    # inference(x2)

        
        
        
        
        
        
        
        
        
        
        
        
        