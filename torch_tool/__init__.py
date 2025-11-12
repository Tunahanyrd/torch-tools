# torch_tool/__init__.py

from .core import * 

__version__ = "0.1.0"
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Tek Bir Transformatör Bloğu Tanımlama
# Bu, LLM'in temel yapı taşıdır.
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        # a) Çok Başlı Dikkat Mekanizması
        # d_model: embedding boyutu, n_head: dikkat başı sayısı
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_head, 
            dropout=dropout,
            batch_first=True # Girişi (Batch, Sequence, Features) olarak bekler
        )
        
        # b) İleri Beslemeli Ağ (MLP)
        # Genellikle d_model'in 4 katı kadar genişletilir.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), # Tıpkı ReLU gibi ama daha yumuşak bir aktivasyon
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # c) Normalizasyon ve Kalan Bağlantıları (Residual Connections)
        # Her alt katmandan (Dikkat ve FFN) önce Normalizasyon yapılır.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Kendi Kendine Dikkat (Self-Attention) Aşaması
        # Önce Normalizasyon
        norm_x = self.norm1(x)
        # Dikkat (x, x, x) ile K, Q, V'nin aynı olduğunu belirtiriz.
        # attn_output: Dikkat mekanizmasının çıktısı
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        # Dikkat çıktısına Kalan Bağlantı (x) eklenir ve Dropout uygulanır.
        x = x + self.dropout(attn_output)
        
        # 2. İleri Beslemeli Ağ (FFN) Aşaması
        # Önce Normalizasyon
        norm_x = self.norm2(x)
        # FFN'den geçir
        ffn_output = self.ffn(norm_x)
        # FFN çıktısına Kalan Bağlantı (x) eklenir.
        x = x + ffn_output
        
        return x

# 2. Ana LLM Modelinde Katmanları Yığma

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int):
        super().__init__()
        
        # a) Başlangıç Katmanları
        # Kelimeleri vektörlere dönüştürme (Embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Konumsal Bilgi Ekleme (Positional Encoding) bu örnekte basit tutulmuştur.
        # Gerçek modellerde daha karmaşık bir P.E. kullanılır.
        self.pos_embedding = nn.Embedding(512, d_model) # Max 512 token varsayımı

        # b) Katmanları Yığma (Sihir burada!)
        # nn.ModuleList, aynı tipte modülleri ardışık olarak tutmak için kullanılır.
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head) for _ in range(n_layer)
        ])
        
        # c) Bitiş Katmanı
        # Son Normalizasyon ve sonuçları kelime olasılıklarına çevirme (Linear + Softmax)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size) # Kelime Dağılımı Çıktısı

    def forward(self, tokens):
        B, T = tokens.shape # Batch Size (B), Sequence Length (T)
        
        # 1. Girişleri Hazırlama: Embedding + Positional Encoding
        token_emb = self.token_embedding(tokens)
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        pos_emb = self.pos_embedding(pos)
        
        x = token_emb + pos_emb # Embedding'leri topluyoruz
        
        # 2. Transformatör Katmanlarından Geçirme
        for layer in self.layers:
            x = layer(x) # Her katman, bir öncekinin çıktısını alır
            
        # 3. Çıktı Üretimi
        x = self.final_norm(x)
        logits = self.lm_head(x) # Logitler (kelime olasılıklarından önceki değerler)
        
        return logits

