"""
Multi-VAE (Variational Autoencoder for Collaborative Filtering)

Reference: Liang et al., "Variational Autoencoders for Collaborative Filtering" (WWW 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse


class MultiVAE(nn.Module):
    """
    Multi-VAE 모델.
    
    Args:
        p_dims: Encoder/Decoder 차원 리스트 (예: [200, 600, num_items])
        dropout: Dropout 비율
    """
    
    def __init__(self, p_dims, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]  # Encoder는 역순
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(len(self.q_dims) - 1):
            self.encoder.append(nn.Linear(self.q_dims[i], self.q_dims[i + 1]))
        
        # Latent space
        self.mu = nn.Linear(self.q_dims[-1], self.q_dims[-1])
        self.logvar = nn.Linear(self.q_dims[-1], self.q_dims[-1])
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(self.p_dims) - 1):
            self.decoder.append(nn.Linear(self.p_dims[i], self.p_dims[i + 1]))
        
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def encode(self, x):
        """Encoder: x -> mu, logvar"""
        h = F.normalize(x, p=2, dim=1)
        h = self.dropout(h)
        
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i < len(self.encoder) - 1:
                h = torch.tanh(h)
        
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decoder: z -> x_recon"""
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1:
                h = torch.tanh(h)
        return h
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch_size, num_items] (binary or counts)
        
        Returns:
            x_recon: [batch_size, num_items] (logits)
            mu, logvar: Latent distribution parameters
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x_recon, x, mu, logvar, anneal=1.0):
        """
        VAE Loss = Reconstruction Loss + KL Divergence.
        
        Args:
            x_recon: Reconstructed logits [batch_size, num_items]
            x: Original input [batch_size, num_items]
            mu, logvar: Latent distribution parameters
            anneal: KL annealing weight (0 to 1)
        
        Returns:
            loss: Scalar loss
        """
        # Reconstruction loss (Multinomial likelihood)
        # BCE with logits for numerical stability
        recon_loss = -torch.mean(torch.sum(F.log_softmax(x_recon, dim=1) * x, dim=1))
        
        # KL Divergence: KL(q(z|x) || p(z))
        # where p(z) = N(0, I), q(z|x) = N(mu, diag(exp(logvar)))
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Total loss
        loss = recon_loss + anneal * kl_loss
        
        return loss, recon_loss, kl_loss


def train_vae_epoch(model, optimizer, train_loader, device, anneal=1.0, scaler=None):
    """
    한 에폭 학습 (Mixed Precision 지원).
    
    Args:
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        avg_loss, avg_recon, avg_kl: Average losses
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    use_amp = scaler is not None
    
    for batch_idx, batch_data in enumerate(train_loader):
        # TensorDataset는 튜플로 반환하므로 언패킹
        if isinstance(batch_data, (list, tuple)):
            batch_data = batch_data[0]
        batch_data = batch_data.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward
        if use_amp:
            # PyTorch 2.1.0에서는 torch.cuda.amp.autocast() 사용
            # FutureWarning 무시 (함수 호출 전에 설정)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                from torch.cuda.amp import autocast
                
                with autocast():
                    x_recon, mu, logvar = model(batch_data)
                    loss, recon_loss, kl_loss = model.loss_function(x_recon, batch_data, mu, logvar, anneal)
            
            # Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_recon, mu, logvar = model(batch_data)
            loss, recon_loss, kl_loss = model.loss_function(x_recon, batch_data, mu, logvar, anneal)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    
    return avg_loss, avg_recon, avg_kl


@torch.no_grad()
def predict_vae(model, user_data, device, k=100, exclude_items=None):
    """
    VAE로 Top-K 아이템 예측.
    
    Args:
        model: Trained VAE model
        user_data: [num_items] sparse vector (user's interaction history)
        device: torch device
        k: Number of items to recommend
        exclude_items: Set of items to exclude (already interacted)
    
    Returns:
        top_k_items: List of item indices
        top_k_scores: List of scores
    """
    model.eval()
    
    # Convert to dense tensor
    if sparse.issparse(user_data):
        user_data = torch.FloatTensor(user_data.toarray()).squeeze()
    else:
        user_data = torch.FloatTensor(user_data)
    
    user_data = user_data.unsqueeze(0).to(device)  # [1, num_items]
    
    # Forward pass
    x_recon, _, _ = model(user_data)
    scores = x_recon.squeeze().cpu().numpy()  # [num_items]
    
    # Exclude already interacted items
    if exclude_items is not None:
        scores[list(exclude_items)] = -np.inf
    
    # Top-K
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_idx]
    
    return top_k_idx.tolist(), top_k_scores.tolist()

