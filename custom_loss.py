import torch
import torch.nn.functional as F

class EnhancedColbertPairwiseCELoss(torch.nn.Module):
    """
    Enhanced implementation of ColBERT pairwise cross-entropy loss
    with better numerical stability and debugging options.
    """
    def __init__(self, temperature=0.05, debug=False):
        super().__init__()
        self.temperature = temperature
        self.debug = debug
        
    def forward(self, query_embeddings, doc_embeddings):
        """
        Compute pairwise CE loss between query and document embeddings.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, query_length, embedding_dim]
            doc_embeddings: Tensor of shape [batch_size, doc_length, embedding_dim]
            
        Returns:
            Loss value (scalar tensor)
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        
        batch_size = query_embeddings.size(0)
        
        if batch_size != doc_embeddings.size(0):
            raise ValueError(f"Batch size mismatch: {batch_size} vs {doc_embeddings.size(0)}")
            
        if self.debug:
            print(f"Query embeddings shape: {query_embeddings.shape}")
            print(f"Doc embeddings shape: {doc_embeddings.shape}")
            print(f"Query norm: {torch.norm(query_embeddings, dim=-1).mean()}")
            print(f"Doc norm: {torch.norm(doc_embeddings, dim=-1).mean()}")
            
        if torch.isnan(query_embeddings).any() or torch.isnan(doc_embeddings).any():
            if self.debug:
                print("WARNING: NaN values detected in embeddings")
            return torch.tensor(1.0, device=query_embeddings.device, requires_grad=True)
            
        total_loss = 0.0
        
        for i in range(batch_size):
            q_emb = query_embeddings[i]
            d_emb = doc_embeddings[i]
            
            similarity = torch.matmul(q_emb, d_emb.transpose(0, 1)) / self.temperature
            
            max_sim_per_q, _ = similarity.max(dim=1)
            
            log_sum_exp = torch.logsumexp(similarity, dim=1)
            
            token_loss = -max_sim_per_q + log_sum_exp
            
            example_loss = token_loss.mean()
            
            if self.debug:
                print(f"Example {i} loss: {example_loss.item()}")
                print(f"Max sim: {max_sim_per_q.mean().item()}, LogSumExp: {log_sum_exp.mean().item()}")
                
            total_loss += example_loss
            
        loss = total_loss / batch_size
        
        if torch.isnan(loss) or torch.isinf(loss) or loss == 0.0:
            if self.debug:
                print(f"WARNING: Abnormal loss value: {loss.item()}")
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            
        return loss


class MaxMarginLoss(torch.nn.Module):
    """
    Alternative loss function using max-margin loss,
    which might be more robust for this application.
    """
    def __init__(self, margin=0.2, debug=False):
        super().__init__()
        self.margin = margin
        self.debug = debug
        
    def forward(self, query_embeddings, doc_embeddings):
        """
        Compute max-margin loss between query and document embeddings.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, query_length, embedding_dim]
            doc_embeddings: Tensor of shape [batch_size, doc_length, embedding_dim]
            
        Returns:
            Loss value (scalar tensor)
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        
        batch_size = query_embeddings.size(0)
        
        if batch_size != doc_embeddings.size(0):
            raise ValueError(f"Batch size mismatch: {batch_size} vs {doc_embeddings.size(0)}")
            
        if self.debug:
            print(f"Query embeddings shape: {query_embeddings.shape}")
            print(f"Doc embeddings shape: {doc_embeddings.shape}")
            
        total_loss = 0.0
        
        for i in range(batch_size):
            q_emb = query_embeddings[i]
            d_emb = doc_embeddings[i]
            
            similarity = torch.matmul(q_emb, d_emb.transpose(0, 1))
            
            max_sim = similarity.max()
            
            neg_loss = 0.0
            neg_count = 0
            
            for j in range(batch_size):
                if j != i:
                    d_neg = doc_embeddings[j]
                    neg_sim = torch.matmul(q_emb, d_neg.transpose(0, 1)).max()
                    neg_loss += torch.clamp(self.margin - max_sim + neg_sim, min=0.0)
                    neg_count += 1
            
            if neg_count > 0:
                neg_loss = neg_loss / neg_count
                example_loss = neg_loss
            else:
                example_loss = torch.tensor(0.1, device=query_embeddings.device, requires_grad=True)
            
            total_loss += example_loss
            
        loss = total_loss / batch_size
        
        if torch.isnan(loss) or torch.isinf(loss) or loss == 0.0:
            if self.debug:
                print(f"WARNING: Abnormal loss value: {loss.item()}")
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            
        return loss