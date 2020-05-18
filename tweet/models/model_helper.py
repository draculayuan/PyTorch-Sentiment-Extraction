import torch

def sentimented_embedding(model,
                          input_ids,
                          token_type_ids=None,
                          attention_mask=None,
                          position_ids=None,
                          head_mask=None):
    '''
    only applicable for QA mode under RoBERTa 
    '''
    assert head_mask is None # does not support head mask now
    head_mask = [None] * model.config.num_hidden_layers
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
        
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    '''
    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * self.config.num_hidden_layers
    '''
    
    embedding_output = model.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
    
    # add in sentiment, but note that this sentiment is being affected by position id (at least i think)
    for sample_idx in range(embedding_output.size(0)):
        end_pos = input_ids.size(1) - 1
        while input_ids[sample_idx, end_pos].item() != 2: # 2 is the end tag for RoBERTa
            end_pos -= 1

        if end_pos < 5 or input_ids[sample_idx, end_pos-2].item() != 2:
            continue
        else:
            sent_embed = embedding_output[sample_idx, end_pos-1, :]
            embedding_output[sample_idx, :end_pos-3] += sent_embed
            
    encoder_outputs = model.encoder(embedding_output,
                                   extended_attention_mask,
                                   head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
    
    return outputs
        