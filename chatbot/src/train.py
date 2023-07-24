import torch
import torch.nn as nn

def epoch_train(model,train_data, optimizer, criterion):

    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_data):

        loss = 0 
        src = batch[0]
        trg = batch[1]
        optimizer.zero_grad()
        output = model(src, trg)

        output_tensor = torch.stack(output['decoder_output'])
        output_tensor_resize = output_tensor.view(-1, output_tensor.shape[-1])
        trg_resize = trg.view(-1) 
        # trg = [trg_len- * batch_size]
        # output = [trg_len-1 * batch_size, output_dim]
        loss = criterion(output_tensor_resize, trg_resize)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss/len(train_data)


def epoch_evaluate(model, val_data, criterion):
    
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        
        for i, batch in enumerate(val_data):
            src = batch[0]
            trg = batch[1]

            output = model(src, trg)

            output_tensor = torch.stack(output['decoder_output'])
            output_tensor_resize = output_tensor.view(-1, output_tensor.shape[-1])
            trg_resize = trg.view(-1)
            loss = criterion(output_tensor_resize, trg_resize)
            epoch_loss += loss.item()
    
    return epoch_loss/len(val_data)

