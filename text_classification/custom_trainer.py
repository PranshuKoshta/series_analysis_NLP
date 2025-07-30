import torch
from torch import nn
from transformers import Trainer 

class CustomTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False):
        labels = inputs.get("labels")

        # Forward Pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()

        self.set_device(device='mps:0') 
        
        # Compute Custom Loss
        loss_fct = nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights, dtype=torch.float).to(device=self.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels ),labels.view(-1))
        return (loss,outputs) if return_outputs else loss

    def set_class_weights(self,class_weights):
        self.class_weights = class_weights
    
    def set_device(self,device):
        self.device = device

    # --- ADD THIS METHOD TO FIX THE SAVING ERROR ---
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Overrides the Trainer's save method to make tensors contiguous before saving.
        """
        # Ensure all model parameters are contiguous
        for param in model.parameters():
            param.data = param.data.contiguous()
        
        # Call the original save method to perform the actual save operation
        super()._save_checkpoint(model, trial, metrics)