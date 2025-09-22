# Thin wrapper around user's training script for GCN+BERT.
# Expects: graph_pt_output/*.pt and class_weights.pt in CWD.
# Writes: gcn_bert_model.pt and logs to stdout.

from .user_scripts.training_gcn import *  
if __name__ == "__main__":
    pass
