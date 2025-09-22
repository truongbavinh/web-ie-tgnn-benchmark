# Thin wrapper that imports user's GCN prediction script.
# Produces CSV (e.g., gcn_predictions3.csv) in CWD.

from .user_scripts.gcn_prediction import *
if __name__ == "__main__":
    pass