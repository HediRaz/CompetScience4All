import torch
import os
import numpy as np
import json
folders = ["Models"]

all_preds = torch.zeros((1_000_000, 0), dtype=torch.float32)

for folder in folders:
    for c in range(len(os.listdir(folder))):
        preds = torch.load(f"Models/{c}/preds.pt").to("cpu")
        preds = torch.unsqueeze(preds, 1)
        all_preds = torch.cat((all_preds, preds), 1)

print(f"On a train {all_preds.shape[1]} tel des fou malades")

all_preds = torch.median(all_preds, 1).values
all_preds = all_preds.detach().numpy()

all_preds = list(np.argsort(-all_preds, 0))

submit_file = "submission.json"
all_idx_list_float=list(map(float, all_preds))
with open(submit_file, "w", encoding="utf8") as json_file:
    json.dump(all_idx_list_float, json_file)


