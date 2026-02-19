import numpy as np
from torch.utils.data import Dataset
from .utils import AttrDict


class TrajectoryCropDataset(Dataset):
    def __init__(self, dataset, seq_len):
        self.dataset = dataset
        self.seq_len = seq_len
        self.start_ids = self._get_seq_id(dataset, seq_len)
        
    def __len__(self):
        return len(self.start_ids)
    
    def __getitem__(self, idx):
        output = AttrDict()
        start_id = self.start_ids[idx]
        for k, v in self.dataset.items():
            output[k] = v[start_id:start_id + self.seq_len]
        return output
    
    def _get_seq_id(self, dataset, length):
        ids = []
        start_id = 0
        terminal_ids = np.where(dataset["terminals"])[0]
        end_ids = terminal_ids.tolist()
        if "timeouts" in dataset.keys():
            timeout_ids = np.where(dataset["timeouts"])[0]
            end_ids += timeout_ids.tolist()
        end_ids.sort()
        for end_id in end_ids:
            for index in range(start_id, end_id):
                if end_id + 1 < index + length:
                    break
                else:
                    ids.append(index)
            start_id = end_id + 1
        return ids

def make_dataset(buffers:list):
    datas = []
    for buffer in buffers:
        batches = [episode.as_batch() for episode in buffer]
        batch_cat = [np.concatenate(prop, axis=0) for prop in zip(*batches)]

        terminals = [np.zeros((batch[0].shape[0], 1), dtype=bool) for batch in batches]
        for terminal in terminals:
            terminal[-1] = True
        batch_cat.append(np.concatenate(terminals, axis=0))
        datas.append(batch_cat)
    
    datas = [np.concatenate(prop, axis=0) for prop in zip(*datas)]
    return {
        "observations": datas[0], "actions": datas[1], "rewards": datas[2], 
        "next_observations": datas[3], "dones": datas[4], "terminals": datas[5]
    }