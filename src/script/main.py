
from model import get_model
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from data import get_train_test_split
from config import get_default_config
import lightning as L

def get_data():
    data_path = '../../data_input/train.csv'
    return pd.read_csv(data_path, usecols=['id', 'text', 'label'])

if __name__ == '__main__':
    cfg = get_default_config()

    df = get_data()
    Xtr, Xval, Ytr, Yval = get_train_test_split(
        df["text"], df["label"], context_length=cfg.context_length, test_size=0.2
    )

    train_dataset = TensorDataset(Xtr, Ytr)
    val_dataset = TensorDataset(Xval, Yval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    model = get_model(cfg)
    
    # train with pytorch lightning
    trainer = L.Trainer(max_epochs=2, accelerator='cpu', strategy='ddp_spawn', devices=4)
    trainer.fit(model, train_loader, val_loader)    
    

