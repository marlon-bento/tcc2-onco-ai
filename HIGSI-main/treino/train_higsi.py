import torch
import numpy as np
import mlflow.pytorch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from prof.model import HIGSI
from prof.dataset_creation import MG_Cifar10_graphs
from sklearn.metrics import accuracy_score
import sys

RAMDOM_SEED = 42
torch.cuda.manual_seed(RAMDOM_SEED)
torch.manual_seed(RAMDOM_SEED)
np.random.seed(RAMDOM_SEED)

NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 300

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("HIGSI - Target 20 nodes")


# ----------------------------
# Corrige o incremento de índices no batch
# ----------------------------
class MultiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'reduced_index':
            return int(value.max().item()) + 1   # incremento correto baseado no nº de clusters
        if key == 'graph_index':
            return self.ng                       # incremento baseado no nº de grafos
        return super().__inc__(key, value, *args, **kwargs)


# ----------------------------
# Sanitizador simples para consistência
# ----------------------------
def sanitize_batch(batch):
    num_nodes = batch.x.size(0)

    # Ajusta reduced_index
    if batch.reduced_index.numel() != num_nodes:
        batch.reduced_index = torch.arange(num_nodes, dtype=torch.long)

    _, inv = torch.unique(batch.reduced_index, return_inverse=True)
    batch.reduced_index = inv

    # Ajusta graph_index
    if batch.graph_index.numel() != num_nodes:
        batch.graph_index = torch.arange(num_nodes, dtype=torch.long)

    _, inv = torch.unique(batch.graph_index, return_inverse=True)
    batch.graph_index = inv

    # Ajusta labels
    if hasattr(batch, "y"):
        batch.y = batch.y.view(-1).long().clamp(min=0)

    return batch


# ----------------------------
# Funções auxiliares
# ----------------------------
def calculate_metrics(y_pred, y_true, epoch, key):
    acc = accuracy_score(y_pred, y_true)
    print(f"{key} Accuracy: {acc}")
    mlflow.log_metric(key=f"Accuracy-{key}", value=float(acc), step=epoch)
    return acc


def train(epoch, train_loader, device, optimizer, model, loss_fn):
    all_preds, all_labels = [], []
    for batch in train_loader:
        batch = sanitize_batch(batch)  # 🔑 aplica sanitização
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(x=batch.x,
                     edge_index=batch.edge_index,
                     edge_attr=batch.edge_attr,
                     batch_index=batch.batch,
                     graph_index=batch.graph_index,
                     reduced_index=batch.reduced_index)

        loss = loss_fn(pred, batch.y.reshape(len(batch), 10))
        loss.backward()
        optimizer.step()

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch), 10).cpu().numpy(), axis=1))

    return loss, calculate_metrics(np.concatenate(all_preds), np.concatenate(all_labels), epoch, "train"), optimizer


def val(epoch, model, loader, loss_fn, device, key):
    all_preds, all_labels = [], []
    model.eval()
    for batch in loader:
        batch = sanitize_batch(batch)  # 🔑 aplica sanitização
        batch = batch.to(device)

        pred = model(x=batch.x,
                     edge_index=batch.edge_index,
                     edge_attr=batch.edge_attr,
                     batch_index=batch.batch,
                     graph_index=batch.graph_index,
                     reduced_index=batch.reduced_index)

        loss = loss_fn(pred, batch.y.reshape(len(batch), 10))
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch), 10).cpu().numpy(), axis=1))

    return loss, calculate_metrics(np.concatenate(all_preds), np.concatenate(all_labels), epoch, key)


# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='train')
    val_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='val')
    test_data = MG_Cifar10_graphs(root="data/", nodes=20, canonized=True, split='test')

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = HIGSI(node_feature_size=train_data[0].x.shape[1],
                  edge_feature_size=train_data[0].edge_attr.shape[1],
                  num_classes=train_data[0].y.shape[0]).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10,
                                                           min_lr=1e-6, factor=0.5, verbose=True)

    best_accu = 0
    save_dir = f"weights/HIGSI/3_blocks_20_nodes_seed={RAMDOM_SEED}.pth"

    with mlflow.start_run() as run:
        for epoch in range(EPOCHS):
            model.train()
            loss, train_acc, optimizer = train(epoch, train_loader, device, optimizer, model, loss_fn)
            print(f"Epoch {epoch+1} | Train Loss {loss}")

            val_loss, val_acc = val(epoch, model, val_loader, loss_fn, device, "val")
            test_loss, test_acc = val(epoch, model, test_loader, loss_fn, device, "test")

            if val_acc > best_accu:
                best_accu = val_acc
                torch.save(model.state_dict(), save_dir)

            scheduler.step(val_acc)

    print("Train Done")


if __name__ == "__main__":
    main()
