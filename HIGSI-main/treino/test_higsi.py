import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch_geometric.loader import DataLoader
from decouple import config

from model import HIGSI

PASTA_FEATURES = config("PASTA_FEATURES")

def main():
    data_path = os.path.join(PASTA_FEATURES, "HIGSI", "1024", "15.pt")
    dataset = torch.load(data_path)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HIGSI(
        node_feature_size=dataset[0].num_node_features,
        edge_feature_size=dataset[0].num_edge_features,
        num_classes=2,
        embedding_size=200
    ).to(device)

    model.load_state_dict(torch.load("weights/HIGSI_best.pth"))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.graph_index, batch.reduced_index)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    print("Test Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds))

if __name__ == "__main__":
    main()
