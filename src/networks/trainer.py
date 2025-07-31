import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from gnn import GATModel, train, evaluate
from utils.utils import load_data_in_batches, GeometricDataset, extract_bounds_from_file


def main():
    val_label_file = "../../valset.json"
    train_data_list = load_data_in_batches(val_label_file, 2)
    exit(0)
    # file_path = "../../array_2-1.i_result.txt"
    # bound, benchmark, labels = extract_bounds_from_file(file_path)
    # print(bound)
    # exit(1)

    # edge_sets = ['AST', 'Data', 'ICFG']
    # dataset = "../../data/final_graphs"
    # train_set = GeometricDataset(trainLabels, dataset, edge_sets, should_cache = False)
    # print(len(train_set))
    # #print(train_set[0])
    # exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}.")
    # in_channels should change
    model = GATModel(in_channels=92, hidden_channels=64).to(device)
    exit(0)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_label_file = "../../data/trainset.json"
    val_label_file = "../../data/valset.json"
    test_label_file = "../../data/testset.json"
    batch_size = 8
    epochs = 10

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        for train_data_list in load_data_in_batches(train_label_file, batch_size):
            if len(train_data_list) == 0:
                continue
            train_loader = DataLoader(train_data_list, batch_size=2, shuffle=True)
            start_time = time.time()
            train_loss += train(model, train_loader, optimizer, criterion, device)
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} , Duration: {epoch_duration:.2f} seconds, "
                f"Using GPU: {device}"
            )
            torch.cuda.empty_cache()
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Train Loss: {train_loss:.4f}")

        val_loss = 0
        for val_data_list in load_data_in_batches(val_label_file, batch_size):
            val_loader = DataLoader(val_data_list, batch_size=2, shuffle=False)
            val_loss += evaluate(model, val_loader, criterion, device)
            torch.cuda.empty_cache()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Val Loss: {val_loss:.4f}")

    # Test loop
    test_loss = 0
    for test_data_list in load_data_in_batches(test_label_file, batch_size):
        test_loader = DataLoader(test_data_list, batch_size=2, shuffle=False)
        test_loss += evaluate(model, test_loader, criterion, device)
        torch.cuda.empty_cache()
    test_loss /= len(test_loader)
    print(f"Avg Test Loss: {test_loss:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), "trained_gat_model.pth")
    print("Model saved to trained_gat_model.pth")


if __name__ == "__main__":
    main()
