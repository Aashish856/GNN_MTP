import torch
from tqdm import tqdm
from .utils.rotations import randomRotate
from .utils.helper import pairwise_distances
from .models.gnn import gnn_model

def train(model, dataset, optimizer, scheduler, cutoff, loss_fn, device, perform_rotations=False):
    model.train()
    total_loss = 0
    total_samples = 0
    individual_losses = None

    for x, y in tqdm(dataset):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        if perform_rotations:
            x = randomRotate(x)

        adjacency_matrix = ((pairwise_distances(x) < cutoff).float() - torch.eye(x.shape[1]).to(device))
        edge_list = [torch.nonzero(adjacency_matrix[i], as_tuple=False).t() for i in range(x.shape[0])]

        x_dash = x.view(-1, 3)
        for i in range(len(edge_list)):
            edge_list[i][0] += x.shape[1] * i
            edge_list[i][1] += x.shape[1] * i
            edge_list[i] = torch.transpose(edge_list[i], 0, 1)

        edge_dash = torch.transpose(torch.cat(edge_list), 0, 1).to(device)
        pred = model(x_dash, edge_dash)

        num_cvs = pred.shape[1]  # number of CVs dynamically from output shape
        if individual_losses is None:
            individual_losses = [0] * num_cvs
        y = y.squeeze()
        
        loss_cv = []
        for cv_idx in range(num_cvs):
            cv_loss = loss_fn(pred[:, cv_idx], y[:, cv_idx])
            individual_losses[cv_idx] += cv_loss.item() * y.size(0)
            loss_cv.append(cv_loss)


        loss = sum(loss_cv)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_individual_losses = [l / total_samples for l in individual_losses]
    return avg_loss, avg_individual_losses


def eval(model, dataset, cutoff, loss_fn, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    individual_losses = None

    with torch.no_grad():
        for x, y in tqdm(dataset):
            x, y = x.to(device), y.to(device)
            y = y.squeeze()

            adjacency_matrix = ((pairwise_distances(x) < cutoff).float() - torch.eye(x.shape[1]).to(device))
            edge_list = [torch.nonzero(adjacency_matrix[i], as_tuple=False).t() for i in range(x.shape[0])]

            x_dash = x.view(-1, 3)
            for i in range(len(edge_list)):
                edge_list[i][0] += x.shape[1] * i
                edge_list[i][1] += x.shape[1] * i
                edge_list[i] = torch.transpose(edge_list[i], 0, 1)

            edge_dash = torch.transpose(torch.cat(edge_list), 0, 1).to(device)
            pred = model(x_dash, edge_dash)

            num_cvs = pred.shape[1]
            if individual_losses is None:
                individual_losses = [0] * num_cvs

            for cv_idx in range(num_cvs):
                cv_loss = loss_fn(pred[:, cv_idx], y[:, cv_idx])
                individual_losses[cv_idx] += cv_loss.item() * y.size(0)

            loss = loss_fn(pred, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_individual_losses = [l / total_samples for l in individual_losses]
    return avg_loss, avg_individual_losses


def run_training(h_dim, cutoff, n_layer, n_atm, train_dataloader, val_dataloader, loss_fn, device, num_epochs=1, learning_rate = 0.0008, perform_rotations=False):
    model_name = f"gnn_model_{h_dim}_{int(cutoff*1000)}_{n_layer}_{num_epochs}_rot{int(perform_rotations)}"
    model, optimizer, scheduler = gnn_model(h_dim, n_layer, n_atm,  learning_rate, device)
    model.to(device)
    print(f"Starting training for {model_name}...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_individual_loss = train(model, train_dataloader, optimizer, scheduler, cutoff, loss_fn, device, perform_rotations)
        print(f"Train loss: {train_loss:.6f}")
        print("Train individual losses:", train_individual_loss)

    val_loss, val_individual_loss = eval(model, val_dataloader, cutoff, loss_fn, device)
    print(f"Validation loss: {val_loss:.6f}")
    print("Validation individual losses:", val_individual_loss)

    # Save the final model state_dict after training
    torch.save(model.state_dict(), f"./results/{model_name}.pth")

    print(f"Finished training for {model_name}.\n")
