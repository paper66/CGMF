import torch
import numpy as np


def evaluate(model, dataloader, device, criterionMSE, criterionL1, scales, rse, rae):
    model.eval()
    total_loss = 0.0
    total_L1loss = 0.0

    num_samples = 0.0

    total_predictions = []
    total_y = []

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            predictions = model(X)
            MSEloss = criterionMSE(predictions * scales, y * scales)
            L1loss = criterionL1(predictions * scales, y * scales)

            total_predictions.append(predictions)
            total_y.append(y)

            total_loss += MSEloss.item()
            total_L1loss += L1loss.item()

            num_samples += predictions.shape[0] * predictions.shape[1]

    total_predictions = torch.cat(total_predictions, 0)
    total_y = torch.cat(total_y, 0)

    mean_p = total_predictions.mean(0)
    mean_y = total_y.mean(0)
    std_p = total_predictions.std(0)
    std_y = total_y.std(0)
    index = (std_y != 0)
    evl_rse = np.sqrt(total_loss / num_samples) / rse
    evl_rae = (total_L1loss / num_samples) / rae
    evl_correlation = ((total_predictions - mean_p) * (total_y - mean_y)).mean(0) / (std_p * std_y)
    evl_correlation = (evl_correlation[index]).mean().item()

    return total_loss / num_samples, evl_correlation, evl_rse, evl_rae
