import torch
import torch.nn as nn
from evaluate import evaluate
import time


def train_epoch(model, train_dataloader, optimizer, device, criterion, scales, accumulate_step, clip_val):
    model.train()
    train_total_loss = 0.0
    train_num_samples = 0
    total_time = 0
    for i, batch in enumerate(train_dataloader):

        X, y = batch
        X = X.to(device)
        y = y.to(device)
        batch_start_time = time.time()
        predictions = model(X)
        loss = criterion(predictions * scales, y * scales)

        loss_back = loss

        loss_back.backward()

        total_time += time.time() - batch_start_time
        # if (i + 1) % 10 == 0:
        #     print("batch time: ", total_time / (i + 1))

        if i % accumulate_step == 0:
            nn.utils.clip_grad_value_(model.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad()

        train_total_loss += loss.item()
        train_num_samples += predictions.shape[0] * predictions.shape[1]

    return train_total_loss / train_num_samples


def train(model, data, optimizer, config, scales, rse_val_d, rae_val_d, rse_test_d, rae_test_d, criterionMSE,
          criterionL1, scheduler=None, logger=None):
    device = torch.device(config.device)

    best_val = 0
    val_data = []
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        # train epoch
        train_loss = train_epoch(model, data["train_dataloader"], optimizer, device, criterionMSE, scales,
                                 config.accumulate_step, config.clip_val)

        val_loss, val_corr, val_rse, val_rae = evaluate(model, data["valid_dataloader"], device, criterionMSE,
                                                        criterionL1, scales, rse_val_d, rae_val_d)

        if logger is not None:
            logger.info(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}  '.format(
                    epoch + 1, (time.time() - epoch_start_time), train_loss, val_rse, val_rae, val_corr))

        val_data.append([val_rse, val_corr])
        if val_corr / val_rse > best_val:
            with open(config.model_path, 'wb') as f:
                torch.save(model, f)

            best_val = val_corr / val_rse

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 5 == 0:
            test_loss, test_corr, test_rse, test_rae = evaluate(model, data["test_dataloader"], device, criterionMSE,
                                                                criterionL1, scales, rse_test_d, rae_test_d)
            if logger is not None:
                logger.info(
                    "test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_rse, test_rae, test_corr))
