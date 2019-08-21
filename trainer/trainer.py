import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop

N_ITERS = 30
lr = 0.00001
use_flip_loss = False
use_flip_loss_valid = False
epoch = 0
loss_thresh = 0.002
if TRAIN:
    t = tqdm(range(N_ITERS), desc="epoch: ")
    for i in t:
        optim = torch.optim.Adam(model.parameters(), lr)
        rec = True
        inner = tqdm(dataloader, "batch: ", leave=False)
        ignored = []
        batch_no = 0
        for batch in inner:
            # ========= Training loss ==============
            optim.zero_grad()
            images = batch[0].to(DEVICE, dtype=torch.float)
            labels = batch[1].to(DEVICE, dtype=torch.float)

            # Forward pass
            NN_out = model(images)
            if use_flip_loss:
                #                 loss = invariant_mse_loss(NN_out, labels)
                loss = flip_loss(NN_out, labels)
            else:
                loss = criterion(NN_out, labels)

            if loss.item() <= loss_thresh:
                # Training
                loss.backward()
                optim.step()

                loss_arr.append(loss.item())

                model.eval()
                # ========= Validation loss ==============
                if batch_no % 1 == 0:
                    batch_iter_valid = iter(dataloader_valid)
                    batch_valid = batch_iter_valid.__next__()
                    input_valid = batch_valid[0].to(DEVICE, dtype=torch.float)
                    label_valid = batch_valid[1].to(DEVICE, dtype=torch.float)
                output_valid = model(input_valid)
                if use_flip_loss_valid:
                    loss_valid = flip_loss(output_valid, label_valid)
                else:
                    loss_valid = criterion(output_valid, label_valid)
                valid_arr.append(loss_valid.item())

                # ========= Display loss ==============
                model.train()
                inner.set_description("loss: {:.6f}, v_loss: {:.6f}".format(loss.item(), loss_valid.item()))
                # Set the first batch loss as the loss in the tqdm description
                if rec == True:
                    t.set_description("loss: {:.8f}, v_loss: {:.8f}".format(loss.item(), loss_valid.item()))
                    rec = False
                batch_no += 1
            else:
                ignored.append(loss.item())

        print("epoch {}:lr={}, loss={}, v_loss={}".format(epoch, lr, loss.item(), loss_valid.item()))
        torch.save(model, "/storage/data/models/wings_resnet34_color_256x256")
        if epoch % 5 == 0:
            lr = lr * 0.5
        epoch += 1


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
