import numpy as np
import torch.utils.data


class Trainer:
    def __init__(self, device: str):
        self.device = device

    def train(self, model, train_ds, val_ds, loss_fn, optimizer, train_batch_size, train_dl_workers,
              val_batch_size, val_dl_workers):
        output_interval = 100

        model.train()

        train_dl = torch.utils.data.DataLoader(train_ds, collate_fn=train_ds.collate_fn,
                                               batch_size=train_batch_size, shuffle=True, num_workers=train_dl_workers)
        val_dl = torch.utils.data.DataLoader(val_ds, collate_fn=val_ds.collate_fn,
                                             batch_size=val_batch_size, shuffle=False, num_workers=val_dl_workers)

        for epoch in range(10):  # loop over the dataset multiple times
            running_losses = np.zeros((3,))
            for i, data in enumerate(train_dl, 0):
                # get the inputs
                image_batch, y_batch = data
                local_image_batch = image_batch.to(self.device)
                local_y_batch = [(l.to(self.device), c.to(self.device)) for l, c in y_batch]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predicted = model(local_image_batch)
                losses = loss_fn(predicted, local_y_batch)
                loss = losses['total']
                loss.backward()
                optimizer.step()

                running_losses += np.array([loss.item(),
                                            losses['classification'].item(),
                                            losses['localization'].item()])

                # print statistics
                if i % output_interval == output_interval - 1:
                    # TODO: Replase this with a modern version of format string or string interpolation
                    print('[%d, %5d] loss: %.3f, class_loss: %.3f, loc_loss: %.3f' %
                          (epoch + 1, i + 1, running_losses[0] / output_interval,
                           running_losses[1] / output_interval, running_losses[2] / output_interval))
                    running_loss = 0.0
                    running_losses = np.zeros((3,))

            val_dl_iterator = iter(val_dl)
            running_losses = np.zeros((3,))
            num_batches = 0
            for j in range(len(val_batch_size) // val_batch_size):
                x_val, y_val = next(val_dl_iterator)
                local_x_val = x_val.to(self.device)
                local_y_val = [(l.to(self.device), c.to(self.device)) for l, c in y_val]
                val_predicted = model(local_x_val)
                val_losses = loss_fn.batch_losses(val_predicted, local_y_val)
                val_loss = loss_fn.loss(val_predicted, local_y_val)
                num_batches += 1

            print('[%d, %5s] val loss: %.3f, val_class_loss: %.3f, val_loc_loss: %.3f' %
                  (epoch + 1, 'VAL', running_losses[0] / num_batches,
                   running_losses[1] / num_batches, running_losses[2] / num_batches))

        print('Finished Training')
