def save_training_checkpoint(model, best_model_state, optimizer, scheduler, loss_train, loss_validation, epoch, path):
  # add things like TPR, FPR later when we start evaluating them
  torch.save({'model_state': model.state_dict(),
              'best_model_state': best_model_state,
              'optimizer_state': optimizer.state_dict(),
              'scheduler_state': scheduler.state_dict(),
              'loss_train': loss_train,
              'loss_validation': loss_validation,
              'epoch': epoch
              }, path)
  print("saved")
  

def load_training_checkpoint(model, optimizer, scheduler, path):
  if not os.path.isfile(path):
    return [], [], -1  # training starts at last epoch + 1
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state'])
  optimizer.load_state_dict(checkpoint['optimizer_state'])
  scheduler.load_state_dict(checkpoint['scheduler_state'])
  return checkpoint['loss_train'], checkpoint['loss_validation'], checkpoint['epoch']
  

def load_best_weights(model, path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['best_model_state'])


def train(model, optimizer, scheduler, loss_func, epochs, datasetLoaders, save_path, loss_train, loss_validation, start_epoch):
  t_start = perf_counter()

  best_model_state = copy.deepcopy(model.state_dict())

  for epoch in range(start_epoch, epochs):
    print("Epoch " + str(epoch) + " out of " + str(epochs))
    for mode in ['training', 'validation']:
      if mode == "training":
        # set model to training mode.
        model.train()
      else:
        # set model to evaluation mode for validation set.
        model.eval()

      # get length of dataset for current mode
      n = len(datasetLoaders[mode].dataset)

      # sum of losses over batches.
      total_loss = 0

      # number of correct predictions.
      total_correct = 0

      epoch_count = 0
      
      for inputs, labels in datasetLoaders[mode]:
        inputs, labels = inputs.to(device), labels.to(device)
        epoch_count += inputs.size(0)
        print("percent {}".format(epoch_count / n))
        # For code brevity, we'll set the reset gradients for model params
        # with the intention of using it for training.
        # We'll use the set_grad_enabled to toggle whether we actually use the
        # gradient for training/validation.
        # https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html
        optimizer.zero_grad()
        use_grad = (mode == 'training')
        with torch.set_grad_enabled(use_grad):
          outputs = model(inputs)
          loss = loss_func(outputs, labels)
          _, preds = torch.max(outputs, 1)
          if use_grad:
            # We are training, so make sure to actually
            # train by using loss/stepping.
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0) # Averages over batch              
        total_correct += (preds == labels.data).sum().item() # what is .data for?

      # calculate average loss over batches
      loss_avg = total_loss / n

      # calculate accuracy
      accuracy = total_correct / n
      
      print("mode: " + mode + ", accuracy: " + str(accuracy) + ", loss: " + str(loss_avg))

      if mode == 'validation':
        # record validation loss
        loss_validation.append(loss_avg)

        if accuracy > np.max(loss_validation):
          # we found a better accuracy with these new weights, so save them
          best_model_state = copy.deepcopy(model.state_dict())

      else:
        # record training accuracy
        loss_train.append(loss_avg)

        # make sure to step through lr update schedule
        #scheduler.step()
    
    save_training_checkpoint(model, best_model_state, optimizer, scheduler, loss_train, loss_validation, epoch, save_path)
      
    
    print("\n")

  t_stop = perf_counter()
  print("Elapsed time during training in seconds",
                                        t_stop-t_start)
  # set our model with best weights
  model.load_state_dict(best_model_state)
  return model, best_validation_accuracy, loss_train, loss_validation


def evaluate(model, loss_func, dataset_loader, test=False):
  # length of data set we are evaluating on.
  n = len(dataset_loader.dataset)

  # initialize the prediction and label lists(tensors) for our confusion matrix
  predlist=torch.zeros(0, dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0, dtype=torch.long, device='cpu')
  conf_mat = None

  # correct predictions.
  correct = 0
  total_loss = 0
  with torch.no_grad():
    for inputs, labels in dataset_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = loss_func(outputs, labels)
      _, predictions = torch.max(outputs, 1)

      # append batch prediction results and labels
      predlist=torch.cat([predlist, predictions.view(-1).cpu()])
      lbllist=torch.cat([lbllist, labels.view(-1).cpu()])

      total_loss += loss.item() * inputs.size(0)  # weighted average with size?
      correct += (predictions == labels).sum().item()  # what is labels.data

  if test:
    # display a Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

  return  correct / n, total_loss / n, conf_mat