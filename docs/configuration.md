# Configuration Guide

Different configuration is needed for different model class you are using.

If you are using **BaseModel** class, you need to specify the following configuraitons.

- pretrained (Boolean): If the image classifier should be pretrained
- num_ft_layers(Integer): The number of layers before output should be freed for fine-tuning
- weighted_loss(Boolean): If weighted loss function should be used for classes imbalance
- epoch(Integer): Epoch for training for a single round of active learning training loop
- loss_function(Pytorch Loss Function<span style="color:red"> *Class*</span>): The loss function <span style="color:red">*class*</span> for training
- labeled_batch_size(Integer): Size of mini-batch for computing the gradient on labeled images
- unlabeled_batch_size(Integer): Size of mini-batch for computing the gradient on unlabeled images

If you are using **LossPredictBaseModel** class, you need further specify the following configurations.

- epoch_loss(Integer): During the training, beyond this number of epoch, the loss of the loss predicting model will not be used for updating the gradient of the classifier
- margin(Float): The margin used in the pair comparison loss(the crafted loss for the loss predicting model)
- lambda(Float): Weight for the loss predicting model loss in training

If you are using **MEBaseModel** class, you need further specify the following configurations.

- query_schedule(list of Integers): If active learning query is combined during training, this item must be provided. Query_schedule is a list of epochs(< configs['epoch']) where the training will pause to wait for new labels input and updating the model
- alpha(Float): Alpha parameter in the exponential moving average
- ramp_length(Integer):The length of the epochs that weight of loss from unlabeled samples ramps up



If you are using **TEBaseModel class**, you need to specify the following configuraitons.

- batch_size(Integer): Size of mini-batch for computing the gradient on both labeled and unlabeled images
- alpha(Float): A momentum term that controls how far the ensemble reaches into training history
- ramp_length(Integer): Define the period for the ramp-up of unsupervised loss. The period of ramp-up is from epoch 0 to epoch `ramp_length`
- weighted_loss(Boolean): If weighted loss function should be used for classes imbalance
- epoch(Integer): Total epoch for the Semi-Supervised training Loop
- query_schedule(List of Integer): The epoch you choose to pause the Simi-Supervised training and add human annotation to model. If you pass a list [10,30,50,100], the training will pause at epoch [10,30,50,100], you can use the `update_target() `function to add the true labels for selected unlabeled images in your dataset. This function is used for the combination of Semi-Supervised Learning and Active Learning



If you are using **NoisyStudentBaseModel** class, you need to specify the following configuraitons.

- student_epoch(List of Integer): A scheduler decided the number of training epochs for each student model. If student_epoch = [10,20], then first student model will train 10 epochs and second student model will train 20 epochs
- labeled_batch_size(Integer): Size of mini-batch for computing the gradient on labeled images
- num_ft_layers(Integer): The number of layers before output should be freed for fine-tuning
- weighted_loss(Boolean): If weighted loss function should be used for classes imbalance
- loss_function(Pytorch Loss Function<span style="color:red"> *Class*</span>): The loss function <span style="color:red">*class*</span> for training
- dropout(Float): the ratio of dropout applied the final layer

