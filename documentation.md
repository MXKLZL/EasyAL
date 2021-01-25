#### **MultiTransformDataset**



*class* **MultiTransformDataset**(path_list, target_list=None, classes=None, class_name_map=None, root_dir='', transform=None)

​	A pytorch dataset subclass that can alternate the type of image transformation by setting the mode. For instance, in active learning tasks, two types of image transformation are needed for a same dataset object. The reason for it is that different transformations are needed for labeled images and unlabeled images, and unlabeled images might get labeled during executing the task. By setting the mode of transformation, you can use intended image transformation type as needed.

Arguments:

​	path_list (list of string): list of path to image files

​    target_list (list of string): list targets of samples, <span style="color:red">*the unlabeled samples should be marked as -1*</span>

​    root_dir (string): root directory to paths

​    classes (list of string): class names corresponding to targets

​    class_name_map(dictionary): mapping between targets and classes

​    transform (list of transformation): list of callable transform object to be applie on a sample, using its index as the mode for the transformation. By default, 0 is the index of transformation used for training(with augmentation), 1 is the index of transformation used for predicting(without augmentation)



function set_mode(mode)

​	Set the current mode for the transformation. The mode is used as the index for the list of transformations.

​	Parameters:

​		mode(integer): index for the list of transformation



function update_target(index, new_target)

​	Update dataset with new labels(in integer form)

Parameters:

​	index(List of Integers): index of samples to update targets

​	new_target(List of Integers):	labels of newly annotated samples



#### BaseModel 

*class* **BaseModel**(dataset, model_name, configs)

​	The base model wrapper of pytorch image classifier for active learning strategies. BaseModel class can be used for random sampling, least confident(maximum uncertain), margin sampling, entropy sampling, k-means sampling, k-means++ sampling, k_center_greedy sampling.

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet']

​	configs(dictionary): configuration for the current active learning task. See configruation guide for details.



function get_labeled_index(dataset)

Parameters:

​	dataset(MultiTransformDataset object): The dataset to parse the index for unlabeled samples.

Returns:

​	list of index for unlabeled samples.



function update()

​	After labeling more samples, update the model to be prepared for the training of next active learning iteration



function fit()

​	Fit classifier based on input data



function predict(data_loader)

​	predict labels (in probabilities) for given data loader

Parameters:

​	data_loader(pyTorch Dataloader): data loader of images to predict label

Returns:

​	A tensor of probabilities of predicted classes for each image



function predict_unlabeled()

​	predict labels (in probabilities) for current unlabeled images

Returns:

​	A tensor of probabilities of predicted classes for each image



function get_embedding(data_loader)

​	get embeddings for each image from a given data loader based on current model. The embedding is the intermediate output of the last layer before fully connected output layer

Parameters:

​	data_loader(pyTorch Dataloader): data loader of images to get embedding

Returns:

​	A tensor of embeddings for each image



function get_embedding_unlabeled()

​	get embeddings for each unlabeled loader based on current model.

Returns:

​	A tensor of embeddings for each image



#### LossPredictBaseModel	

*class* LossPredictBaseModel(dataset, model_name, configs)

​	The subclass of BaseModel to be used for loss sampling and confident coreset strategies.

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet']

​	configs(dictionary): configuration for the current active learning task. See configruation guide for details.



function predict_loss(data_loader)

​	get predicted classification loss for each image of a given data loader

Returns:

​	A tensor of predicted loss



function  predict_unlabeled_loss()

​	get predicted classification loss for each unlabeled image

Returns:

​	A tensor of predicted loss



#### MEBaseModel

*class* **MEBaseModel**(dataset, model_name, configs, test_ds=None, weight=True)

​	The subclass of BaseModel to be used for Mean Teacher semi-supervised learning model. This semi-supervised learning model can be combined with active learning query strategies, excluding loss sampling and confident coreset.

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet']

​	configs(dictionary): configuration for the current active learning task. See configruation guide for details.

​	test_ds(pyTorch Dataset): Dataset object of testset. Default None. If it's 'none', test accuracy will not be calculated.

​	use_weight(Boolean): If training should use loss from unlabeled samples. Default True. If False, model training will be purely supervised.



function pred_acc(testloader, test_target, criterion='f1')

​	Calculate test accuracy on given test set

Parameters:

​	testloader(pyTorch Dataloader): Dataloader of test set to calculate accuracy on

​    test_target(list of Integers): List of targets correponding to given test set

​	criterion(String): Type of evaluation metric to compute. Can be chosen from ['precision','recall','f1'] Default 'f1'

Returns:

​	Evaluation metric calculated on predictions from given test set









### Configuration Guide

Different configuration is needed for different model class you are using.

If you are using BaseModel class, you need to specify the following configuraitons.

- pretrained (Boolean): If the image classifier should be pretrained.
- num_ft_layers(Integer): The number of layers before output should be freed for fine-tuning
- weighted_loss(Boolean): If weighted loss function should be used
- epoch(Integer): Epoch for training for a single round of active learning training loop
- loss_function(Pytorch Loss Function<span style="color:red"> *Class*</span>): The loss function <span style="color:red">*class*</span> for training
- labeled_batch_size(Integer): Size of mini-batch for computing the gradient on labeled images
- unlabeled_batch_size(Integer): Size of mini-batch for computing the gradient on unlabeled images

If you are using LossPredictBaseModel class, you need further specify the following configurations.

- epoch_loss(Integer): During the training, beyond this number of epoch, the loss of the loss predicting model will not be used for updating the gradient of the classifier
- margin(Float): The margin used in the pair comparison loss(the crafted loss for the loss predicting model)
- lambda(Float): Weight for the loss predicting model loss in training

If you are using MEBaseModel class, you need further specify the following configurations.

- query_schedule(list of Integers): If active learning query is combined during training, this item must be provided. Query_schedule is a list of epochs(< configs['epoch']) where the training will pause to wait for new labels input and updating the model
- alpha(Float): Alpha parameter in the exponential moving average
- ramp_length(Integer):The length of the epochs that weight of loss from unlabeled samples ramps up







