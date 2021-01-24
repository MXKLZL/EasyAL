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



​	

​	

​	

 





Configuration Guide:

Different configuration is needed for different model class you are using.

If you are using BaseModel class, you need to specify the following configuraitons.

- pretrained (Boolean): If the image classifier should be pretrained.
- num_ft_layers(Integer): The number of layers before output should be freed for fine-tuning
- weighted_loss(Boolean): If weighted loss function should be used
- epoch(Integer): Epoch for training for a single round of active learning training loop
- loss_function(Pytorch Loss Function<span style="color:red"> *Class*</span>): The loss function <span style="color:red">*class*</span> for training
- labeled_batch_size(Integer): Size of mini-batch for computing the gradient on labeled images
- unlabeled_batch_size(Integer): Size of mini-batch for computing the gradient on unlabeled images





