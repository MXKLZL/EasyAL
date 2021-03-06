# Documentation

### MultiTransformDataset

*class* **MultiTransformDataset**(path_list, target_list=None, classes=None, class_name_map=None, root_dir='', transform=None)

​	A pytorch dataset subclass that can alternate the type of image transformation by setting the mode. For instance, in active learning tasks, two types of image transformation are needed for a same dataset object. The reason for it is that different transformations are needed for labeled images and unlabeled images, and unlabeled images might get labeled during executing the task. By setting the mode of transformation, you can use intended image transformation type as needed.

Arguments:

​	path_list (list of string): list of path to image files

​    target_list (list of string): list targets of samples, <span style="color:red">*the unlabeled samples should be marked as -1*</span>

​    root_dir (string): root directory to paths

​    classes (list of string): class names corresponding to targets

​    class_name_map(dictionary): mapping between targets and classes

​    transform (list of pyTorch data transformation object): list of callable transform object to be applie on a sample, using its index as the mode for the transformation. By default, 0 is the index of transformation used for training(with augmentation), 1 is the index of transformation used for predicting(without augmentation)





function set_mode(mode)

​	Set the current mode for the transformation. The mode is used as the index for the list of transformations.

​	Parameters:

​		mode(integer): index for the list of transformation





function update_target(index, new_target)

​	Update dataset with new labels(in integer form)

Parameters:

​	index(List of Integers): index of samples to update targets

​	new_target(List of Integers):	labels of newly annotated samples





*class* **TransformTwice**(transform)

​	A wrapper for pyTorch transform object so that a single image to pyTorch dataset can generate two  transformed image with same transformation process. These two transformed images could be different due to stochastic data augmentation. This kind of transformation is required for Mean Teacher model.

Arguments:

​	transform(pyTorch data transformation object): Callable transform object to be applie on a sample twice



*callable function* \__call__()

Returns

​	Two images generated with the transform process within on the input image	







### BaseModel 

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







### LossPredictBaseModel	

*class* **LossPredictBaseModel**(dataset, model_name, configs)

​	The subclass of BaseModel to be used for loss sampling and confident coreset strategies.

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet'].

​	configs(dictionary): configuration for the current active learning task. See configruation guide for details.





function predict_loss(data_loader)

​	get predicted classification loss for each image of a given data loader

Returns:

​	A tensor of predicted loss





function  predict_unlabeled_loss()

​	get predicted classification loss for each unlabeled image

Returns:

​	A tensor of predicted loss





### MEBaseModel

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







### TEBaseModel 

*class* **TEBaseModel**(dataset, model_name, configs, test_ds = None, weight = True, test_mode = False )

​	The subclass of BaseModel to be used for Temporal Ensembling Semi-Supervised Learning

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet'].

​	configs(dictionary): configuration for the Semi-Supervised Learning task. See configruation guide for details.

​	test_ds(dataset): get accuracy of current model on given dataset after each epoch of the training loop.

​	weight(boolean): If the unsupervised loss should be used during the back-propagation. If weight = False, there is not weight for unlabelled loss and the tranining become pure supervised learning task. Default value is True.

​	test_mode(boolean): additionally get the accuracry of current model on unlabeled data.





function pred_acc(testloader, test_target, criterion='f1')

​	Calculate test accuracy on given test set

Parameters:

​	testloader(pyTorch Dataloader): Dataloader of test set to calculate accuracy on

​    test_target(list of Integers): List of targets correponding to given test set

​	criterion(String): Type of evaluation metric to compute. Can be chosen from ['precision','recall','f1'] Default 'f1'

Returns:

​	Evaluation metric calculated on predictions from given test set





### NoisyStudentBaseModel

*class* **NoisyStudentBaseModel**(dataset, model_name, configs, student_list = None)

​	The subclass of BaseModel to be used for Noisy Student Semi-Supervised Learning

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet'].

​	configs(dictionary): configuration for the Semi-Supervised Learning task. See configruation guide for details.

​	student_list(list of string): List of student model's name for the noisy student training loop.







### Query Strategy

function query(strategy, model_class, label_per_round, alpha=0.5, add_uncertainty=False, distance_name='euclidean', standardize=True)

​	A function from query_strategy.py used to query the indexs of images for human annotation during active learning loop.

Parameters:

​	strategy (string): name of the query strategy you want to use, Can be chosen from['random', 'uncertain', 'margin', 'entropy', 'k_means', 'k_means++', 'k_center_greedy', 'confident_coreset'].

​	model_class (BaseModel Object or the subclass of BaseModel Object): model class of active learning loop. Can be chosen from `BaseModel()`, `LossPredictBaseModel()`, `TEBaseModel()`, `NoisyStudentBaseModel()`, `MEBaseModel()`. If strategy equal to `loss`, you can only pass a `LossPredictBaseModel()` object to model class.

​	label_per_round (integer): number of images you want to query

​	add_uncertainty(string): the type of uncertainty measure you want to add into the distance query strategy. Can be chosen from['uncertain', 'margin', 'entropy']. For all distance based queries except confident coreset, you can add the uncertainty into the consideration to make a your query strategy consider both distances and uncertainties of images. For example, `strategy = k_means, add_uncertainty = 'uncertain' `will take the distance measure and uncertainty measure in the account when searching for centroid image. For `strategy = confident_coreset`, please leave `add_uncertainty = None` and it will automatically add `loss` in the distance calculation by definition in this [paper](https://arxiv.org/pdf/2004.02200.pdf) 

​	alpha (float): alpha denotes an weighting hyperparameter to balance between the distribution-based score and the uncertainty based score when you combine both distance-based query and uncertainty based query

​	distance_name(string): the distance function you used to calculate the distance between a pair of images in embedding space, Can be chosen from['cosine','euclidean']

​	standardize(boolean): If every dimension of image's embedding should be normalized before the distance calculation. Set it equal to true to aviod the scale mismatch between the dimensions of embedding space.

Returns: 

​	A tuple`(Float, Numpy Array)`with the number of second to finnish the query in index 0 and a numpy array of integer contained the selected indexs of images for human annotation in index 1.







### ModelConstructor

function get_model(dataset,model_name,train_configs,model_type = 'Basic',test_ds=None) 

​	A function from ModelConstructor.py used to build an appropriate model class based on your need

Parameters:

​	dataset(MultiTransformDataset object): The dataset for active learning loop or semi-supervised learning task.

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet']

​	model_type(string): name of the algorithm you want to use. Can be chosen from['Basic', 'Loss', 'MeanTeacher', 'TemporalEnsembling', 'NoisyStudent' ]. 'MeanTeacher', 'TemporalEnsembling', 'NoisyStudent' should be chosen for three semi-supervised learning algorithm and 'Basic' should be chosen for all active learning loop except loss learning algorithm which shoud use 'Loss' option

​	test_ds(dataset): if you choose `MeanTeacher` or `TemporalEnsembling` as model_type, you can pass a test dataset to get accuracy of current model on that after each epoch of the training loop. 



Returns:

​	An appropriate model class object used for your active learning or semi-supervised learning task







### Utils

function get_mapping(ds)

​	Get the two-way mappings between index and name of the training data stored

Parameters:

​	ds (MultiTransformDataset object): The dataset to obtain the mappings

Returns:

​	index to base name mapping (list) and base name to index mapping (dict)

​	



function read_from_oracle(completion_path, idx2base, base2idx)

​	Read the label from the output file of label studio, return the index and labeled class as two lists

Parameters:

​	completion_path (str): The path of the JSON output from label studio	

​	idx2base (list) - The index to base name mapping

​	base2idx (dict) - The base name to index mapping

Returns:

​	index and labeled class as two lists





function update_json(task_json_path, query_indices, idx2base, base2idx, model, ds, class_name_map)

​	Update the task JSON of label studo so the images queried by active learning algorithms can be labeled
with label studio

Parameters: 

​	task_json_path (str): The path of task JSON path of label studio

​    query_indeices (list):  The indices of training data queried by active learning algorithm

​	idx2base (list): The index to base name mapping

​	base2idx (dict): The base name to index mapping

​	model (Model object):  The current Model

​	ds (MultiTransformDataset object): The dataset to obtain the mappings

​	class_name_map (dict): mapping between targets and classes

