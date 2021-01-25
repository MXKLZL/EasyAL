<p align="center">
  <img src="logo.png" width="300" height="300">
</p>

# Getting Started

### EasyAL

A modular active learning package for image classification with state-of-the-art active learning strategies

### Requirements
 - Dependency file [pyproject.toml](./pyproject.toml)

### Installation
- Install label studio with command `$pip install label-studio`
- Run command line `$cd [the path you pull EasyAL package folder to]/EasyAL`
- Run command line `$poetry shell` and then `$poetry install` to install all dependencies for EasyAL

### Quick Start
Guide to create label studio work folder and start active learning loop:


- Manually create a configuration file, see example in [example/config.xml](./example/config.xml)

- Have a folder with images for annotation

- Run command line  `$label-studio init [your label studio project name] --input-path=[path of image folder] --input-format=image-dir --label-config=[path of the config file/config.xml] --allow-serving-local-files`

- A folder with the following structure will appear in your current path 

   ![img](./example/label_studio_work_folder.png)

- Run `label-studio start ./[your label studio project name]` and your browser with launch automatically with the label studio interface

- Choose the labeling button on the top of the interface to begin labeling

- After your first round of labeling, change environment variable  "sampling": "sequential" in `./[your label studio project name]/.config.json` to "sampling": "prediction-score-max" to activate active learning mode for future rounds

- Use EasyAL package to train your image classification model and query images to label with active learning strategies. See example in [example/ALRun.py](./example/ALRun.py)

- Using the functions in utils.py to parse JSON files in [your label studio project name]/completions folder and update the labels of images in your self-create dataset. See example in [example/ALRun.py](./example/ALRun.py)

- Visit https://labelstud.io/ for any question related to label studio



# Documentation



#### **MultiTransformDataset**

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



#### TEBaseModel 

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



#### NoisyStudentBaseModel

*class* **NoisyStudentBaseModel**(dataset, model_name, configs, student_list = None)

​	The subclass of BaseModel to be used for Noisy Student Semi-Supervised Learning

Arguments:

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'mobilenet'].

​	configs(dictionary): configuration for the Semi-Supervised Learning task. See configruation guide for details.

​	student_list(list of string): List of student model's name for the noisy student training loop.





#### Query Strategy

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



​	

#### Others

function get_model(dataset,model_name,train_configs,model_type = 'Basic',test_ds=None) 

​	A function from ModelConstructor.py used to build an appropriate model class based on your need

Parameters:

​	dataset(MultiTransformDataset object): The dataset for active learning loop or semi-supervised learning task.

​	model_name(string): name of the image classifier you want to use. Can be chosen from ['resnet18', 'resnet34', 'resnet50', 'mobilenet']

​	model_type(string): name of the algorithm you want to use. Can be chosen from['Basic', 'Loss', 'MeanTeacher', 'TemporalEnsembling', 'NoisyStudent' ]. 'MeanTeacher', 'TemporalEnsembling', 'NoisyStudent' should be chosen for three semi-supervised learning algorithm and 'Basic' should be chosen for all active learning loop except loss learning algorithm which shoud use 'Loss' option

​	test_ds(dataset): if you choose `MeanTeacher` or `TemporalEnsembling` as model_type, you can pass a test dataset to get accuracy of current model on that after each epoch of the training loop. 



Returns:

​	An appropriate model class object used for your active learning or semi-supervised learning task

​	



### Configuration Guide

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



### Data Transformation Guide

When you create **MultiTransformDataset()** object, you need pass a list transformation to deal with the different scenarios during the active learning loop.

- Parameter transforms (list of transformation) is list of callable transform object to be applie on a sample, using its index as the mode for the transformation. By default, 0 is the index of transformation used for training(with augmentation), 1 is the index of transformation used for predicting(without augmentation).
- If `MEBaseModel` is used, `TransformTwice` object need to be appended to the `tranformations` list as the last item. The input for TransformTwice wrapper should be the transformation used for training, that is, identical to the first transformation in the list.
- For the NoisyStudent Algorithm, we recommend to add transformation of Brightness, Contrast, Sharpness during the training step to help student model outperform their teacher. For more information, refer to [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)





# Reference

- [D. Lewis and W. Gale. A sequential algorithm for training text classifiers. in proceedings of
the acm sigir conference on research and development in information retrieval, pages 3–12.
acm/springer, 1994.](https://arxiv.org/abs/cmp-lg/9407020)

- [Tobias Scheffer, Christian Decomain, and Stefan Wrobel. Active hidden markov models for
information extraction, 2001.](https://link.springer.com/chapter/10.1007/3-540-44816-0_31)

- [Donggeun Yoo and In So Kweon. Learning loss for active learning, 2019.](https://arxiv.org/abs/1905.03677)

- [Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set
approach, 2018.](https://arxiv.org/abs/1708.00489)

- [Seong Tae Kim, Farrukh Mushtaq, and Nassir Navab. Confident coreset for active learning in
medical image analysis, 04 2020.](https://arxiv.org/abs/2004.02200)

- [Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student
improves imagenet classification, 2020.](https://arxiv.org/abs/1911.04252)

- [Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning, 2017.](https://arxiv.org/abs/1610.02242)

- [Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged
consistency targets improve semi-supervised deep learning results, 2018.](https://arxiv.org/abs/1703.01780)








