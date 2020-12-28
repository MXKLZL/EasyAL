# EasyAL
Modular Active Learning Package for Image classification with state of the art active learning strategy

# Requirement
 [requirement](./pyprojtect.toml)

# Installation
- install label studio with command $pip install label-studio
- run command line $poetry shell and then $poetry install to install all dependencies for our package

# Getting Strated
Guide to create label studio work folder and start active learning loop:


3. manually create a configuration file, config.xml(see example folder)
4. Have a folder with images for annotation
5. run command line  $label-studio init [your label studio project name] --input-path=[path of image folder] --input-format=image-dir --label-config=[path of the config file/config.xml] --allow-serving-local-files
6. A folder with the following structure will appear in your current path 

   ![img](./example/label_studio_work_folder.png)

7. Run label-studio start ./[your label studio project name] and your browser with launch automatically with the label studio interface
8. Choose the labeling button on the top of the interface to begin labeling
9. After your first round of labeling, change environment variable  "sampling": "sequential" in ./[your label studio project name]/.config.json to "sampling": "prediction-score-max" to activate active learning mode for future rounds
10. using the functions in utils.py to parse JSON files in [your label studio project name]/completions folder and update the labels of images in your self-create dataset(see example folder for an active learning loop)
11. Visit https://labelstud.io/ for any question related to label studio

# Reference


