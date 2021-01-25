# Getting Started

#### EasyAL

A modular active learning package for image classification with state-of-the-art active learning strategies

#### Requirements
 - Dependency file [pyproject.toml](../pyproject.toml)

#### Installation
- Install label studio with command `$pip install label-studio`
- Run command line `$cd [the path you pull EasyAL package folder to]/EasyAL`
- Run command line `$poetry shell` and then `$poetry install` to install all dependencies for EasyAL

#### Quick Start
Guide to create label studio work folder and start active learning loop:


- Manually create a configuration file, see example in [example/config.xml](../example/config.xml)

- Have a folder with images for annotation

- Run command line  `$label-studio init [your label studio project name] --input-path=[path of image folder] --input-format=image-dir --label-config=[path of the config file/config.xml] --allow-serving-local-files`

- A folder with the following structure will appear in your current path 

   ![img](./example/label_studio_work_folder.png)

- Run `label-studio start ./[your label studio project name]` and your browser with launch automatically with the label studio interface

- Choose the labeling button on the top of the interface to begin labeling

- After your first round of labeling, change environment variable  "sampling": "sequential" in `./[your label studio project name]/.config.json` to "sampling": "prediction-score-max" to activate active learning mode for future rounds

- Use EasyAL package to train your image classification model and query images to label with active learning strategies. See example in [example/ALRun.py](../example/ALRun.py)

- Using the functions in utils.py to parse JSON files in [your label studio project name]/completions folder and update the labels of images in your self-create dataset. See example in [example/ALRun.py](../example/ALRun.py)

- Visit https://labelstud.io/ for any question related to label studio




