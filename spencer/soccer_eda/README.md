**Setup**
1. download the dataset from [here](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset/code)
2. find the soccer_1 directory and place the `soccer_1.sqlite` file inside of the database directory.
3. install python 3.9 if you havent already and initiate the notebook with a 3.9 virtual env
4. download [this repo](https://github.com/microsoft/Table-Pretraining) and place it within this directory
5. change directories to the downloaded repo, and hop into the `data_generator` directory
6. on line 11, comment out importlib-metadata==4.8.1. this version conflicts with the python notebook. `# importlib-metadata==4.8.1`
7. anything that hasn't been added should be able to be pip installed
...