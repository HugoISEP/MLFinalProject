# Machine learning final project

## Setup
Download [the dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification?fbclid=IwAR00Cf-LRtyxvbPzAy119Q0YL0mlsDNeCqYTKjUBuu4jQx5MF1aQdSxUFVI) and put it at the root of the project

You should have that files disposition in the project
```shell
.
├── Data
│   ├── features_30_sec.csv
│   ├── features_3_sec.csv
│   ├── genres_original
│   ├── images_original
...
```

**Important:** The file *Data/genres_original/jazz/jazz.00054.wav* is corrupted, you must remove it before running the script

Install all dependencies:  
```pip install -r requirements.txt```

## Sources
- https://www.tensorflow.org/tutorials/load_data/images