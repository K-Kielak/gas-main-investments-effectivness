# Gas Main Investments Effectivness
Research project looking into the effectivness of gas main investments in Poland. 

It consists of different machine learning models that calculate the expected prices of different types of gas mains
relatively to the most standard model (see `config.py`) based on the provided training data.

### Training models
Data to train the models should consists of data collection year in the first column followed by features specified in 
the config.py file. All features (and the year) should be in separate columns delimited by the semicolons. Each 
datapoint should be in separate row. Datapoint collection year is used to group together all of the datapoints 
from the same year and center them around the standard yearly datapoint to avoid influence of different geopolitical 
and economic factors (e.g. trade deals or inflation) that are beyond the simple gas mains data.

To configure the training process change appropiate data and models configuration in the `config.py` file. Additionally,
change beginning of the `main.py` file to add/remove/modify the solvable and non-solvable models that should run.

To run the training for specified models and specified configuration simply run the `main.py` script.