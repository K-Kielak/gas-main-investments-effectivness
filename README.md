# Gas Main Investments Effectiveness
Research project looking into the effectiveness of gas main investments in Poland. 

It consists of different machine learning models that calculate the expected prices of different types of gas mains
relatively to the most standard model (see `config.py`) based on the provided training data.

### Results
You can see sample results for given data in the `results.ipynb` notebook. To be able to  fully experience the jupyter 
notebook results view it in the nbviewer. I.e. go to
https://nbviewer.jupyter.org/github/K-Kielak/gas-main-investments-effectivness/blob/master/results.ipynb


### Training models
Data to train the models should consists of data collection year in the first column followed by features specified in 
the config.py file. All features (and the year) should be in separate columns delimited by the semicolons. Each 
datapoint should be in separate row. Datapoint collection year is used to group together all of the datapoints 
from the same year and center them around the standard yearly datapoint to avoid influence of different geopolitical 
and economic factors (e.g. trade deals or inflation) that are beyond the simple gas mains data.

To configure the training process change appropriate data and models configuration in the `config.py` file.

To run the training for specified models and specified configuration simply run the `train.py` script.