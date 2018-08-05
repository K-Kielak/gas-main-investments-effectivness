# Gas Main Investments Effectivness
Research project looking into the effectivness of gas main investments in Poland. 

### Training models
To configure the training process change appropiate data and models configuration in the `config.py` file. Additionally,
change beginning of the `main.py` file to add/remove/modify the solvable and non-solvable models that should run.
In case of modifying the solvable models list make sure to also modify appropriate feature sets in the 
`data_for_solvables` list.

To run the training for specified models and specified configuration simply run the `main.py` script.