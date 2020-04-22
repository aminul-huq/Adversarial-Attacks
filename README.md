# Adversarial-Attacks 

Adversarial Attacks testing on DNN models
Work in Progress

Target : Will contain several adversarial attack techniques, perform these attacks on both standard trained models and adversarial trained modes. Both standard and adversarial attacks will be implemented here.

# Attacks Included 

1. FGSM
2. MIFGSM
3. PGD
4. DeepFool

# Usage 

Clone the repository
and 
```
bash run.sh
```

# Arguments

| Command        | Default_value        | 
| -------------  |:--------------------:| 
| --attack       | 0                    | 
| --lr           | 0.001                |  
| --itr          | 5                    |   
| --eps      | 8./255.                 |
| --step_size   | 7                 |  
| --momentum       | 0.9                | 
| --overshoot    | 0.2                |
| --preprocess  | 0                   |  
