# Learn Oculomotor Behaviors from Scanpath



- Beibin Li
- [Talk Video](https://drive.google.com/file/d/1y2wAdpdjxQXGKGIBLtu0ibfcz5iWjG3A/view?usp=sharing)
- [Slides](https://drive.google.com/file/d/1lR8he5IntXxh75oIfY0unZaSXBiakL7i/view?usp=sharing)
- Please contact [beibin@cs.washington.edu](beibin@cs.washington.edu) if you have any questions regarding to the research or code.

Feel free to cite the following article if you found this repo or research is helpful. Thanks!

Beibin Li, Nicholas Nuechterlein, Erin Barney, Claire Foster, Minah Kim, Monique Mahony, Adham Atyabi, Li Feng, Quan Wang, Pamela Ventola, Linda Shapiro, and Frederick Shic. 2021. Learning Oculomotor Behaviors from Scanpath. In Proceedings of the 2021 International Conference on Multimodal Interaction (ICMI '21), October 18–22, 2021, Montréal, QC, Canada. ACM, New York, NY, USA 9 Pages. https://doi.org/10.1145/3462244.3479923


```
@article{Li2021Oculomotor,
  title={Learning Oculomotor Behaviors from Scanpath},
  author={Li, Beibin and Nuechterlein, Nicholas and Barney, Erin and Foster, Claire and Kim, Minah and Mahony, Monique and Atyabi, Adham and Feng, Li and Wang, Quan and Ventola, Pamela and others},
  booktitle={Proceedings of the 2021 International Conference on Multimodal Interaction (ICMI '21), October 18--22, 2021, Montréal, QC, Canada},
  organization={ACM}  
  year={2021}
  url={https://doi.org/10.1145/3462244.3479923}
}
```



## Oculomotor Behavior Framework (OBF)

### Code Organization


#### OBF folder
All training code are inside the [obf/](obf/) folder, where the models are stored in the [obf/model/ae.py](obf/model/ae.py) file, and 

-  [obf/](obf/)
   -   [obf/dataloader](obf/dataloader): data loading functions and modules.
   -   [obf/execution](obf/execution): execution code for training.
   -   [obf/model](obf/model): deep learning model architecture.
   -   [obf/utils](obf/utils): utility for training.

#### sampled data
We provided some sampled data and data pre-processing code to the [sample_data/](sample_data/) folder. 

Please download public eye-tracking datasets, and then use the provided Python scripts to process the signals.


#### Configs and JSON setting
We control the experiment setting with JSON files. Some sample configurations are in the [config/](config/) folder.

- JSON setting
  - "pretrain data setting": defines where to load pre-training unlabelled data from disk.
    - "datasets": contains the path and signal length (in miliseconds) for input data.
    - "batch size": an integer to define mini-batch size.
  - "fixation identification setting": contains information for the I-VT algorithm
  - "experiment": for the pre-training process.

### How to Use
Please check the [Pretrain.ipynb](Pretrain.ipynb) Jupyter Notebook file to see examples on how to train a new model from scratch. Or, you can use the provided model directly for your downstream application, as shown below.

## Downstream Applications

Please check the [stim_prediction.ipynb](stim_prediction.ipynb) Jupyter Notebook file to see a downstream application example. In this application, we use the pre-trained model, and fine-tune it for the MIT1003 dataset to predict which stimulus a subject was watching.

