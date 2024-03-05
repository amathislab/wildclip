[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

<h1 style="text-align: center;">Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models.</h1>

<h2>Overview</h2>

WildCLIP is an adapted CLIP model that allows to retrieve camera-trap events with natural language from the Snapshot Serengeti dataset. Our work was selected as an oral at CVPR CV4animal in Vancouver in June 2023. An extended version is available in [biorxiv](https://www.biorxiv.org/content/10.1101/2023.12.22.572990v1).

This project intends to demonstrate how vision-language models may assist the annotation process of camera-trap datasets. 

<p align="center">
    <img src="https://github.com/amathislab/wildclip/blob/main/resources/overview.png" width="600" alt="Overview">
</p>


We actively seek to expand the training set. **Reach out if you want to collaborate (see information below)!** 

This repository is currently primarily intended at machine-learning practitioners that wish to test, understand and contribute to WildCLIP.

If you find this code or ideas presented in our work useful, please cite:

[WildCLIP: Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models](https://www.biorxiv.org/content/10.1101/2023.12.22.572990v1) 
by  Valentin Gabeff,  Marc Russwurm,  Devis Tuia and Alexander Mathis

<h2>Code usage (setup, inference and training)</h2>

Interested in trying out? 

First install the necessary dependencies and download the models/data. 

<h3>Code requirements (installation) </h3>

Required python packages are listed in the requirements.yml which can be used to build a conda environment.

```
conda env create --file environment.yml
conda activate wildclip
pip install clip@git+https://github.com/openai/CLIP.git
```

<h3>Data requirements</h3>


For trying out the WildCLIP, please download the model and the data available [here](https://zenodo.org/records/10479317).

First clone this repository, then copy the annotation files in the folder `/data`, and the images in the folder /dataset. (These paths can freely be changed in the command line and configuration files, respectively).

<h3>Model inference</h3>

From the command line in the src folder:

```
python eval_clip.py \
-F config_files/wildclip_vitb16_t1.yml \
-I ../data/test_dataset_crops_single_animal_template_captions_T1T8T10.csv \
-C captions/serengeti_test_queries_T1T8T10_nattr_1.txt \
-M ../results/wildclip_vitb16_t1/wildclip_vitb16_t1_last_ckpt.pth
```

<h3>Model training</h3>

To fine-tune CLIP with your own data:
```
python fine_tune_clip.py \
-F config_files/wildclip_vitb16_t1.yml \
-I ../data/train_dataset.csv \
-V ../data/val_dataset.csv
```

To fine-tune CLIP with the VR-LwF loss:
```
python fine_tune_clip.py \
-F config_files/wildclip_vitb16_t1_lwf.yml \
-I ../data/train_dataset.csv \
-V ../data/val_dataset.csv \
--lwf_loss \
--path_replayed_vocabulary captions/serengeti_anchors.txt
```

To fine-tune CLIP-adapter with few-shots from a pretrained CLIP model:
```
python fine_tune_clip.py \
-F config_files/wildclip_vitb16_t1_fs.yml \
-I ../data/few_shots_dataset.csv \
-M ../results/wildclip_vitb16_t1/wildclip_vitb16_t1_last_ckpt.pth \
--few_shots \
-K 1 \
--override_adapter
```

<h2>Results</h2>

_Qualitative performance of WildCLIP in comparison to CLIP on the Snapshot Serengeti test set_

<p align="left">
    <img src="https://github.com/amathislab/wildclip/blob/main/resources/quali_combinations.png" width="700" alt="Overview">
</p>

More examples can be found in the Appendix of the associated publication.

Results can be reproduced in the ```src/notebooks/ResultsEvaluation.ipynb``` notebook.

<h2>Data sources</h2>

The complete Snapshot Serengeti dataset is available on [lila.science](https://lila.science/datasets/snapshot-serengeti).

Their data set is released under the [Community Data License Agreement (permissive variant)](https://cdla.dev/permissive-1-0/).

The data subset provided corresponds to the test data of Snapshot Serengeti containing single animals only and cropped according to MegaDetector output (MDv4 at the time of the study).

<h2>Testing on your data</h2>

We also provide code to easily test the original CLIP model given a folder of images and a list of text queries.

This will output a prediction CSV file of cosine similarities between each image and the provided queries where the results can be visualized with ```src/notebooks/PredictionVisualization.ipynb```:
```
python predict_clip.py -I /path/to/image_folder -C path/to/queries.txt -O /path/to/output_folder --zero-shot-clip
```

Since ```src/eval_clip.py``` requires true labels of the test images to run, ```src/predict_clip.py``` can also be used to run predictions with a fine-tuned wildclip model on a new set of images:
```
python predict_clip.py -I /path/to/image_folder -C path/to/queries.txt -O /path/to/output_folder -M wildclip_vitb16_t1_lwf_last_ckpt.pth
```

<h2>Future directions</h2>

This project proposes the development of ecology-specific vision-language models to facilitate the annotation of wildlife data. Our work showed a proof a principle.

In principle, to be more usable to the ecological community, WildCLIP will benefit from multiple improvements:
- Extend WildCLIP to more geographical regions, species and behaviors. Currently, there are not many camera trap datasets with behavioral attributes. Reach out, if you want to collaborate
- Improve fine-tuning of WildCLIP from few images and captions to adapt to a task of interest.
- Improve open-vocabulary capabilities of WildCLIP.
- Integrate WildCLIP and adaptation functions in a graphical interface to facilate annotation process of camera trap datasets.

<h2>Contributing</h2>

If you are interested in contributing to one of the aforementioned points, or work on a similar project and wish to collaborate, please reach out to [ECEO](https://www.epfl.ch/labs/eceo) or to the [Mathis Group](https://www.mathislab.org) at EPFL.

For code related contributions, suggestions or inquires, please open a github issue. Code is still under active development.

If you use this code in your research, please consider citing us:

```
@article{gabeff2023wildclip,
  title={WildCLIP: Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models},
  author={Gabeff, Valentin and Russwurm, Marc and Tuia, Devis and Mathis, Alexander},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

<h2>Code acknowledgments</h2>

We acknowledge the following code repositories that helped to create WildCLIP:  

https://github.com/openai/CLIP  
https://github.com/mlfoundations/open_clip  
https://github.com/gaopengcuhk/CLIP-Adapter/  
https://github.com/locuslab/FLYP/  

and the following article for the WildCLIP-LwF variant:
https://arxiv.org/pdf/2207.09248.pdf

Thank you! Sources are mentioned in the relevant code sections. 
