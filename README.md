<h1 style="text-align: center;">WildCLIP</h1>
<h3 style="text-align: center;">Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models.</h3>


<center>
    <img src="https://github.com/amathislab/wildclip/blob/main/resources/overview.png" alt="Overview">
</center>

<h3 style="text-align: center;">Code coming soon</h3>

## Reference

If you find this code or ideas presented in our work useful, please cite:

[WildCLIP: Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models](https://www.biorxiv.org/content/10.1101/2023.12.22.572990v1) 
by  Valentin Gabeff,  Marc Russwurm,  Devis Tuia and Alexander Mathis


```
@article{Gabeff2023WildClip,
	author = {Valentin Gabeff and Marc Russwurm and Devis Tuia and Alexander Mathis},
	title = {WildCLIP: Scene and animal attribute retrieval from camera trap data with domain-adapted vision-language models},
	elocation-id = {2023.12.22.572990},
	year = {2023},
	doi = {10.1101/2023.12.22.572990},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Wildlife observation with camera traps has great potential for ethology and ecology, as it gathers data non-invasively in an automated way. However, camera traps produce large amounts of uncurated data, which is time-consuming to annotate. Existing methods to label these data automatically commonly use a fixed pre-defined set of distinctive classes and require many labeled examples per class to be trained. Moreover, the attributes of interest are sometimes rare and difficult to find in large data collections. Large pretrained vision-language models, such as Contrastive Language Image Pretraining (CLIP), offer great promises to facilitate the annotation process of camera-trap data. Images can be described with greater detail, the set of classes is not fixed and can be extensible on demand and pretrained models can help to retrieve rare samples. In this work, we explore the potential of CLIP to retrieve images according to environmental and ecological attributes. We create WildCLIP by fine-tuning CLIP on wildlife camera-trap images and to further increase its flexibility, we add an adapter module to better expand to novel attributes in a few-shot manner. We quantify WildCLIP{\textquoteright}s performance and show that it can retrieve novel attributes in the Snapshot Serengeti dataset. Our findings outline new opportunities to facilitate annotation processes with complex and multi-attribute captions. The code will be made available at https://github.com/amathislab/wildclip.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/12/23/2023.12.22.572990},
	eprint = {https://www.biorxiv.org/content/early/2023/12/23/2023.12.22.572990.full.pdf},
	journal = {bioRxiv}
}
```
