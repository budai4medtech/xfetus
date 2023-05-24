<h1 align="center">
Good practices in AI/ML for Ultrasound Fetal Brain Imaging Synthesis
</h1>
<div align="center">

Harvey Mannering, Sofia Miñano, and Miguel Xochicale    

University College London    
The deep learning and computer vision Journal Club     
UCL Centre for Advance Research Computing     
1st of June 2023, 15:00 GMT
</div>

## Abstract
Medical image datasets for AI and ML methods must be diverse to generalise well unseen data (i.e. diagnoses, diseases, pathologies, scanners, demographics, etc).
However there are few public ultrasound fetal imaging datasets due to insufficient amounts of clinical data, patient privacy, rare occurrence of abnormalities, and limited experts for data collection and validation.
To address such challenges in Ultrasound Medical Imaging, Miguel will discuss two proposed generative adversarial networks (GAN)-based models: diffusion-super-resolution-GAN and transformer-based-GAN, to synthesise images of fetal Ultrasound brain image planes from one public dataset.
Similarly, Miguel will present and discuss AI and ML workflow aligned to good ML practices by FDA, and methods for quality image assessment (e.g., visual Turing test and FID scores).
Finally, a simple prototype in GitHub, google-colabs and guidelines to train it using Myriad cluster will be presented as well as applications for Medical Image Synthesis e.g., classification, augmentation, segmentation, registration and other downstream tasks, etc. will be discussed.
The resources to reproduce the work of this talk are available at https://github.com/mxochicale/medisynth.


## Clone repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (or [here](https://github.com/mxochicale/tools/blob/main/github/SSH.md))
* Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine:
```
cd && mkdir -p $HOME/repositories/mxochicale && cd  $HOME/repositories/mxochicale
git clone git@github.com:mxochicale/medisynth.git
```


## Contributors
Thanks goes to all these people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):  
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<!-- ADD GITHUB USERNAME AND HASH FOR GITHUB PHOTO -->
		<a href="https://github.com/???"><img src="https://avatars1.githubusercontent.com/u/23114020?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>ADD NAME SURNAME</b> </sub>        
		</a>
		<br />
			<!-- ADD GITHUB REPOSITORY AND PROJECT, TITLE AND EMOJIS -->
			<a href="https://github.com/$PROJECTNAME/$REPOSITORY_NAME/commits?author=" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/sfmig"><img src="https://avatars1.githubusercontent.com/u/33267254?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Sofia Miñano</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/mxochicale/medisynth/commits?author=sfmig" title="Code">💻</a> 
			<a href="https://github.com/mxochicale/medisynth/commits?author=sfmig" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/harveymannering"><img src="https://avatars1.githubusercontent.com/u/60523103?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Harvey Mannering</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/mxochicale/medisynth/commits?author=harveymannering" title="Code">💻</a> 
			<a href="https://github.com/mxochicale/medisynth/commits?author=harveymannering" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/mxochicale"><img src="https://avatars1.githubusercontent.com/u/11370681?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>Miguel Xochicale</b></sub>          
			<br />
		</a>
			<a href="https://github.com/mxochicale/medisynth/commits?author=mxochicale" title="Code">💻</a> 
			<a href="ttps://github.com/mxochicale/medisynth/commits?author=mxochicale" title="Documentation">📖  🔧 </a>
	</td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This work follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.  
Contributions of any kind welcome!
