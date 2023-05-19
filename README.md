<h1 align="center">:page_facing_up: 
Good practices in AI/ML for Medical Image Synthesis
</h1>
<div align="center">

Harvey Mannering, Sofia Miñano, and Miguel Xochicale    


University College London    
The deep learning and computer vision Journal Club     
UCL Centre for Advance Research Computing     
31st of May 2023
</div>

## Abstract
Medical Image Synthesis has been making great progress since the publication of generative models and the most recent diffusion models.
In this talk, I will provide background on applications for medical image synthesis e.g. classification, augmentation, segmentration, registration and other downstreams tasks, etc.
Similarly, I will overview the balance between GANS, VANs and Diffusion models and cover implemenation workflows following good practices and FDA guidelines, aiming to provide understanding of essentials to train reliable, repeatable, reproducible and validted models for medical image synthesis.
Particularly, I will discuss an ML workflow for fetal brain ultrasound image sysntehsis, and its quality image assemsment (visual turing test and FID scores).
Finally, I will present a quick prototype in github, google-colabs and guidelines to train it using myriam server.

## Tutorial
* Google colabs 
* Quick guidelines and demos for myriad
	* How to run and re-train in myriad


## Clone repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) (or [here](https://github.com/mxochicale/tools/blob/main/github/SSH.md))
* Clone the repository by typing (or copying) the following line in a terminal at your selected path in your machine:
```
cd && mkdir -p $HOME/repositories/mxochicale && cd  $HOME/repositories/mxochicale
git clone git@github.com:mxochicale/prototyping-pipelines-for-medical-image-synthesis.git
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
			<a href="https://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=sfmig" title="Code">💻</a> 
			<a href="https://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=sfmig" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/harveymannering"><img src="https://avatars1.githubusercontent.com/u/60523103?v=4?s=100" width="100px;" alt=""/>
		<br />
			<sub> <b>Harvey Mannering</b> </sub>        
		</a>
		<br />
			<a href="https://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=harveymannering" title="Code">💻</a> 
			<a href="https://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=harveymannering" title="Research">  🔬 🤔  </a>
	</td>
	<!-- CONTRIBUTOR -->
	<td align="center">
		<a href="https://github.com/mxochicale"><img src="https://avatars1.githubusercontent.com/u/11370681?v=4?s=100" width="100px;" alt=""/>
			<br />
			<sub><b>Miguel Xochicale</b></sub>          
			<br />
		</a>
			<a href="https://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=mxochicale" title="Code">💻</a> 
			<a href="ttps://github.com/mxochicale/prototyping-pipelines-for-medical-image-synthesis/commits?author=mxochicale" title="Documentation">📖  🔧 </a>
	</td>
  </tr>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This work follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.  
Contributions of any kind welcome!
