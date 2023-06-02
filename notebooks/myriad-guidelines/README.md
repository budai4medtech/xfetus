# AI/ML workflow for Myriad cluster
**Author(s):** Harvey Mannering [@harveymannering](https://github.com/harveymannering) and Miguel Xochicale [@mxochicale](https://github.com/mxochicale)

## Background
"Myriad is designed for high I/O, high throughput jobs that will run within a single node rather than multi-node parallel jobs."
You need to have an UCL account to which you need to apply for a Myriad accounts the [Research Computing sign up process](https://www.rc.ucl.ac.uk/docs/Account_Services/).
See more details here https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/

## Steps to run your AI pipeline in Myriad cluster
To setup up Myriad and run your jobs just follow the following steps
0. You might not be connected by UCL network to which you need to run [Connecting to the UCL VPN with Linux](https://www.ucl.ac.uk/isd/how-to/connecting-to-ucl-vpn-linux).
1. Make sure you can log in on the command line using `ssh ucaXXXX@myriad.rc.ucl.ac.uk` where ucaXXXX is your UCL username. Then use `exit` to log out.
![fig](fig1.png)

2. Transfer the dataset and files onto Myriad cluster. Go the the path where your files and type
```
scp FETAL_PLANES_ZENODO.zip ucaXXXX@myriad.rc.ucl.ac.uk:~/Scratch/
scp loading_modules.sh trainSimpleModel.qsub.sh simple_model.py ucaXXXX@myriad.rc.ucl.ac.uk:~/Scratch/
```
2.1 Then unzip FETAL_PLANES_ZENODO.zip in the scratch directory
```
unzip FETAL_PLANES_ZENODO.zip
```

3. Log into Myriad cluster and run lines 

3.1 Load modules 
```
source loading_modules.sh #REF1  
```
3.2 Create conda virtual environments
 
3.2.1 Open a new terminal to copy virtual environment
```
cd medisynth/dependencies
scp vem.yml ucaXXXX@myriad.rc.ucl.ac.uk:~/Scratch/
```
3.2.2 Create conda env. This will take some 30 minutes to grab dependencies 
```
conda env create -f vem.yml
```

4. Submit, queue up, your job in the cluster
```
qsub trainSimpleModel.qsub.sh 
```

5. To check if your job has been correctly queued use:
```
qstat
```

NB. If nothing appears, it means there has been a problem add your job to the queue.  
But  if a table of jobs is shown, it means you jobs has been queued.  
The state column tells you if you job is running.  
If it is set to 'qw', it is waiting in the queue.  If it is set to 'r', it is running.

## Other commands
You can check both quotas on Myriad by running:
```
lquota
```


## References
* #REF1 `When you say source runit.sh, it’s like typing the module command directly into your interactive shell.`  
        `But when you say ./runit.sh, you are running a new, non-interactive shell.`
        https://unix.stackexchange.com/questions/194893/why-cant-i-load-modules-while-executing-my-bash-script-but-only-when-sourcing
