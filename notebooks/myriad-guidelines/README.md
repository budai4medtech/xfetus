# AI/ML workflow for Myriad cluster
**Author(s):** Harvey Mannering [@harveymannering](https://github.com/harveymannering) and Miguel Xochicale [@mxochicale](https://github.com/mxochicale)

## Background
"Myriad is designed for high I/O, high throughput jobs that will run within a single node rather than multi-node parallel jobs."
You need to have an UCL account to which you need to apply for a Myriad accounts the [Research Computing sign up process](https://www.rc.ucl.ac.uk/docs/Account_Services/).
See more details here https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/

## Steps to run your AI pipeline in Myriad cluster
To setup up Myriad and run your jobs just follow the following steps

1. Make sure you can log in on the command line using `ssh ucaXXXX@myriad.rc.ucl.ac.uk` where ucXXXX is your UCL username. Then use `exit` to log out.
2. Transfer the dataset onto Myriad using
```
scp FETAL_PLANES_ZENODO.zip ucaXXXX@myriad.rc.ucl.ac.uk:~/Scratch/
```
2.1 Then unzip FETAL_PLANES_ZENODO.zip in the scratch directory
```
unzip FETAL_PLANES_ZENODO.zip
```

3. Log into Myriad cluster and run lines 

3.1 Load modules 
```
sh loading_modules.sh
```
3.2 Create conda virtual environments 
```
sh creating_conda_virtual_environment.sh
```


4. Copy the `trainSimpleModel.qsub.sh` and `simple_model.py` files into that same directory and then run to queue up a job.
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