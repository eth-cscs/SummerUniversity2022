
# CSCS-USI HPC/Data Analytics Summer University 2022

This repository contains the materials used in the Summer University, including source code, lecture notes and slides.
Material will be added to the repository throughout the course, which will require that students either update their copy of the repository, or download/checkout a new copy of the repository.

## Announcements

We will be using Slack to post news and links relevant to the event: you should have received an invitation to join the Summer University Slack workspace.

## Schedule

### Week 1
![SU-Week1-v1](https://user-images.githubusercontent.com/4578156/176634095-59a122be-5225-42dd-bb61-2cfc923cf7d9.png)

### Week 2

![SU-Week2-v2](https://user-images.githubusercontent.com/4578156/176634104-1bf2d2bf-b8a2-4d59-81de-82ae750d038d.png)

## Link to materials

- CUDA (Day 1, 2 & 3)
- OpenACC (Day 3 & 4)
- ISO C++ (Day 4)
- [Kokkos](https://github.com/eth-cscs/SummerUniversity2022/tree/main/kokkos) (Day 5)
- Interactive Supercomputing (Day 6)
- Python HPC (Day 6)
- Introduction to Deep Learning (Day 7 & 8)
- Introduction to Machine Learning and Rapids (Day 9)

## Obtaining a copy of this repository

### On your own computer

You will want to download the repository to your laptop to get all of the slides.
The best method is to use git, so that you can update the repository as more slides and material are added over the course of the event.
So, if you have git installed, use the same method as for Piz Daint below (in a directory of your choosing).

You can also download the code as a zip file by clicking on the green __Clone or download__ button on the top right hand side of the github page, then clicking on __Download zip__.

### On Piz Daint via JupyterLab

- Go to https://jupyter.cscs.ch/ and sign in using your CSCS course credentials 
- Launch JupyterLab (might take a couple of minutes)
  - Advanced reservation 'summer_uni1' 
  - Default values for the other fields (unless told otherwise by the instructor)
- Launch a new terminal : File -> New -> Terminal
- Issue the following commands on the terminal:
```bash
ln -s $SCRATCH scratch
cd $SCRATCH
git clone https://github.com/eth-cscs/SummerUniversity2022.git
```

### On Piz Daint via ssh

This is an alternative method to the JupyterLab method above

```bash
# log onto Piz Daint ...
ssh classNNN@ela.cscs.ch
ssh daint

# go to scratch
cd $SCRATCH
git clone https://github.com/eth-cscs/SummerUniversity2022.git
```

## Updating the repository

Lecture slides and source code will be continually added and updated throughout the course.
To update the repository you can simply go inside the path

```
git pull origin main
```

There is a posibility that you might have a conflict between your working version of the repository and the origin.
In this case you can ask one of the assistants for help.

# How to access Piz Daint

This will be covered in the lectures and you can find more details in the [CSCS User Portal](https://user.cscs.ch/access/running/piz_daint/).
