<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/9b0c3482-a55c-440e-afea-ff1a935836c2"/>  

# Train HRDA

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-apps">Related apps</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/hrda)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/hrda/sly_app_train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/hrda/sly_app_train.png)](https://supervise.ly)

</div>

# Overview

xxx

# How To Run

1. Run the application from the Ecosystem and select the input project, or run from the context menu of a project <br> </br>
2. Select either a pre-trained model, or provide your own weights
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/98a331d6-7692-4c05-af92-25412029b035" /> <br> </br>
3. Select the classes that will be used for training
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/a8a26831-db7f-4775-947a-779543416f51" /> <br> </br>
4. Define source, target and validation datasets
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/b3c58004-4746-4482-8f8b-a5b0374dd38c" /> <br> </br>
5. Use either pre-defined or custom augmentations
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/8f23509b-642a-4875-bc43-bd97688352ee" /> <br> </br>
6. Configure training hyperparameters
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/f04cfcdc-fb9c-4bba-8fbd-21289487d730" /> <br> </br>
7. Press `Train` button and observe the logs, charts and prediction visualizations
   <img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/924fa93e-ef42-4daa-987e-5629da1c1530" /> <br> </br>
   
# Related apps

- [Export to YOLOv8 format](https://ecosystem.supervise.ly/apps/export-to-yolov8) - app allows to transform data from Supervisely format to YOLOv8 format.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/export-to-yolov8" src="https://github.com/supervisely-ecosystem/yolov8/assets/115161827/01d6658f-11c3-40a3-8ff5-100a27fa1480" height="70px" margin-bottom="20px"/>  

- [Serve HRDA](https://ecosystem.supervise.ly/apps/hrda/sly_app_serve) - app allows to deploy YOLOv8 model as REST API service.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_serve" src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/9539f5f3-3413-40d1-8880-bed7e1061c3c" height="70px" margin-bottom="20px"/>
  
# Screenshot

<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/c3e4bae6-02b9-4d2e-8f59-ac65996505e7"/>


# Acknowledgment

This app is based on the great work `HRDA` ([github](https://github.com/lhoyer/HRDA)). ![GitHub Org's stars](https://img.shields.io/github/stars/lhoyer/HRDA?style=social)
