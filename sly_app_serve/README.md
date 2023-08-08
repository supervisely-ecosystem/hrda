<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/hrda/assets/115161827/59697391-5f58-472c-bdcb-7341bfc7ec79"/>  

# Serve HRDA

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-apps">Related Apps</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/hrda)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/hrda/sly_app_serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/hrda/sly_app_serve.png)](https://supervise.ly)

</div>

# Overview

xxx

# How To Run

## Custom models

This model does not come with pre-trained models option. To create a custom model, use the application below:
- [Train HRDA](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train) - app allows to create custom HRDA weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_train" src="xxx" height="70px" margin-bottom="20px"/>

To serve the custom model, copy model file path from Team Files, paste it into the dedicated field, select the device and press `Serve` button

<img src="xxx"/>

# Related apps

- [NN Image Labeling](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fnn-image-labeling%252Fannotation-tool) - integrate any deployed NN to Supervisely Image Labeling UI. Configure inference settings and model output classes. Press `Apply` button (or use hotkey) and detections with their confidences will immediately appear on the image.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/nn-image-labeling/annotation-tool" src="https://i.imgur.com/hYEucNt.png" height="70px" margin-bottom="20px"/>

- [Apply NN to Videos Project](https://ecosystem.supervise.ly/apps/apply-nn-to-videos-project) - app allows to label your videos using served Supervisely models.  
  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-nn-to-videos-project" src="https://imgur.com/LDo8K1A.png" height="70px" margin-bottom="20px" />

- [Train HRDA](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/hrda/sly_app_train) - app allows to create custom HRDA weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/hrda/sly_app_train" src="xxx" height="70px" margin-bottom="20px"/>
    
# Acknowledgment

This app is based on the great work `HRDA` ([github](https://github.com/lhoyer/HRDA)). ![GitHub Org's stars](https://img.shields.io/github/stars/lhoyer/HRDA?style=social)
