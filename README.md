# ODIR 2019 Multi-label Eye Disease Classification Challenge
Welcome to our project repository, where we focus on the ODIR 2019, a comprehensive challenge for multi-label eye disease classification. Our work centers on exploring and advancing the capabilities of the RETFound retinal disease foundational model.

## About the Project
This project evaluates the adaptability of the RETFound model to custom tasks, comparing its performance against established baseline models on the ODIR 2019 dataset. Our comparative analysis includes the following models:

ResNet50
ViT-16 Vision Transformer
EfficientNet (B3)
Each model's configuration involves concatenating the extracted features from image data and employing a classifier to diagnose patient diseases. Additionally, we explore the integration of a MIL-head in a plug-and-play manner with the foundational model, though this did not yield improvements over the baselines.

In pursuit of enhanced performance, we also experimented with fine-tuning the RETFound model for single-eye classification.

## Repository Structure
src/models: Contains all model files and their different configurations.
src/cropping.py: Script for cropping the field of view in retinal images to eliminate background noise. We recommend executing this cropping process once before training to optimize performance.
Dataset
The retinal images used in this challenge are available for download from the Grand Challenge website. We have applied specific preprocessing techniques like field of view cropping to the images to prepare them for effective training.

## Getting Started
To begin using this repository for your research or projects, clone the repo and ensure to perform the field of view cropping on your dataset as described in src/cropping.py. This preprocessing step is crucial and takes approximately 20 minutes per dataset.
