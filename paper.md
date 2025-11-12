---
title: 'AdaptChem: A Adaptive Atmospheric Chemistry Mechanism Framework'
tags:
  - Python
  - Reinforcement Learning
  - Machine Learning
  - Atmospheric chemistry
authors:
  - name: Fanghe Zhao
    orcid: 0009-0000-1317-1368
    affiliation: "1"
  - name: Yuhang Wang
    orcid: 0000-0002-7290-2551
    affiliation: "1"
affiliations:
 - name: School of Earth & Atmospheric Sciences, Georgia Institute of Technology, US
   index: 1

date: 4 December 2024
bibliography: paper.bib
---

# Summary

AdaptChem is a Python framework designed to facilitate the adaptation of pre-trained neural network models for atmospheric chemistry mechanisms. The framework addresses the challenge of updating machine learning surrogates when new observational data becomes available or when chemical mechanisms need to be extended with additional species or reactions. AdaptChem provides three complementary adaptation strategies: (1) traditional fine-tuning with uncertainty quantification through Monte Carlo sampling, (2) reinforcement learning using Soft Actor-Critic (SAC) for continuous optimization, and (3) direct preference optimization (DPO) for learning from comparative data quality. The framework also includes utilities for dynamically extending model dimensions while preserving learned representations, and tools for exporting adapted models to TorchScript for integration with legacy atmospheric chemistry codes written in C or Fortran. By combining multiple adaptation approaches with uncertainty awareness and interoperability features, AdaptChem enables researchers to maintain and update machine learning components of atmospheric chemistry models efficiently as new data and requirements emerge.

# Statement of need

Atmospheric chemistry models are computationally expensive, often requiring hours to days of compute time for regional or global simulations. Machine learning surrogates have emerged as a promising approach to accelerate these simulations by learning to emulate chemical mechanisms from pre-computed training data. However, a critical challenge arises when new observational data becomes available or when the chemical mechanism needs to be extended—the existing neural network model must be adapted rather than retrained from scratch, which would be prohibitively expensive and wasteful of the knowledge already captured.

Existing machine learning frameworks provide general-purpose fine-tuning capabilities, but they lack the domain-specific features needed for atmospheric chemistry applications. Specifically, atmospheric measurements come with associated uncertainties that should inform the adaptation process, chemical mechanisms may need to accommodate new species (requiring input/output dimension changes), and adapted models often need to integrate with existing atmospheric codes written in Fortran or C. Furthermore, different adaptation scenarios may benefit from different optimization strategies—supervised fine-tuning for well-characterized observations, reinforcement learning for optimizing against simulation-based rewards, or preference optimization when comparative data quality information is available.

AdaptChem fills this gap by providing a unified framework that combines multiple adaptation methods with domain-specific features for atmospheric chemistry. The framework's uncertainty quantification through Monte Carlo sampling allows models to weight observations appropriately based on measurement confidence. The dimension extension capability enables adding new chemical species without retraining the entire model. The reinforcement learning approach allows optimization against complex reward functions that may include physical constraints or domain knowledge. The direct preference optimization method enables learning from pairwise comparisons of data quality, which is particularly valuable when absolute uncertainty estimates are unavailable but relative quality can be assessed.

By providing these capabilities in a modular, easy-to-use package, AdaptChem lowers the barrier for atmospheric chemistry researchers to leverage machine learning in their work and maintain these models over time. The framework is designed to work with standard PyTorch models, making it compatible with existing workflows while adding atmospheric chemistry-specific functionality. The C/Fortran integration capabilities ensure that adapted models can be deployed in production atmospheric chemistry codes, bridging the gap between machine learning research and operational atmospheric modeling.

AdaptChem is intended for researchers working at the intersection of atmospheric chemistry and machine learning, as well as operational forecasting centers that wish to incorporate or update machine learning components in their atmospheric models. The framework has been used in ongoing research on adaptive chemical mechanisms and is released as open-source software to benefit the broader atmospheric science community.

# Acknowledgements

We would like to acknowledge high-performance computing support from the Derecho system (doi:10.5065/qx9a-pg09) provided by the NSF National Center for Atmospheric Research (NCAR), sponsored by the National Science Foundation under project number of UGIT0038.
