# AdaptChem
Adaptive Atmospheric Chemistry Mechanism Toolkit and Framework with Hybrid Machine Learning


# Stucture of the Framework

    This repo follows the structure below:

    - Expand 
    Expand the current network using the function in the expander

    - Adapt 
    Finetune the current network based on the new mechanism or observation using the function
    inside of adaptor

    - wrapper
    Wrap the model into static lib so that the C or Fortran Code can use the Mechanism
    using the function inside of the wrapper

# Example of Use of the Code

# Installation
pip install git+https://github.com/fzhao70/AdaptChem.git