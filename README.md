# DGRL_solve_multicut
This repository contains the source code and instance sets for the paper titled "Deep Graph Reinforcement Learning for Solving Multicut Problem".

--------------------------------------------------
# Folders:

* /DGRL_multicut
    - our SS2V-D3QN method, based on deep graph reinforcement learning (DGRL)
  
* /baselines
    - baseline multicut solvers

* /CPLEX
    - exact solver

* /BA-20, /BA-40, /BA-60, /WS-20, /WS-40, /WS-60, /ER-20, /ER-40, /ER-60
    - multicut instance sets (synthetic)

* /generalization
    - out-of-distribution generalization ability

* /larger
    - generalization ability to much larger instances from the same graph model

* /realworld
    - multicut instance sets (real-world)

--------------------------------------------------
# Operating System: CentOS 7.9
# Python Environments:

* To run /DGRL_multicut/train.py, /DGRL_multicut/valid.py, /DGRL_multicut/test.py, /DGRL_multicut/generalization.py, /DGRL_multicut/larger.py:
    - Refer to requirements_for_DGRL_multicut.txt

* To run /baselines/baselines.py:
   - Refer to requirements_for_baselines.txt and the 'nifty' library with QPBO functionality

* To run /CPLEX/cplex.py:
    - Refer to requirements_for_CPLEX.txt and the 'nifty' library with CPLEX functionality

Note: 'nifty' source code is available at https://github.com/DerThorsten/nifty?tab=readme-ov-file
