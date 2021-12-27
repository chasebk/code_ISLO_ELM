# A study on swarm intelligence optimizing neural networks for workload elasticity prediction
* This project is based on previous project of our group: https://github.com/chasebk/code_OCRO_MLNN
* The proposed model included:
    * ELM network
    * SLO/ISLO algorithm
    * SLO/ISLO + ELM network

## How to read my repository
1. data: 
   + app: include formatted data
   + benchmark: results of benchmark functions
2. utils: Helped functions such as IO, Draw, Math, Settings (for all model and parameters), Preprocessing...
3. paper: include 2 main folders: 
    * results: forecasting results of all models (3 folders inside) 
        * final: final forecasting results (runs on server)
        * stability: final stability results(runs on server)
4. model: (4 folders)
    * app: The code for application 
      * hybrid_cfnn: MHA + CFNN 
      * hybrid_elm: MHA + ELM
      * hybrid_flnn: MHA + FLNN
      * hybrid_mlp: MHA + MLP 
      * mha: 
    * draw_opposition: Draw the figure in opposition part in paper
    * benchmark.py: The code for benchmark function
    * COA.py: COA algorithm
    * SLO.py: SLO and ISLO algorithm 

## Contact
* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

* Take a look at this repos, the simplify code using python (numpy) for all algorithms above. (without neural networks)
	
	* https://github.com/thieunguyen5991/metaheuristics

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  
