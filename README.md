** compatibility notes about OWL and COOM

- gym microgrid is outtodate. Needed to change np.boolto bool in microgrid.py fiole of gym library
- also COOM and OWL repos here should be downloaded as code and not cloned
- COOM especially is probably broken. Should be installed via 
   $ pip install COOM 
   and then the COOM folder (not the whole COOM-main) should be put in "C:\Users\<USER>\anaconda3\envs\rl_env2\lib\site-packages\"


   ** MODIFIES FILES in COOM library:
   a "COOM - modified" folder is included in this repo. Comments in the form of three hastags + uppercase show the lines of code modifies by me; example: '### THIS IS A COMMENT BY ME'