Data files and codes related to the paper: "Confidence modulates the decodability of scene prediction during partially-observable maze exploration" by Katayama, Yoshida and Ishii, and codes for the computational model of subjects’ exploration behavior.

## The "Data" folder is organized as follows:  
 * /Behavior/subX_sesY.mat:  
  The behavioral data for the Y th session of subject X.  
  
  Subjects 1-26 were included the behavior, imaging and decoding analyses.
  
  Subject 27 was excluded from the imaging and decoding analyses due to his/her large head motion.
  
  Subjects 28-33 were excluded from the analyses due to their low scene prediction performance (see also Supplementary Methods and Supplementary Figure 2).
  
 * /Decoding_results/Xth_decoding_period_result.mat:  
  The time-series decoding results for the X-th decoding period.
  
  We constructed the decoders of the scene prediction and the subject’s reported confidence level at nine different time points, the 0th to the 8th decoding periods: the decoders at the t-th period used four consecutive scans starting from t s after the onset of the delay period. For details, please see the Method section in the main manuscript.
  
  Each field of the structure corresponds to each decoding condition, [Nsbj x Nroi]. Each column corresponds to the decoding results within each ROI: SPL, SMG, PMd and aPFC, respectively from the left column.
  
 * /activate_sceneprediction_26sbj.nii:  
 the result of a univariate general linear model analysis during the prediction of the upcoming scene (first 4 s of the delay period) (see Figure 3a and Methods).  
  
## The "BehaviorModels" folder is organized as follows:  
* StrategySwitching.m:  
* AlternativeModels.m:  
* Maze25.mat:  
* model_parameters_est.mat:  
