#功能：one_shot_fixed_prompt_for_mimic_data

#...接各自测试的前边代码
ImpressionGPT_MIMIC_one_shot="""
You are a chest radiologist that identifies the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS. What are the main findings and diagnosis or impression based on the given Finding and the given Impression in chest X-ray report:
. Here's an example:
### Input: findings: There has been interval widening of the mediastinum due to vascular engorgement.  In addition, there is new bilateral interstitial edema.  A possible left pleural effusion and atelectasis obscure the left cardiac and hemidiaphragmatic contours more than the prior day.  The small right pleural effusion and basilar atelectasis is unchanged.  There is no pneumothorax.  The support and indwelling lines are unchanged and in their expected locations.
### Response: Mild acute congestive heart failure.
"""
ImpressionGPT_OpenI_one_shot="""
You are a chest radiologist that identifies the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS. What are the main findings and diagnosis or impression based on the given Finding and the given Impression in chest X-ray report:
. Here's an example:
### Input: findings: Heart size and pulmonary vascularity normal. There is a small right pleural effusion. There is infrahilar interstitial prominence which may represent bronchovascular crowding lung. Small left pleural effusion. No pneumothorax.
### Response: Small bilateral pleural effusions and right infrahilar infiltrate versus bronchovascular crowding.
"""

#...接入各自测试的后续代码
#...接各自测试的前边代码
ImpressionGPT_MIMIC_five_shot="""
You are a chest radiologist that identifies the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS. What are the main findings and diagnosis or impression based on the given Finding and the given Impression in chest X-ray report:
. Here are five examples:
### Input: findings: Frontal and lateral radiographs of the chest redemonstrate a round calcified pulmonary nodule in the posterior right lung base, unchanged from multiple priors and consistent with prior granulomatous disease.  A known enlarged right hilar lymph node seen on CT of ___ likely accounts for the increased opacity at the right hilum.  A known right mediastinal lymph node conglomerate accounts for the fullness at the right paratracheal region.  No pleural effusion, pneumothorax or focal consolidation is present.  The patient is status post median sternotomy and CABG with wires intact.  The cardiac silhouette is normal in size.  The mediastinal and hilar contours are unchanged from the preceding radiograph.
### Response: No acute cardiopulmonary process.
### Input: findings: The lungs are well expanded and clear.  The cardiomediastinal silhouette, hilar contours, and pleural surfaces are normal. No pleural effusion or pneumothorax is present.  Sternal wires are intact.
### Response: No acute intrathoracic process.
### Input: findings: Two images of the chest shows a small consolidation at the right base, most consistent with pneumonia.  There are no other consolidations.  There is no evidence of interstitial edema.  There are no pleural effusions.  The heart size is at the upper limits of normal.  The mediastinal contours are normal.  There are sternotomy wires in place.
### Response: Consolidation in the right base is most consistent with pneumonia.
### Input: findings: Frontal and lateral views of the chest were obtained.  Left-sided Port-A-Catheter is similar in position, terminating at the cavoatrial/right atrial junction.  Patient has diffuse increase in interstitial markings bilaterally consistent with patient's underlying history of chronic interstitial lung disease with likely overlying pulmonary edema improved since ___, but similar in appearance as compared to ___.  No definite focal consolidation or pleural effusion.  Multilevel vertebroplasties are seen along the thoracic spine, similar to prior.
### Response: Pulmonary edema superimposed on known lung fibrosis.
### Input: findings: There has been improvement in mild-to-moderate pulmonary edema with decreased interstitial markings compared to most recent prior study.  Small bilateral pleural effusions have resolved.  There is no focal consolidation or pneumothorax.  Heart size is moderately enlarged and stable.  A left chest wall Port-A-Cath terminates in the RA.  The patient is status post multiple vertebroplasties.
### Response: Improved but not resolved mild-to-moderate pulmonary edema.
"""
ImpressionGPT_OpenI_five_shot="""
You are a chest radiologist that identifies the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS. What are the main findings and diagnosis or impression based on the given Finding and the given Impression in chest X-ray report:
. Here are five examples:
### Input: findings: Cardiac and mediastinal contours are within normal limits. The lungs are clear. Acromioclavicular arthritis is present, XXXX severe.
### Response: No acute preoperative findings.
### Input: findings: The cardiomediastinal silhouette is within normal limits for size. Pulmonary vasculature is within normal limits. No focal consolidations, effusions, or pneumothoraces. Mild degeneration of the thoracic spine without acute bony abnormality.
### Response: No acute cardiopulmonary abnormality.
### Input: findings: Heart is within normal limits. Negative for focal pulmonary consolidation, pleural effusion, or pneumothorax. Mild streaky opacity lateral right lung, atelectasis versus scarring.
### Response: Small streaky opacity lateral right lung, subsegmental atelectasis versus scarring.
### Input: findings: In the interval, bibasilar interstitial infiltrates and pulmonary venous engorgement have resolved. Heart size is now normal. No XXXX infiltrates.
### Response: Chest. Resolving pulmonary interstitial edema and pulmonary venous hypertension.
### Input: findings: Consolidation and some atelectasis are present in the left lower lobe. Patchy interstitial infiltrates are also present in the right lower lobe. Bilateral costophrenic XXXX blunting is present. Heart and pulmonary XXXX are normal.
### Response: Bibasilar airspace disease, left worse right. Bilateral pleural fluid.
"""

#----------------------------------------------------------
### Input: findings: 
### Response: 
### Input: findings: 
### Response: 
### Input: findings: 
### Response: 
### Input: findings: 
### Response: 


