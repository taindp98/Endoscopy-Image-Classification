# Semi-supervised Learning with Transformers for Gastrointestinal Pathologies Classification in Endoscopic Images

• In progress building up a deep learning model using the PyTorch framework to classify the abnormalities in endoscopic images.

• Dealing with the lack of labeled images which is a challenge in medical image analysis by applying Semi-supervised learning (FixMatch and CoMatch).

## Image Labels
Hyper-Kvasir includes the follow image labels for the labeled part of the dataset:

| ID | Label | ID | Label
| --- | --- | --- | --- |
| 0  | barretts | 12 |  oesophagitis-b-d
| 1  | bbps-0-1 | 13 |  polyp
| 2  | bbps-2-3 | 14 |  retroflex-rectum
| 3  | dyed-lifted-polyps | 15 |  retroflex-stomach
| 4  | dyed-resection-margins | 16 |  short-segment-barretts
| 5  | hemorrhoids | 17 |  ulcerative-colitis-0-1
| 6  | ileum | 18 |  ulcerative-colitis-1-2
| 7  | impacted-stool | 19 |  ulcerative-colitis-2-3
| 8  | normal-cecum | 20 |  ulcerative-colitis-grade-1
| 9  | normal-pylorus | 21 |  ulcerative-colitis-grade-2
| 10 | normal-z-line | 22 |  ulcerative-colitis-grade-3
| 11 | oesophagitis-a |  |  |
