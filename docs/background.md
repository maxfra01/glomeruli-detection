## Diabetic Kidney Disease: Challenges, Progress, and Possibilities

[Clinical journal of the American Society of Nephrology, 2017](https://pubmed.ncbi.nlm.nih.gov/28522654/)

This paper presents a comprehensive overview of Diabetic Kidney Disease (DKD), which is a major microvascular complication of diabetes and the leading cause of chronic kidney disease (CKD) worldwide. DKD affects approximately 30% of patients with type 1 diabetes and 40% of those with type 2 diabetes. It is associated with significant morbidity and mortality, particularly from cardiovascular causes, often preceding the need for dialysis or kidney transplantation.

The authors describe the natural history of DKD, which typically begins with glomerular hyperfiltration and progresses through stages of albuminuria and declining glomerular filtration rate (GFR), culminating in end-stage renal disease (ESRD). 
Structural changes in the kidney:
- Thickening of the glomerular basement membrane (GBM) – one of the earliest and most consistent alterations.
- Mesangial matrix expansion – increases progressively with disease severity.
- Loss of podocytes – including foot process effacement and detachment from the GBM.
- Formation of Kimmelstiel–Wilson nodules – nodular glomerulosclerosis characteristic of advanced DKD.
- Segmental and global glomerulosclerosis – key markers of irreversible damage and fibrosis.
- Capillary and tubular basement membrane thickening – contributing to impaired filtration.
- Arteriolar hyalinosis and microaneurysms – linked with luminal narrowing and ischemia.

Key risk factors are categorized into susceptibility (e.g., age, genetics), initiation (e.g., hyperglycemia, hypertension), and progression (e.g., poor metabolic control, obesity). Intensive glycemic control has shown long-term benefits, especially when initiated early, though later-stage interventions yield diminishing returns and may increase the risk of hypoglycemia.

The diagnosis of DKD relies on persistently elevated albuminuria and/or reduced eGFR, but the authors highlight that histopathological damage may be present even when clinical markers appear normal. This underscores the limitations of current diagnostic criteria and the need for more sensitive and specific tools.

Therapeutically, the mainstays remain glycemic and blood pressure control, particularly using renin-angiotensin system inhibitors in patients with albuminuria. However, there is still a large unmet need, and several new therapeutic agents targeting fibrosis, inflammation, and glomerular hyperfiltration are in development.

The paper emphasizes the importance of early intervention, the need for better biomarkers, and the potential for innovation—particularly in how we assess renal damage. This opens the door to integrating artificial intelligence in pathology for automated and more precise glomerular lesion detection.

## Glomerulosclerosis identification in whole slide images using semantic segmentation

[Computer methods and programs in biomedicine, 2020](https://pubmed.ncbi.nlm.nih.gov/31891905/)

Glomeruli identiﬁcation, i.e., detection and characterization, is a key procedure in many nephropathology studies. In this paper, semantic segmentation based on convolutional neural networks (CNN) is proposed to detect glomeruli using Whole Slide Imaging (WSI) follows by a classiﬁcation CNN to divide the glomeruli into normal and sclerosed.
Two approaches are proposed:
- A 3-class segmentation network to classify glomeruli into normal, sclerosed, and non-glomeruli (background)


- A 2-class segmentation network to classify glomeruli into normal and non-glomeruli (background) and a classification network to classify the glomeruli into normal and sclerosed

The **data preprocessing** pipeline is as follows: the WSIs are divided into smaller images (tiles) of size 2000x2000 pixels. Then the set of tiles are labeled as sclerotic glomeruli, normal glomeruli, or background. They obtained 1245 glomerular structures. Data **augmentation** is applied to the tiles to increase the number of training samples.

The **3-class pipeline** is as follows:
1. From the WSI, 2000x2000 patches are extracted
2. The patches are resized to 400x400 pixels
3. The patches are fed to a semantic segmentation network to classify them into normal glomeruli, sclerotic glomeruli, and background 

The **2-class pipeline** is as follows:
1. From the WSI, 2000x2000 patches are extracted
2. The patches are resized to 400x400 pixels
3. The patches are fed to a semantic segmentation network to classify them into glomeruli and non-glomeruli structures
4. The boundaries of the glomeruli are extracted 
5. The glomeruli are classified into normal and sclerotic using a classification network

U-net and SegNet (VGG16 and VGG19) architectures are used for the segmentation task. AlexNet is used for classification in the 2-class approach. Hyperparameters are provided in the paper. 
Metrics used for evaluation are based on True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN) values.

The best results are obtained with the 2-class segmentation network, i.e. consecutive CNNs for segmentation and classification. 