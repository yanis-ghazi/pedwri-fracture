Test set: 2830 exams from 2312 patients, 1898 with at least one fracture box.

| Subgroup | Exams | AUC |
|---|---|---|
| all | 2830 | 0.985 (0.981-0.989) |
| initial_exam | 1510 | 0.979 (0.971-0.986) |
| follow_up | 1320 | 0.957 (0.938-0.972) |
| certain_labels | 2754 | 0.988 (0.985-0.991) |
| two_views | 218 | 0.994 (0.986-0.998) |

Image level: AUC 0.986 (0.982-0.989) on 3050 images, 2033 with a fracture box.

| Operating point (val) | Threshold | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| spec_0.80 | 0.020 | 0.985 (0.979-0.990) | 0.806 (0.780-0.831) | 0.912 (0.899-0.924) | 0.963 (0.949-0.976) |
| spec_0.90 | 0.108 | 0.973 (0.965-0.980) | 0.889 (0.869-0.909) | 0.947 (0.937-0.957) | 0.941 (0.924-0.956) |
| spec_0.95 | 0.413 | 0.944 (0.933-0.954) | 0.953 (0.938-0.966) | 0.976 (0.968-0.983) | 0.893 (0.873-0.913) |

| Specificity on the test ROC | Sensitivity |
|---|---|
| 0.80 | 0.985 |
| 0.90 | 0.969 |
| 0.95 | 0.946 |

Lesion level (2716 fracture boxes, IoU >= 0.3):

| FP per image | Sensitivity |
|---|---|
| 0.25 | 0.959 |
| 0.5 | 0.971 |
| 1.0 | 0.980 |
| 2.0 | 0.985 |

mAP50 (all classes) 0.596, mAP50-95 0.372, fracture AP50 0.938
