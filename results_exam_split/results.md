Test set: 1605 exams from 1405 patients, 1120 with at least one fracture box.

| Subgroup | Exams | AUC |
|---|---|---|
| all | 1605 | 0.989 (0.983-0.993) |
| initial_exam | 871 | 0.986 (0.978-0.992) |
| follow_up | 734 | 0.904 (0.812-0.971) |
| certain_labels | 1570 | 0.991 (0.986-0.995) |
| two_views | 1403 | 0.989 (0.983-0.993) |

Image level: AUC 0.988 (0.983-0.992) on 3050 images, 2044 with a fracture box.

| Operating point (val) | Threshold | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| spec_0.80 | 0.017 | 0.991 (0.985-0.996) | 0.796 (0.758-0.831) | 0.918 (0.902-0.933) | 0.975 (0.959-0.990) |
| spec_0.90 | 0.085 | 0.985 (0.977-0.992) | 0.903 (0.873-0.931) | 0.959 (0.946-0.972) | 0.963 (0.944-0.979) |
| spec_0.95 | 0.287 | 0.971 (0.960-0.980) | 0.940 (0.917-0.962) | 0.974 (0.963-0.984) | 0.933 (0.909-0.955) |

| Specificity on the test ROC | Sensitivity |
|---|---|
| 0.80 | 0.991 |
| 0.90 | 0.986 |
| 0.95 | 0.968 |

Lesion level (2749 fracture boxes, IoU >= 0.3):

| FP per image | Sensitivity |
|---|---|
| 0.25 | 0.958 |
| 0.5 | 0.971 |
| 1.0 | 0.983 |
| 2.0 | 0.983 |

mAP50 (all classes) 0.552, mAP50-95 0.334, fracture AP50 0.934
