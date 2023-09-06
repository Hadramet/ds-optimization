### Dataset (Real image) Investigation:
base on https://arxiv.org/pdf/2110.03613.pdf

1. **Ensemble Prediction**:
   - For each sample in your dataset, obtain predictions from each of the pretrained models: TrOCR, PaddleOCR, and KerasOCR.
   
2. **Consensus Voting**:
   - For each sample, determine if there's a consensus among the predictions. If all (or a majority) of the models agree on the transcription, it's likely that the transcription is correct.
   
3. **Flagging Discrepancies**:
   - If the models disagree on a transcription, flag that sample for manual review. The disagreement indicates potential issues with the sample, such as mislabeling, poor image quality, or inherent ambiguity.
   
4. **Manual Review**:
   - Review the flagged samples manually. Correct any mislabeled samples and, if necessary, remove any samples that are of poor quality or are too ambiguous for reliable transcription.
   
5. **Confidence-based Sorting (Optional)**:
   - If the models provide confidence scores for their predictions, sort samples based on the average confidence of the ensemble. Samples with lower confidence scores can be reviewed manually as they might be challenging or mislabeled.

### N-fold Cross-Validation:

1. **Data Splitting**:
   - Divide your optimized dataset into N equally sized folds. Ensure that each fold has a representative distribution of different characters, words, or any other relevant criteria.
   
2. **Iterative Training and Validation**:
   - For each iteration:
     - Use N-1 folds for training and the remaining fold for validation.
     - Train your ResNet34-based OCR model on the training folds.
     - Validate the model on the validation fold and record the performance metrics.
   
3. **Performance Assessment**:
   - After completing the N iterations, average the performance metrics from each validation fold to get an overall assessment of your model's performance.
   - This approach helps ensure that your model is robust and performs well on various subsets of your data.

4. **Final Model Training**:
   - Once satisfied with the cross-validation results, train your model on the entire dataset for final deployment.
