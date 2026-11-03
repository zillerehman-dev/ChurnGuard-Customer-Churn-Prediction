# Intern notes: ChurnGuard (Machine Learning, Week 2)
Intern: Zille Rehman

## Approach
Predict telecom customer churn, explain each prediction, and turn the score into a retention decision using a cost model.
1. **Data and leakage:** 7,043 customers, 19 model inputs. `Churn Score`, `Churn Reason` and `CLTV` are excluded because they are derived from or recorded after the outcome; CLTV is used only in a labelled reporting view.
2. **Split and selection:** stratified 80/20 split (seed 42). Model selection, calibration and threshold tuning use the training split only (5-fold stratified CV, PR-AUC); the test set is scored once.
3. **Models:** dummy baseline, Logistic Regression, Random Forest and XGBoost, with class weights or SMOTE (SMOTE inside the imblearn Pipeline). XGBoost with class weights was chosen, narrowly (0.6859 vs 0.6793 for Random Forest).
4. **Calibration:** isotonic, chosen by Brier score (0.1534 uncalibrated to 0.1291). Sigmoid is slightly better on log loss (0.4008 vs 0.4190); noted in the README.
5. **Decision threshold:** 0.32, chosen by minimising expected cost on out-of-fold training predictions (break-even probability 33.3%).
6. **Explanations:** SHAP values summed back to the original features, shown as the top 3 reasons per customer.
7. **App:** Streamlit app with single prediction and CSV upload; also shows value at stake and an offer-priority rank.

## Final test metrics (1,409 customers, scored once)
PR-AUC 0.6724, ROC-AUC 0.8538. At threshold 0.32: precision 0.546, recall 0.749, F1 0.631 (TN 802, FP 233, FN 94, TP 280). Expected saving vs no offers: $6,540 under assumed costs (loss $200, offer $20, 30% success); contacting everyone loses $5,740.

## Deployment
- **Streamlit Community Cloud:** live at ADD_STREAMLIT_LINK. Chosen because the app is Streamlit: it builds directly from the GitHub repo and `requirements.txt` and redeploys on push.
- **Hugging Face Spaces:** my first attempt failed. Exact error from the Space build/runtime log:
  `PASTE_THE_EXACT_ERROR_TEXT_HERE`
  Likely cause (check it against the log above): the README metadata said `sdk: gradio` and the README described a Gradio app, but `app.py` was a copy of the Streamlit app, and Hugging Face no longer has a built-in Streamlit SDK. Fix: the README now uses `sdk: docker` with `app_port: 8501`, there is a `Dockerfile`, the unused `gradio` pin was removed from `requirements.txt`, and the duplicate `app.py` was deleted so there is a single entry point (`streamlit_app.py`).
  Result of the second attempt: PASTE_RESULT_AND_SPACE_LINK_OR_REMAINING_ERROR_HERE

## Demo video
ADD_VIDEO_LINK

## Revisions after supervisor review
README numbers regenerated from the saved reports; draft text removed; Streamlit documented and pinned; modelling notebook added; sample CSV added; calibration/SHAP note added to the app; per-customer value estimate added.
