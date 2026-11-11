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
- **Streamlit Community Cloud:** live at https://churnguard-customer-churn-prediction-cfs6yhah5jvsce2n5msxkt.streamlit.app/. Chosen because the app is Streamlit: it builds directly from the GitHub repo and `requirements.txt` and redeploys on push.
- **Hugging Face Spaces: not deployed.** First attempt failed because the README metadata said `sdk: gradio` while the code was a Streamlit app. Hugging Face has no built-in Streamlit SDK, so the app would need a Docker Space, which was not available on a free account when I tried. Pushing the repo to the Space was also rejected by the server: "Your push was rejected because it contains binary files" (offending files: `models/model.joblib`, `reports/shap_summary.png`). The full explanation is in the README under "Why not Hugging Face Spaces". The Dockerfile and Hugging Face metadata were removed because the app is deployed on Streamlit Cloud only.

## Fixes after the second supervisor review
- Removed the template placeholders from this file.
- README: removed the `scripts/check_readme.py` reference (the script no longer exists) and Docker from "Technologies", added a key-results table, a "Why not Hugging Face Spaces" section with the real errors, a "why Streamlit instead of Gradio" note, and a note that results were produced with the versions pinned in `requirements.txt`.
- App: dollar signs are now escaped in `streamlit_app.py` so `$200` is no longer rendered as LaTeX math (checked with Streamlit's AppTest).
- Git: history is no longer rewritten; every change is a separate commit.

## Demo video
https://www.loom.com/share/e02ab61b52a04553903ef844bb82fc77

Shows a high-risk and a low-risk customer with the top 3 reasons, then a CSV upload with the scored table and the download.