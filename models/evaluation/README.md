This folder contains end-to-end predictions of waiting times for the test set. As E2E 
prediction takes some time, the raw predictions generated via `src.evaluation.test_e2e`
are stored here and used in the evaluation notebooks. 

The raw predictions are not checked into git for file size reasons, you can reproduce 
them via `make e2e_test`.