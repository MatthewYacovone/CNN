import cnn_ensemble
import ood_classifier

for n in range(4, 12):
    print(f'\n=== Running ensemble with {n} models ===')

    # Generate disagreement df for current ensemble size
    disagreement_df = cnn_ensemble.run_ensemble(n_models=n)

    # Save progress to a CSV
    disagreement_df.to_csv(f'ensemble_disagreement_{n}.csv', index=False)

    # Run OOD classifier on this df
    print(f'--- Naive Bayes classifier report for ensemble size {n} ---')
    report = ood_classifier.run_classifier(disagreement_df)

    # Save report to a file
    with open(f'ood_report_{n}.txt', 'w') as f:
        f.write(report)