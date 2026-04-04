"""
Example script demonstrating KTSPClassifier usage
"""

from pipeline import (
    KTSPClassifier, 
    load_and_normalize_data, 
    split_data_processing_labels
)

def main():
    # Configuration
    DATA_PATH = r"D:\fitchburg\Bu_ML\Data\Lung.txt"
    TOP_GENES = 300
    TOP_PAIRS = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print("=" * 60)
    print("KTSPClassifier Example")
    print("=" * 60)
    
    # Step 1: Load and split data
    print("\n1. Loading data...")
    X, y = load_and_normalize_data(DATA_PATH)
    print(f"   Data shape: {X.shape}")
    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of genes: {X.shape[1]}")
    
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = split_data_processing_labels(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Step 2: Initialize classifier
    print("\n3. Initializing KTSPClassifier...")
    clf = KTSPClassifier(top_genes=TOP_GENES, top_pairs=TOP_PAIRS)
    print(f"   Top genes: {clf.top_genes}")
    print(f"   Top pairs: {clf.top_pairs}")
    
    # Step 3: Train classifier
    print("\n4. Training classifier...")
    clf.fit(X_train, y_train)
    print(f"   ✓ Selected {len(clf.selected_genes)} genes")
    print(f"   ✓ Selected {len(clf.gene_pairs)} gene pairs")
    print(f"   Top 3 gene pairs:")
    for i, (gene_i, gene_j, score) in enumerate(clf.gene_pairs[:3], 1):
        print(f"      {i}. Gene {gene_i} vs Gene {gene_j} (score: {score:.4f})")
    
    # Step 4: Make predictions
    print("\n5. Making predictions...")
    y_pred = clf.predict(X_test)
    print(f"   Predicted labels: {y_pred[:10]}...")  # Show first 10
    
    # Step 5: Get prediction probabilities
    print("\n6. Getting prediction probabilities...")
    probs = clf.predict_proba(X_test)
    print(f"   Decision scores: {probs[:10]}...")  # Show first 10
    
    # Step 6: Evaluate model
    print("\n7. Evaluating model...")
    results = clf.evaluate(X_test, y_test)
    print(f"   Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Confusion Matrix:")
    print(f"{results['confusion_matrix']}")
    
    if results['roc_data']:
        print(f"   ROC AUC: {results['roc_data']['auc']:.4f}")
    
    # Step 7: Visualize results
    print("\n8. Visualizing results...")
    clf.visualize(X_test, y_test)
    print("   ✓ Plots displayed")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


