from train_advanced import run_experiment

# Test with just MobileNetV2
print("Testing fixed MobileNetV2 training...")
try:
    classifier, metrics = run_experiment(
        architecture='mobilenetv2',
        img_size=224,
        batch_size=32,
        epochs=15,  # Reduced for faster testing
        fine_tune_epochs=5
    )
    print("✅ MobileNetV2 training completed successfully!")
    print(f"Final accuracy: {metrics['test_accuracy']:.4f}")
except Exception as e:
    print(f"❌ Error: {e}")