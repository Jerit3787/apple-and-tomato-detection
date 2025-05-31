import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json
import itertools
from datetime import datetime
import matplotlib.pyplot as plt

class HyperparameterOptimizer:
    def __init__(self, base_dir="working"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "Training")
        self.val_dir = os.path.join(base_dir, "Validation")
        self.test_dir = os.path.join(base_dir, "Test")
        
        # Create output directory
        self.output_dir = "output_optimization"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define hyperparameter search space
        self.hyperparams = {
            'learning_rate': [0.001, 0.0001, 0.01],
            'batch_size': [16, 32, 64],
            'architecture': ['resnet50', 'mobilenetv2'],
            'optimizer': ['adam', 'sgd'],
            'augmentation_level': ['light', 'medium', 'heavy']
        }
        
        self.results = []
        
    def create_augmentation_generator(self, level='medium'):
        """Create data augmentation generators with different intensity levels"""
        if level == 'light':
            augmentation_params = {
                'rescale': 1.0/255.0,
                'rotation_range': 10,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'horizontal_flip': True
            }
        elif level == 'medium':
            augmentation_params = {
                'rescale': 1.0/255.0,
                'rotation_range': 20,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'shear_range': 0.1,
                'zoom_range': 0.1,
                'horizontal_flip': True,
                'brightness_range': [0.9, 1.1]
            }
        else:  # heavy
            augmentation_params = {
                'rescale': 1.0/255.0,
                'rotation_range': 30,
                'width_shift_range': 0.3,
                'height_shift_range': 0.3,
                'shear_range': 0.2,
                'zoom_range': 0.2,
                'horizontal_flip': True,
                'vertical_flip': True,
                'brightness_range': [0.8, 1.2],
                'channel_shift_range': 0.1,
                'fill_mode': 'nearest'
            }
        
        return ImageDataGenerator(**augmentation_params)
    
    def create_model(self, architecture, num_classes, img_size=224):
        """Create model with specified architecture"""
        if architecture == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(img_size, img_size, 3)
            )
        elif architecture == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(img_size, img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze base model
        base_model.trainable = False
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def train_with_hyperparams(self, hyperparams, experiment_id):
        """Train model with specific hyperparameters"""
        print(f"\nExperiment {experiment_id}: {hyperparams}")
        
        # Create data generators
        train_datagen = self.create_augmentation_generator(hyperparams['augmentation_level'])
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(224, 224),
            batch_size=hyperparams['batch_size'],
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(224, 224),
            batch_size=hyperparams['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        # Create model
        num_classes = len(train_generator.class_indices)
        model = self.create_model(hyperparams['architecture'], num_classes)
        
        # Create optimizer
        if hyperparams['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        else:  # sgd
            optimizer = SGD(learning_rate=hyperparams['learning_rate'], momentum=0.9)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Calculate steps
        steps_per_epoch = min(50, train_generator.samples // hyperparams['batch_size'])
        validation_steps = min(20, val_generator.samples // hyperparams['batch_size'])
        
        # Train model
        start_time = datetime.now()
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=15,  # Reduced for faster optimization
            callbacks=callbacks,
            verbose=0
        )
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Get best metrics
        best_val_accuracy = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        final_train_accuracy = history.history['accuracy'][-1]
        
        # Calculate model size (parameters)
        model_params = model.count_params()
        
        # Store results
        result = {
            'experiment_id': experiment_id,
            'hyperparams': hyperparams,
            'best_val_accuracy': float(best_val_accuracy),
            'best_val_loss': float(best_val_loss),
            'final_train_accuracy': float(final_train_accuracy),
            'training_time_seconds': training_time,
            'model_parameters': model_params,
            'epochs_trained': len(history.history['accuracy']),
            'history': {k: [float(x) for x in v] for k, v in history.history.items()}
        }
        
        # Save individual experiment result
        with open(f'{self.output_dir}/experiment_{experiment_id}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Val Accuracy: {best_val_accuracy:.4f}, Training Time: {training_time:.1f}s")
        
        return result
    
    def random_search(self, n_trials=20):
        """Perform random hyperparameter search"""
        print(f"Starting random search with {n_trials} trials...")
        
        for i in range(n_trials):
            # Sample random hyperparameters
            hyperparams = {}
            for param, values in self.hyperparams.items():
                hyperparams[param] = np.random.choice(values)
            
            try:
                result = self.train_with_hyperparams(hyperparams, f"random_{i+1}")
                self.results.append(result)
            except Exception as e:
                print(f"Error in experiment random_{i+1}: {str(e)}")
                continue
        
        return self.results
    
    def grid_search(self, param_subset=None):
        """Perform grid search on subset of hyperparameters"""
        if param_subset is None:
            # Use a smaller subset for grid search to avoid explosion
            param_subset = {
                'learning_rate': [0.001, 0.0001],
                'batch_size': [32, 64],
                'architecture': ['resnet50'],
                'optimizer': ['adam'],
                'augmentation_level': ['medium', 'heavy']
            }
        
        # Generate all combinations
        param_names = list(param_subset.keys())
        param_values = list(param_subset.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Starting grid search with {len(combinations)} combinations...")
        
        for i, combination in enumerate(combinations):
            hyperparams = dict(zip(param_names, combination))
            
            try:
                result = self.train_with_hyperparams(hyperparams, f"grid_{i+1}")
                self.results.append(result)
            except Exception as e:
                print(f"Error in experiment grid_{i+1}: {str(e)}")
                continue
        
        return self.results
    
    def analyze_results(self):
        """Analyze and visualize optimization results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        # Sort by validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['best_val_accuracy'], reverse=True)
        
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*80)
        
        # Print top 5 results
        print("\nTop 5 Configurations:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Arch':<10} {'LR':<8} {'BS':<4} {'Opt':<6} {'Aug':<8} {'Val Acc':<8} {'Time(s)':<8}")
        print("-" * 100)
        
        for i, result in enumerate(sorted_results[:5], 1):
            hp = result['hyperparams']
            print(f"{i:<4} {hp['architecture']:<10} {hp['learning_rate']:<8} "
                  f"{hp['batch_size']:<4} {hp['optimizer']:<6} {hp['augmentation_level']:<8} "
                  f"{result['best_val_accuracy']:.4f}   {result['training_time_seconds']:.1f}")
        
        # Analyze hyperparameter importance
        self.analyze_hyperparameter_importance()
        
        # Create visualizations
        self.create_optimization_plots()
        
        # Save comprehensive results
        analysis = {
            'total_experiments': len(self.results),
            'best_configuration': sorted_results[0],
            'top_5_configurations': sorted_results[:5],
            'all_results': self.results
        }
        
        with open(f'{self.output_dir}/optimization_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def analyze_hyperparameter_importance(self):
        """Analyze which hyperparameters have the most impact"""
        print("\nHyperparameter Impact Analysis:")
        print("-" * 50)
        
        for param in self.hyperparams.keys():
            param_impact = {}
            
            for result in self.results:
                param_value = result['hyperparams'][param]
                if param_value not in param_impact:
                    param_impact[param_value] = []
                param_impact[param_value].append(result['best_val_accuracy'])
            
            # Calculate average accuracy for each parameter value
            avg_scores = {}
            for value, scores in param_impact.items():
                avg_scores[value] = np.mean(scores)
            
            # Sort by average score
            sorted_impact = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n{param.upper()}:")
            for value, avg_score in sorted_impact:
                print(f"  {value}: {avg_score:.4f} (avg)")
    
    def create_optimization_plots(self):
        """Create visualization plots for optimization results"""
        if len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        val_accuracies = [r['best_val_accuracy'] for r in self.results]
        training_times = [r['training_time_seconds'] for r in self.results]
        learning_rates = [r['hyperparams']['learning_rate'] for r in self.results]
        batch_sizes = [r['hyperparams']['batch_size'] for r in self.results]
        
        # Plot 1: Validation accuracy distribution
        axes[0,0].hist(val_accuracies, bins=10, alpha=0.7, color='blue')
        axes[0,0].set_title('Distribution of Validation Accuracies')
        axes[0,0].set_xlabel('Validation Accuracy')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Training Time
        axes[0,1].scatter(training_times, val_accuracies, alpha=0.7)
        axes[0,1].set_title('Validation Accuracy vs Training Time')
        axes[0,1].set_xlabel('Training Time (seconds)')
        axes[0,1].set_ylabel('Validation Accuracy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Impact
        lr_groups = {}
        for i, lr in enumerate(learning_rates):
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(val_accuracies[i])
        
        lrs = list(lr_groups.keys())
        lr_means = [np.mean(lr_groups[lr]) for lr in lrs]
        axes[1,0].bar(range(len(lrs)), lr_means, alpha=0.7)
        axes[1,0].set_title('Average Accuracy by Learning Rate')
        axes[1,0].set_xlabel('Learning Rate')
        axes[1,0].set_ylabel('Average Validation Accuracy')
        axes[1,0].set_xticks(range(len(lrs)))
        axes[1,0].set_xticklabels([str(lr) for lr in lrs])
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Batch Size Impact
        bs_groups = {}
        for i, bs in enumerate(batch_sizes):
            if bs not in bs_groups:
                bs_groups[bs] = []
            bs_groups[bs].append(val_accuracies[i])
        
        batch_sizes_unique = list(bs_groups.keys())
        bs_means = [np.mean(bs_groups[bs]) for bs in batch_sizes_unique]
        axes[1,1].bar(range(len(batch_sizes_unique)), bs_means, alpha=0.7, color='orange')
        axes[1,1].set_title('Average Accuracy by Batch Size')
        axes[1,1].set_xlabel('Batch Size')
        axes[1,1].set_ylabel('Average Validation Accuracy')
        axes[1,1].set_xticks(range(len(batch_sizes_unique)))
        axes[1,1].set_xticklabels([str(bs) for bs in batch_sizes_unique])
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create training curves for top 3 models
        self.plot_top_training_curves()
    
    def plot_top_training_curves(self):
        """Plot training curves for top performing models"""
        # Sort by validation accuracy and get top 3
        sorted_results = sorted(self.results, key=lambda x: x['best_val_accuracy'], reverse=True)
        top_3 = sorted_results[:3]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        colors = ['blue', 'red', 'green']
        
        for i, result in enumerate(top_3):
            history = result['history']
            label = f"Exp {result['experiment_id']} (Acc: {result['best_val_accuracy']:.3f})"
            
            # Plot accuracy
            axes[0].plot(history['accuracy'], color=colors[i], linestyle='-', alpha=0.7, label=f'{label} - Train')
            axes[0].plot(history['val_accuracy'], color=colors[i], linestyle='--', alpha=0.7, label=f'{label} - Val')
        
        axes[0].set_title('Training Curves - Top 3 Models (Accuracy)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for i, result in enumerate(top_3):
            history = result['history']
            label = f"Exp {result['experiment_id']}"
            
            # Plot loss
            axes[1].plot(history['loss'], color=colors[i], linestyle='-', alpha=0.7, label=f'{label} - Train')
            axes[1].plot(history['val_loss'], color=colors[i], linestyle='--', alpha=0.7, label=f'{label} - Val')
        
        axes[1].set_title('Training Curves - Top 3 Models (Loss)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run hyperparameter optimization"""
    optimizer = HyperparameterOptimizer()
    
    print("Starting Hyperparameter Optimization for Apple & Tomato Detection")
    print("="*70)
    
    # Option 1: Random search (faster, good exploration)
    print("\nRunning Random Search...")
    optimizer.random_search(n_trials=15)
    
    # Option 2: Grid search on subset (systematic but limited)
    print("\nRunning Grid Search on key parameters...")
    optimizer.grid_search()
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = optimizer.analyze_results()
    
    print(f"\nOptimization completed! Results saved in '{optimizer.output_dir}' directory.")
    print("Check the following files:")
    print("- optimization_analysis.json: Complete analysis")
    print("- optimization_analysis.png: Visualization plots")
    print("- experiment_*.json: Individual experiment results")

if __name__ == "__main__":
    main()