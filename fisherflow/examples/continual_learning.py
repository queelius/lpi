"""
Continual Learning with Elastic Weight Consolidation (EWC) as Fisher Flow.

This example demonstrates:
1. How EWC prevents catastrophic forgetting using Fisher Information
2. Sequential task learning without losing previous knowledge
3. Fisher Information as a measure of parameter importance
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from fisherflow.autograd import Value, Module, MLP
from fisherflow.optimizers import ElasticWeightConsolidation, DiagonalFisherFlow


def generate_task_data(task_id, n_samples=200):
    """Generate data for different tasks."""
    np.random.seed(task_id * 42)
    
    if task_id == 0:
        # Task A: Learn sin(x)
        X = np.random.uniform(-np.pi, np.pi, n_samples)
        y = np.sin(X) + np.random.randn(n_samples) * 0.1
        task_name = "sin(x)"
    elif task_id == 1:
        # Task B: Learn x^2
        X = np.random.uniform(-2, 2, n_samples)
        y = X**2 + np.random.randn(n_samples) * 0.1
        task_name = "x²"
    elif task_id == 2:
        # Task C: Learn cos(2x)
        X = np.random.uniform(-np.pi, np.pi, n_samples)
        y = np.cos(2*X) + np.random.randn(n_samples) * 0.1
        task_name = "cos(2x)"
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
    
    return X, y, task_name


def create_batches(X, y, batch_size=32):
    """Create batches from data."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    batches = []
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches


def mse_loss(batch, model):
    """Mean squared error loss for regression."""
    X, y = batch
    total_loss = Value(0)
    
    for xi, yi in zip(X, y):
        xi_val = Value(xi)
        pred = model([xi_val])
        if isinstance(pred, list):
            pred = pred[0]
        loss = (pred - yi) ** 2
        total_loss = total_loss + loss
    
    return total_loss / len(X)


def evaluate_model(model, X, y):
    """Evaluate model performance."""
    predictions = []
    for xi in X:
        xi_val = Value(xi)
        pred = model([xi_val])
        if isinstance(pred, list):
            pred = pred[0]
        predictions.append(pred.data)
    
    predictions = np.array(predictions)
    mse = np.mean((predictions - y) ** 2)
    return mse, predictions


def train_task(model, optimizer, X, y, epochs=100, verbose=True):
    """Train model on a single task."""
    batches = create_batches(X, y, batch_size=32)
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch in batches:
            optimizer.zero_grad()
            loss = mse_loss(batch, model)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
        
        avg_loss = epoch_loss / len(batches)
        history.append(avg_loss)
        
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return history


def continual_learning_experiment():
    """Run continual learning experiment comparing EWC with standard learning."""
    print("=" * 70)
    print("Continual Learning: EWC as Fisher Flow Regularization")
    print("=" * 70)
    
    # Generate data for three tasks
    tasks = []
    for task_id in range(3):
        X, y, name = generate_task_data(task_id)
        tasks.append({'X': X, 'y': y, 'name': name, 'id': task_id})
    
    # Initialize two models: one with EWC, one without
    model_ewc = MLP(1, [10, 10, 1])
    model_standard = MLP(1, [10, 10, 1])
    
    # Copy initial weights
    for p_ewc, p_std in zip(model_ewc.parameters(), model_standard.parameters()):
        p_std.data = p_ewc.data
    
    # Initialize optimizers
    optimizer_ewc = ElasticWeightConsolidation(model_ewc.parameters(), lr=0.01, ewc_lambda=1000)
    optimizer_standard = DiagonalFisherFlow(model_standard.parameters(), lr=0.01)
    
    # Store results
    results_ewc = {'losses': [], 'evaluations': {}}
    results_standard = {'losses': [], 'evaluations': {}}
    
    # Train on tasks sequentially
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"Training on Task {task['id']}: {task['name']}")
        print(f"{'='*50}")
        
        # Train with EWC
        print("\nWith EWC (Fisher Flow Regularization):")
        loss_history_ewc = train_task(model_ewc, optimizer_ewc, 
                                      task['X'], task['y'], epochs=100)
        results_ewc['losses'].append(loss_history_ewc)
        
        # After training, consolidate knowledge (compute Fisher Information)
        if task_idx < len(tasks) - 1:  # Don't consolidate after last task
            print("  Consolidating knowledge with Fisher Information...")
            batches = create_batches(task['X'], task['y'])
            
            def loss_fn(batch, model):
                return mse_loss(batch, model)
            
            # Create a simple data loader
            class SimpleDataLoader:
                def __init__(self, batches):
                    self.batches = batches
                def __iter__(self):
                    return iter(self.batches)
            
            data_loader = SimpleDataLoader(batches)
            optimizer_ewc.consolidate(model_ewc, data_loader, loss_fn)
        
        # Train standard model
        print("\nWithout EWC (Standard Learning):")
        loss_history_std = train_task(model_standard, optimizer_standard,
                                     task['X'], task['y'], epochs=100)
        results_standard['losses'].append(loss_history_std)
        
        # Evaluate on all tasks seen so far
        print(f"\nEvaluation after Task {task_idx}:")
        print("-" * 40)
        
        for eval_task_idx in range(task_idx + 1):
            eval_task = tasks[eval_task_idx]
            
            # Evaluate EWC model
            mse_ewc, pred_ewc = evaluate_model(model_ewc, eval_task['X'], eval_task['y'])
            
            # Evaluate standard model
            mse_std, pred_std = evaluate_model(model_standard, eval_task['X'], eval_task['y'])
            
            # Store results
            key = f"task_{task_idx}_eval_{eval_task_idx}"
            results_ewc['evaluations'][key] = {'mse': mse_ewc, 'predictions': pred_ewc}
            results_standard['evaluations'][key] = {'mse': mse_std, 'predictions': pred_std}
            
            print(f"  Task {eval_task_idx} ({eval_task['name']}):")
            print(f"    EWC MSE:      {mse_ewc:.4f}")
            print(f"    Standard MSE: {mse_std:.4f}")
            print(f"    Forgetting prevented: {(mse_std - mse_ewc) / mse_std * 100:.1f}%")
    
    # Visualize results
    visualize_continual_learning(tasks, results_ewc, results_standard)
    
    return results_ewc, results_standard


def visualize_continual_learning(tasks, results_ewc, results_standard):
    """Visualize continual learning results."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, n_tasks + 1, figsize=(15, 12))
    
    # Plot predictions for each task after each training phase
    for train_task_idx in range(n_tasks):
        for eval_task_idx in range(train_task_idx + 1):
            ax = axes[train_task_idx, eval_task_idx]
            
            eval_task = tasks[eval_task_idx]
            X_test = np.linspace(eval_task['X'].min(), eval_task['X'].max(), 100)
            
            # Get predictions
            key = f"task_{train_task_idx}_eval_{eval_task_idx}"
            
            # Plot true function
            if eval_task_idx == 0:
                y_true = np.sin(X_test)
            elif eval_task_idx == 1:
                y_true = X_test**2
            else:
                y_true = np.cos(2*X_test)
            
            ax.plot(X_test, y_true, 'k-', alpha=0.3, label='True')
            
            # Plot EWC predictions
            if key in results_ewc['evaluations']:
                X_eval = eval_task['X']
                pred_ewc = results_ewc['evaluations'][key]['predictions']
                indices = np.argsort(X_eval)
                ax.plot(X_eval[indices], pred_ewc[indices], 'b-', 
                       label=f'EWC', alpha=0.7)
            
            # Plot standard predictions
            if key in results_standard['evaluations']:
                pred_std = results_standard['evaluations'][key]['predictions']
                ax.plot(X_eval[indices], pred_std[indices], 'r--', 
                       label=f'Standard', alpha=0.7)
            
            ax.set_title(f'After Task {train_task_idx}\nEval Task {eval_task_idx}', 
                        fontsize=10)
            ax.set_ylim([-2, 5])
            ax.grid(True, alpha=0.3)
            
            if train_task_idx == 0 and eval_task_idx == 0:
                ax.legend(fontsize=8)
    
    # Plot training losses
    for train_task_idx in range(n_tasks):
        ax = axes[train_task_idx, n_tasks]
        
        if train_task_idx < len(results_ewc['losses']):
            ax.plot(results_ewc['losses'][train_task_idx], 'b-', 
                   label='EWC', alpha=0.7)
            ax.plot(results_standard['losses'][train_task_idx], 'r--', 
                   label='Standard', alpha=0.7)
        
        ax.set_title(f'Training Loss\nTask {train_task_idx}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if train_task_idx == 0:
            ax.legend(fontsize=8)
    
    # Clear unused subplots
    for i in range(n_tasks):
        for j in range(n_tasks + 1):
            if j > i and j < n_tasks:
                axes[i, j].axis('off')
    
    plt.suptitle('Continual Learning: EWC (Fisher Flow) vs Standard Learning', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_fisher_information_importance():
    """Visualize how Fisher Information captures parameter importance."""
    print("\n" + "=" * 70)
    print("Fisher Information as Parameter Importance")
    print("=" * 70)
    
    # Create a simple model and train on a task
    model = MLP(1, [5, 5, 1])
    X, y, name = generate_task_data(0, n_samples=100)
    
    # Train the model
    optimizer = DiagonalFisherFlow(model.parameters(), lr=0.01)
    print(f"\nTraining on {name}...")
    train_task(model, optimizer, X, y, epochs=50, verbose=False)
    
    # Compute Fisher Information for each parameter
    print("\nComputing Fisher Information...")
    fisher_values = []
    param_names = []
    
    for i, p in enumerate(model.parameters()):
        # Accumulate squared gradients (diagonal Fisher)
        fisher_accumulator = 0
        n_samples = 0
        
        batches = create_batches(X, y, batch_size=32)
        for batch in batches:
            model.zero_grad()
            loss = mse_loss(batch, model)
            loss.backward()
            fisher_accumulator += p.grad ** 2
            n_samples += 1
        
        fisher_value = fisher_accumulator / n_samples
        fisher_values.append(fisher_value)
        param_names.append(f"Param {i}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of Fisher Information
    ax1.bar(range(len(fisher_values)), fisher_values)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Fisher Information (Importance)')
    ax1.set_title('Parameter Importance via Fisher Information')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of Fisher values
    ax2.hist(fisher_values, bins=20, edgecolor='black')
    ax2.set_xlabel('Fisher Information Value')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Parameter Importance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nFisher Information Statistics:")
    print(f"  Mean:   {np.mean(fisher_values):.6f}")
    print(f"  Std:    {np.std(fisher_values):.6f}")
    print(f"  Min:    {np.min(fisher_values):.6f}")
    print(f"  Max:    {np.max(fisher_values):.6f}")
    print(f"  Range:  {np.max(fisher_values) - np.min(fisher_values):.6f}")
    
    # Identify most and least important parameters
    sorted_indices = np.argsort(fisher_values)[::-1]
    print(f"\nMost important parameters (top 5):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {param_names[idx]}: {fisher_values[idx]:.6f}")
    
    print(f"\nLeast important parameters (bottom 5):")
    for i in range(max(0, len(sorted_indices)-5), len(sorted_indices)):
        idx = sorted_indices[i]
        print(f"  {param_names[idx]}: {fisher_values[idx]:.6f}")


if __name__ == "__main__":
    # Run continual learning experiment
    results_ewc, results_standard = continual_learning_experiment()
    
    # Visualize Fisher Information as importance
    plot_fisher_information_importance()